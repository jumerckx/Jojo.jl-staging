using MLIR
includet("utils.jl")
using Brutus
import Brutus: MemRef, @mlirfunction
using Brutus.Types
using BenchmarkTools, MLIR, MacroTools

using MLIR.Dialects: arith, linalg, transform, builtin, gpu
using MLIR.Dialects
using MLIR.IR
using MLIR.IR: @affinemap

using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

################################################################################################
Brutus.BoolTrait(::Type{<: i1}) = Brutus.Boollike()

@mlirfunction Base.:+(a::index, b::index)::index = index(Dialects.index.add(a, b)|>IR.result)
@mlirfunction Base.:-(a::index, b::index)::index = index(Dialects.index.sub(a, b)|>IR.result)
@mlirfunction Base.:*(a::index, b::index)::index = index(Dialects.index.mul(a, b)|>IR.result)

@mlirfunction Base.:+(a::i64, b::i64)::i64 = i64(arith.addi(a, b))
@mlirfunction Base.:-(a::i64, b::i64)::i64 = i64(arith.subi(a, b))
@mlirfunction Base.:*(a::i64, b::i64)::i64 = i64(arith.muli(a, b))
@mlirfunction Base.:/(a::i64, b::i64)::i64 = i64(arith.divsi(a, b))
@mlirfunction Base.:>(a::i64, b::i64)::i1 = i1(arith.cmpi(a, b, predicate=arith.Predicates.sgt))
@mlirfunction Base.:>=(a::i64, b::i64)::i1 = i1(arith.cmpi(a, b, predicate=arith.Predicates.sge))
@mlirfunction function Base.getindex(A::memref{T}, i::Int)::T where T
    # this method can only be called with constant i since we assume arguments to `code_mlir` to be MLIR types, not Julia types.
    i = Types.index(Dialects.index.constant(; value=Attribute(i, IR.IndexType())) |> IR.result)
    A[i]
end
@mlirfunction function Base.getindex(A::memref{T, 1}, i::Types.index)::T where T
    oneoff = Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result
    new_index = Dialects.index.sub(i, oneoff) |> IR.result
    T(Dialects.memref.load(A, [new_index]) |> IR.result)
end
@mlirfunction function Base.setindex!(A::memref{T, 1}, v, i::Int)::T where {T}
    # this method can only be called with constant i since we assume arguments to `code_mlir` to be MLIR types, not Julia types.
    i = Types.index(Dialects.index.constant(; value=Attribute(i, IR.IndexType())) |> IR.result)
    A[i] = v
end
@mlirfunction function Base.setindex!(A::memref{T, 1}, v::T, i::Types.index)::T where T
    oneoff = Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result
    new_index = Dialects.index.sub(i, oneoff) |> IR.result
    Dialects.memref.store(v, A, [new_index])
    return v
end

#############################################################################################

@mlirfunction function Types.i64(x::Int)::i64
    return i64(arith.constant(; value=Attribute(x)) |> IR.result)
end

@mlirfunction function threadIdx(dim::Symbol)::index
    oneoff = index(MLIR.Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result)
    
    dimension = parse(IR.Attribute, "#gpu<dim $dim>")
    i = index(gpu.thread_id(; dimension) |> IR.result)
    i + oneoff
end

function threadIdx()::@NamedTuple{x::index, y::index, z::index}
    (; x=threadIdx(:x), y=threadIdx(:y), z=threadIdx(:z))
end

@mlirfunction function blockIdx()::@NamedTuple{x::index, y::index, z::index}
    oneoff = MLIR.Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result
    indices = map(('x', 'y', 'z')) do d
        dimension = parse(IR.Attribute, "#gpu<dim $d>")
        i = gpu.block_id(; dimension) |> IR.result
        index(arith.addi(i, oneoff) |> IR.result)
    end
    return (; x=indices[1], y=indices[2], z=indices[3])
end

@mlirfunction function blockDim()::@NamedTuple{x::index, y::index, z::index}
    oneoff = MLIR.Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result
    indices = map(('x', 'y', 'z')) do d
        dimension = parse(IR.Attribute, "#gpu<dim $d>")
        i = gpu.block_dim(; dimension) |> IR.result
        index(arith.addi(i, oneoff) |> IR.result)
    end
    return (; x=indices[1], y=indices[2], z=indices[3])
end


Brutus.generate(Tuple{i64, i64}) do a, b
    a>b ? a : b
end


Brutus.generate(Tuple{i64, i64}) do a, b
    if (a>b)
        return a*(a+b)
    else
        return a-b
    end
end

Base.code_ircode(Tuple{i64, i64}, interp=Brutus.MLIRInterpreter()) do a, b
    a>b ? a : b
end

Brutus.generate(Tuple{}) do 
    threadIdx().x, blockDim().y
end

Base.code_ircode(Tuple{}, interp=Brutus.MLIRInterpreter()) do 
    threadIdx().x, blockDim().y
end

function vadd(a, b, c)
    i = (blockIdx().x) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end

# Base.code_ircode(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}}, interp=Brutus.MLIRInterpreter())

Brutus.generate(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}})

struct GPUCodegenContext <: Brutus.AbstractCodegenContext
    cg::Brutus.CodegenContext
end
GPUCodegenContext(f, types) = GPUCodegenContext(Brutus.CodegenContext(f, types))

for f in (
    :(Base.values),
    :(Brutus.args),
    :(Brutus.blocks),
    :(Brutus.returntype),
    :(Brutus.ir),
    :(Brutus.generate_goto),
    :(Brutus.generate_gotoifnot),
    :(Brutus.currentblock),
    :(Brutus.currentblockindex),
    :(Brutus.setcurrentblockindex!),
    :(Brutus.region),
    :(Brutus.entryblock),

    :(Brutus.generate_goto),
    :(Brutus.generate_gotoifnot))
    eval(
        quote
            function $f(cg::GPUCodegenContext, args...; kwargs...)
                $f(cg.cg, args...; kwargs...)
            end
        end
    )
end
function Brutus.generate_return(cg::GPUCodegenContext, values; location)
    if (length(values) != 0)
        error("GPU kernel should return Nothing, got values of type $(typeof(values))")
    end
    return Dialects.gpu.terminator(; location)
end

Brutus.generate(GPUCodegenContext(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}}))



@mlirfunction function scf_yield(results)::Nothing
    Dialects.scf.yield(results)
    nothing
end

@mlirfunction function scf_for(body, initial_value::T, lb::index, ub::index, step::index)::T where T
    @info "body IR" @nonoverlay Base.code_ircode(body, Tuple{index, T}, interp=Brutus.MLIRInterpreter())
    region = @nonoverlay Brutus.generate(body, Tuple{index, T}, emit_region=true, skip_return=true)
    op = Dialects.scf.for_(lb, ub, step, [initial_value]; results=IR.Type[IR.Type(T)], region)
    return T(IR.result(op))
end

Brutus.generate(Tuple{index, index, index, i64, i64}, do_simplify=false) do lb, ub, step, initial, cst

    a = scf_for(initial, lb, ub, step) do i, carry
        b = scf_for(initial, lb, ub, step) do j, carry2
            scf_yield((carry2 * cst, ))
        end
        scf_yield((b + cst, ))
    end
    return a
end
