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

@mlirfunction Base.:+(a::f32, b::f32)::f32 = f32(arith.addf(a, b)|>IR.result)

@mlirfunction Base.:+(a::index, b::index)::index = index(Dialects.index.add(a, b)|>IR.result)
@mlirfunction Base.:-(a::index, b::index)::index = index(Dialects.index.sub(a, b)|>IR.result)
@mlirfunction Base.:*(a::index, b::index)::index = index(Dialects.index.mul(a, b)|>IR.result)

@mlirfunction Base.:+(a::i64, b::i64)::i64 = i64(arith.addi(a, b))
@mlirfunction Base.:-(a::i64, b::i64)::i64 = i64(arith.subi(a, b))
@mlirfunction Base.:*(a::i64, b::i64)::i64 = i64(arith.muli(a, b))
@mlirfunction Base.:/(a::i64, b::i64)::i64 = i64(arith.divsi(a, b))
@mlirfunction Base.:>(a::i64, b::i64)::i1 = i1(arith.cmpi(a, b, predicate=arith.Predicates.sgt))
@mlirfunction Base.:>=(a::i64, b::i64)::i1 = i1(arith.cmpi(a, b, predicate=arith.Predicates.sge))
@mlirfunction function Base.getindex(A::Types.MLIRMemref{T}, i::Int)::T where T
    # this method can only be called with constant i since we assume arguments to `code_mlir` to be MLIR types, not Julia types.
    i = Types.index(Dialects.index.constant(; value=Attribute(i, IR.IndexType())) |> IR.result)
    A[i]
end
@mlirfunction function Base.getindex(A::Types.MLIRMemref{T, 1}, i::Types.index)::T where T
    oneoff = Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result
    new_index = Dialects.index.sub(i, oneoff) |> IR.result
    T(Dialects.memref.load(A, [new_index]) |> IR.result)
end
@mlirfunction function Base.setindex!(A::Types.MLIRMemref{T, 1}, v, i::Int)::T where {T}
    # this method can only be called with constant i since we assume arguments to `code_mlir` to be MLIR types, not Julia types.
    i = Types.index(Dialects.index.constant(; value=Attribute(i, IR.IndexType())) |> IR.result)
    A[i] = v
end
@mlirfunction function Base.setindex!(A::Types.MLIRMemref{T, 1}, v::T, i::Types.index)::T where T
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
    oneoff = index(MLIR.Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result)
    indices = map(('x', 'y', 'z')) do d
        dimension = parse(IR.Attribute, "#gpu<dim $d>")
        i = index(gpu.block_id(; dimension) |> IR.result)
        i + oneoff
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


Base.promote_rule(::Type{T}, ::Type{I}) where {T<:Brutus.Types.MLIRInteger, I<:Integer} = T

@mlirfunction function Base.convert(::Type{T}, x::Integer)::T where {T <: Brutus.Types.MLIRInteger}
    op = arith.constant(value=Attribute(x), result=IR.Type(T))
    T(IR.result(op))
end

Base.code_ircode(Tuple{i64}, interp=Brutus.MLIRInterpreter()) do a
    a+2
end

Brutus.generate(Tuple{i64}) do a
    Base.convert(i64, 2)
    # a+2
end


Base.promote_rule(::Type{T}, ::Type{I}) where {T<:Types.index, I<:Integer} = Types.index
@mlirfunction function Base.convert(::Type{Types.index}, x::Integer)::Types.index
    # @info "converting $x to type index"
    op = Dialects.index.constant(value=Attribute(x, IR.Type(Types.index)), result=IR.Type(Types.index))
    Types.index(IR.result(op))
end

Base.code_ircode(Tuple{Types.index, Int}, interp=Brutus.MLIRInterpreter()) do a, b
    Base.convert(Types.index, 2)
    # Base.promote(a, b)
    a-b
end


function vadd(a, b, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end

Base.code_ircode(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}}, interp=Brutus.MLIRInterpreter())

@time Brutus.generate(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}})

T_in = Brutus.Types.MLIRMemref{f32, 1, nothing, 1, nothing, 0}
@time Brutus.generate(vadd, Tuple{T_in, T_in, T_in})

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
    return Dialects.gpu.return_(values; location)
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

function gpu_func(f, args)
    cg = GPUCodegenContext(f, args)
    region = Brutus.generate(cg, emit_region=true)
    input_types = IR.Type[
            IR.type(IR.argument(Brutus.entryblock(cg), i))
            for i in 1:IR.nargs(Brutus.entryblock(cg))]
    result_types = IR.Type[IR.Type.(Brutus.unpack(Brutus.returntype(cg)))...]
    ftype = IR.FunctionType(input_types, result_types)
    op = Dialects.gpu.func(;
        function_type=ftype,
        body=region
    )
    IR.attr!(op, "sym_name", IR.Attribute("test"))
end

function gpu_module(funcs::Vector{IR.Operation})
    block = IR.Block()
    for f in funcs
        push!(block, f)
    end
    push!(block, Dialects.gpu.module_end())
    bodyRegion = IR.Region()
    push!(bodyRegion, block)
    op = Dialects.gpu.module_(;
        bodyRegion,
    )
    IR.attr!(op, "sym_name", IR.Attribute("test"))
    op
end

op = gpu_module([
    IR.attr!(
        gpu_func(vadd, Tuple{T_in, T_in, T_in})
        , "gpu.kernel", IR.UnitAttribute()),
    ]) |> Brutus.simplify

mod = IR.Module()
push!(IR.body(mod), op)
attr!(IR.Operation(mod), "gpu.container_module", IR.UnitAttribute())
IR.Operation(mod) |> Brutus.simplify


mod

mlir_opt(mod, "gpu.module(strip-debuginfo,convert-gpu-to-nvvm),nvvm-attach-target,gpu-to-llvm")
mlir_opt(mod, "reconcile-unrealized-casts")
op = first(IR.OperationIterator(first(IR.BlockIterator(first(IR.RegionIterator(IR.Operation(mod)))))))

data = API.mlirSerializeGPUModuleOp(op)

print(String(data))

using CUDA

md = CUDA.CuModule(data.data)
vadd_cu = CuFunction(md, "test")

a = rand(Float32, 10)
b = rand(Float32, 10)
ad = CuArray(a)
bd = CuArray(b)

c = zeros(Float32, 10)
c_d = CuArray(c)
null = CuPtr{Cfloat}(0);
cudacall(vadd_cu,
            (CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat},
            CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat},
            CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}),
            null, ad, null, null, null,
            null, bd, null, null, null,
            null, c_d, null, null, null;
            threads=10)
c = Array(c_d)
c ≈ a+b


# open("compiled.out", "w") do file
#     write(file, Base.unsafe_wrap(Vector{Int8}, pointer(data.data), Int(data.length), own=false))
# end
