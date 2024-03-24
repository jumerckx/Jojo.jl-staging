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

@mlirfunction function threadIdx()::@NamedTuple{x::index, y::index, z::index}
    oneoff = index(MLIR.Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result)
    indices = map(('x', 'y', 'z')) do d
        dimension = parse(IR.Attribute, "#gpu<dim $d>")
        i = index(gpu.thread_id(; dimension) |> IR.result)
        i + oneoff
    end
    return (; x=indices[1], y=indices[2], z=indices[3])
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

Brutus.code_mlir(Tuple{i64, i64}) do a, b
    a>b ? a : b
end

Base.code_ircode(Tuple{i64, i64}, interp=Brutus.MLIRInterpreter()) do a, b
    a>b ? a : b
end

Brutus.code_mlir(Tuple{}) do 
    threadIdx().x, blockDim().y
end

function vadd(a, b, c)
    i = (blockIdx().x) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end

# Base.code_ircode(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}}, interp=Brutus.MLIRInterpreter())

Brutus.code_mlir(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}})
