module Types

export memref, i1, i8, i16, i32, i64, f32, f64, index, tensor

using MLIR
using MLIR.IR
using MLIR.IR: Value, Attribute, get_value, result, Operation, Convertible
using MLIR.API: mlirMemRefTypeGet, mlirStridedLayoutAttrGet, mlirRankedTensorTypeGet, mlirIntegerTypeGet, mlirShapedTypeGetDynamicSize, mlirF64TypeGet, mlirF32TypeGet
import MLIR.IR: MLIRValueTrait

abstract type MLIRArrayLike{T, N} <: AbstractArray{T, N} end
function Base.reinterpret(::Type{Tuple{A}}, array::A) where {A<:MLIRArrayLike}
    return (array, )
end
MLIRValueTrait(::Type{<:MLIRArrayLike}) = Convertible()
Base.show(io::IO, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")

struct MLIRMemref{T, N, Shape, Memspace, Stride, Offset} <: MLIRArrayLike{T, N}
    value::Value
end
function MLIR.IR.Type(::Type{MLIRMemref{T, N, Shape, Memspace, Stride, Offset}}) where {T, N, Shape, Memspace, Stride, Offset}
    memspace(a::IR.Attribute) = a
    memspace(::Nothing) = Attribute()
    memspace(i::Integer) = Attribute(i)

    shape(::Nothing) = Int[mlirShapedTypeGetDynamicSize() for _ in 1:N]
    shape(s) = Int[s.parameters...]

    # default to column-major layout
    stride(::Nothing) = Int[1, [mlirShapedTypeGetDynamicSize() for _ in 2:N]...]
    stride(s) = shape(s)

    offset(::Nothing) = mlirShapedTypeGetDynamicSize()
    offset(i::Integer) = i

    IR.Type(mlirMemRefTypeGet(
        IR.Type(T),
        N,
        shape(Shape),
        Attribute(mlirStridedLayoutAttrGet(
            context().context,
            offset(Offset),
            N,
            stride(Stride))),
        memspace(Memspace)
    ))

end
const memref{T, N} = MLIRMemref{T, N, nothing, nothing, nothing, 0}

struct MLIRTensor{T, N} <: MLIRArrayLike{T, N}
    value::Value
end
MLIR.IR.Type(::Type{MLIRTensor{T, N}}) where {T, N} = mlirRankedTensorTypeGet(
    N,
    Int[mlirShapedTypeGetDynamicSize() for _ in 1:N],
    IR.Type(T),
    Attribute()) |> IR.Type
const tensor = MLIRTensor

struct MLIRInteger{N} <: Integer
    value::Value
    MLIRInteger{N}(i::Value) where {N} = new(i)
    MLIRInteger{N}(i::Operation) where {N} = new(result(i))
end
MLIR.IR.Type(::Type{MLIRInteger{N}}) where {N} = IR.Type(mlirIntegerTypeGet(MLIR.IR.context(), N))
MLIRValueTrait(::Type{<:MLIRInteger}) = Convertible()

const i1 = MLIRInteger{1}

const i8 = MLIRInteger{8}
const i16 = MLIRInteger{16}
const i32 = MLIRInteger{32}
const i64 = MLIRInteger{64}

abstract type MLIRFloat <: AbstractFloat end
MLIRValueTrait(::Type{<:MLIRFloat}) = Convertible()

struct MLIRF64 <: MLIRFloat
    value::Value
end
const f64 = MLIRF64
MLIR.IR.Type(::Type{MLIRF64}) = mlirF64TypeGet(MLIR.IR.context())

struct MLIRF32 <: MLIRFloat
    value::Value
end
const f32 = MLIRF32
MLIR.IR.Type(::Type{MLIRF32}) = mlirF32TypeGet(MLIR.IR.context())

struct MLIRIndex <: Integer
    value::Value
end
const index = MLIRIndex
MLIR.IR.Type(::Type{MLIRIndex}) = MLIR.IR.IndexType()
MLIRValueTrait(::Type{<:MLIRIndex}) = Convertible()

end # Types
