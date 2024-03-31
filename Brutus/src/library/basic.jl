import MLIR.IR
using MLIR.IR: Value, Attribute, get_value, result, Operation, Convertible, context, IndexType, MLIRValueTrait
using MLIR.Dialects
using MLIR.API: mlirMemRefTypeGet, mlirStridedLayoutAttrGet, mlirRankedTensorTypeGet, mlirIntegerTypeGet, mlirShapedTypeGetDynamicSize, mlirF64TypeGet, mlirF32TypeGet
using ..Brutus: @mlirfunction

### int ###
struct MLIRInteger{N} <: Integer
    value::Value
    MLIRInteger{N}(i::Value) where {N} = new(i)
    MLIRInteger{N}(i::Operation) where {N} = new(result(i))
end
MLIRValueTrait(::Type{<:MLIRInteger}) = Convertible()
IR.Type(::Type{MLIRInteger{N}}) where {N} = IR.Type(mlirIntegerTypeGet(context(), N))

const i1 = MLIRInteger{1}
BoolTrait(::Type{<: i1}) = Boollike()

const i8 = MLIRInteger{8}
const i16 = MLIRInteger{16}
const i32 = MLIRInteger{32}
const i64 = MLIRInteger{64}

@mlirfunction (Base.:+(a::T, b::T)::T) where {T<:MLIRInteger} = T(Dialects.arith.addi(a, b)|>result)
@mlirfunction (Base.:-(a::T, b::T)::T) where {T<:MLIRInteger} = T(Dialects.arith.subi(a, b)|>result)
@mlirfunction (Base.:*(a::T, b::T)::T) where {T<:MLIRInteger} = T(Dialects.arith.muli(a, b)|>result)
@mlirfunction (Base.:/(a::T, b::T)::T) where {T<:MLIRInteger} = T(Dialects.arith.divi(a, b)|>result)

@mlirfunction (Base.:>(a::T, b::T)::i1) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, predicate=Dialects.arith.Predicates.sgt))
@mlirfunction (Base.:>=(a::T, b::T)::i1) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, predicate=Dialects.arith.Predicates.sge))
@mlirfunction (Base.:<(a::T, b::T)::i1) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, predicate=Dialects.arith.Predicates.slt))
@mlirfunction (Base.:<=(a::T, b::T)::i1) where {T<:MLIRInteger} = i1(Dialects.arith.cmpi(a, b, predicate=Dialects.arith.Predicates.sle))

### float ###
abstract type MLIRFloat <: AbstractFloat end
MLIRValueTrait(::Type{<:MLIRFloat}) = Convertible()

struct MLIRF64 <: MLIRFloat
    value::Value
end
struct MLIRF32 <: MLIRFloat
    value::Value
end

const f64 = MLIRF64
const f32 = MLIRF32

IR.Type(::Type{MLIRF64}) = mlirF64TypeGet(context())
IR.Type(::Type{MLIRF32}) = mlirF32TypeGet(context())

@mlirfunction (Base.:+(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.addf(a, b)|>result)
@mlirfunction (Base.:-(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.subf(a, b)|>result)
@mlirfunction (Base.:*(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.mulf(a, b)|>result)
@mlirfunction (Base.:/(a::T, b::T)::T) where {T<:MLIRFloat} = T(Dialects.arith.divf(a, b)|>result)

# TODO: 
# @mlirfunction Base.:>(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))
# @mlirfunction Base.:>=(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))
# @mlirfunction Base.:<(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))
# @mlirfunction Base.:<=(a::T, b::T)::i1 where {T<:MLIRFloat} = i1(Dialects.arith.cmpf(a, b, predicate=...))

# promote constant julia integers to int
Base.promote_rule(::Type{T}, ::Type{I}) where {T<:MLIRInteger, I<:Integer} = T
@mlirfunction function Base.convert(::Type{T}, x::Integer)::T where {T<:MLIRInteger}
    op = arith.constant(value=Attribute(x), result=IR.Type(T))
    T(result(op))
end

### index  ###
struct MLIRIndex <: Integer
    value::Value
end
const index = MLIRIndex
IR.Type(::Type{MLIRIndex}) = IndexType()
MLIRValueTrait(::Type{<:MLIRIndex}) = Convertible()

@mlirfunction Base.:+(a::index, b::index)::index = index(Dialects.index.add(a, b)|>result)
@mlirfunction Base.:-(a::index, b::index)::index = index(Dialects.index.sub(a, b)|>result)
@mlirfunction Base.:*(a::index, b::index)::index = index(Dialects.index.mul(a, b)|>result)
@mlirfunction Base.:/(a::index, b::index)::index = index(Dialects.index.divs(a, b)|>result)

# TODO:
# @mlirfunction Base.:>(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, predicate=...)|>result)
# @mlirfunction Base.:>=(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, predicate=...)|>result)
# @mlirfunction Base.:<(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, predicate=...)|>result)
# @mlirfunction Base.:<=(a::index, b::index)::i1 = i1(Dialects.index.cmp(a, b, predicate=...)|>result)

# promote constant julia integers to index
Base.promote_rule(::Type{index}, ::Type{I}) where {I<:Integer} = index
@mlirfunction function Base.convert(::Type{index}, x::Integer)::index
    op = Dialects.index.constant(value=Attribute(x, IR.Type(index)), result=IR.Type(index))
    index(result(op))
end

### abstract type for array-like types ###
abstract type MLIRArrayLike{T, N} <: AbstractArray{T, N} end

# implementation detail: reinterpret shouldn't try reinterpreting individual elements:
function Base.reinterpret(::Type{Tuple{A}}, array::A) where {A<:MLIRArrayLike}
    return (array, )
end
MLIRValueTrait(::Type{<:MLIRArrayLike}) = Convertible()
Base.show(io::IO, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")
Base.show(io::IO, ::MIME{Symbol("text/plain")}, a::A) where {A<:MLIRArrayLike{T, N}} where {T, N} = print(io, "$A[...]")

### memref ###
struct MLIRMemref{T, N, Shape, Memspace, Stride, Offset} <: MLIRArrayLike{T, N}
    value::Value
end
function IR.Type(::Type{MLIRMemref{T, N, Shape, Memspace, Stride, Offset}}) where {T, N, Shape, Memspace, Stride, Offset}
    memspace(a::Attribute) = a
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

Base.size(A::MLIRMemref{T, N, Shape}) where {T, N, Shape} = Tuple(Shape.parameters)

@mlirfunction function Base.getindex(A::MLIRMemref{T, 1}, i::index)::T where T
    oneoff = Dialects.index.constant(; value=Attribute(1, IndexType())) |> result
    new_index = Dialects.index.sub(i, oneoff) |> result
    T(Dialects.memref.load(A, [new_index]) |> result)
end
@mlirfunction function Base.getindex(A::MLIRMemref{T}, i::Int)::T where T
    i = index(Dialects.index.constant(; value=Attribute(i, IndexType())) |> result)
    A[i]
end

@mlirfunction function Base.setindex!(A::MLIRMemref{T, 1}, v::T, i::index)::T where T
    oneoff = Dialects.index.constant(; value=Attribute(1, IndexType())) |> result
    new_index = Dialects.index.sub(i, oneoff) |> IR.result
    Dialects.memref.store(v, A, [new_index])
    return v
end
@mlirfunction function Base.setindex!(A::MLIRMemref{T, 1}, v, i::Int)::T where {T}
    # this method can only be called with constant i since we assume arguments to `code_mlir` to be MLIR types, not Julia types.
    i = index(Dialects.index.constant(; value=Attribute(i, IndexType())) |> result)
    A[i] = v
end

### tensor ###
struct MLIRTensor{T, N} <: MLIRArrayLike{T, N}
    value::Value
end
IR.Type(::Type{MLIRTensor{T, N}}) where {T, N} = mlirRankedTensorTypeGet(
    N,
    Int[mlirShapedTypeGetDynamicSize() for _ in 1:N],
    IR.Type(T),
    Attribute()) |> IR.Type
const tensor = MLIRTensor
