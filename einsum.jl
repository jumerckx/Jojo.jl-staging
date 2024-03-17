using MLIR
includet("utils.jl")
using Brutus
import Brutus: MemRef, @mlirfunction, @code_mlir
using Brutus.Types
using Brutus.Types: MLIRFloat
using BenchmarkTools, MLIR, MacroTools

using MLIR.Dialects: arith, index, linalg, transform, builtin
using MLIR.Dialects
using MLIR.IR
using MLIR.AffineUtils

using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

@mlirfunction (Base.:+(a::T, b::T)::T) where {T <: MLIRFloat} = T(arith.addf(a, b) |> IR.get_result)
@mlirfunction (Base.:-(a::T, b::T)::T) where {T <: MLIRFloat} = T(arith.subf(a, b) |> IR.get_result)
@mlirfunction (Base.:*(a::T, b::T)::T) where {T <: MLIRFloat} = T(arith.mulf(a, b) |> IR.get_result)
@mlirfunction (Base.:/(a::T, b::T)::T) where {T <: MLIRFloat} = T(arith.divf(a, b) |> IR.get_result)

@mlirfunction Base.:+(a::i64, b::i64)::i64 = i64(arith.addi(a, b))
@mlirfunction Base.:-(a::i64, b::i64)::i64 = i64(arith.subi(a, b))
@mlirfunction Base.:*(a::i64, b::i64)::i64 = i64(arith.muli(a, b))
@mlirfunction Base.:/(a::i64, b::i64)::i64 = i64(arith.divsi(a, b))
@mlirfunction Base.:>(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sgt))
@mlirfunction Base.:>=(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sge))
@mlirfunction function Base.:+(a::T, b::T)::T where {I<:Types.MLIRInteger, T<:Union{I, tensor{I}}}
    T(IR.get_result(arith.addi(a, b)))
end
@mlirfunction function Base.:*(a::T, b::T)::T where {I<:Types.MLIRInteger, T<:Union{I, tensor{I}}}
    T(IR.get_result(arith.muli(a, b)))
end
@mlirfunction function Base.getindex(A::memref{T}, i::Int)::T where T
    # this method can only be called with constant i since we assume arguments to `code_mlir` to be MLIR types, not Julia types.
    i = Types.index(index.constant(; value=Attribute(i, IR.IndexType())) |> IR.get_result)
    A[i]
end
@mlirfunction function Base.getindex(A::memref{T, 1}, i::Types.index)::T where T
    oneoff = index.constant(; value=Attribute(1, IR.IndexType())) |> IR.get_result
    new_index = arith.subi(i, oneoff) |> IR.get_result
    T(Dialects.memref.load(A, [new_index]) |> IR.get_result)
end

@mlirfunction function linalgyield(x::T)::Nothing where {T}
    linalg.yield([x])
    return nothing
end

function maps(output_indices, inputs_indices)
    parallel, reduction = parse.(Ref(IR.Attribute), (
        "#linalg.iterator_type<parallel>",
        "#linalg.iterator_type<reduction>"
    ))

    # get all index symbols used in the inputs, the output can't contain any different symbols.
    symbols = Dict()
    iterator_types = IR.Attribute[]
    for arg in inputs_indices
        map(arg) do i
            get!(symbols, i) do 
                # indices that don't occur in output have to be reduced
                push!(iterator_types, i ∉ output_indices ? reduction : parallel)
                
                return API.mlirAffineDimExprGet(context(), length(symbols))
            end
        end
    end

    # function to create affinemap
    function get_map(indices)
        exprs = map(indices) do i
            symbols[i]
        end
        API.mlirAffineMapGet(
            context(), length(symbols),
            0, length(indices),
            # [exprs...]
            collect(exprs)
        ) |> IR.AffineMap
    end
    indexing_maps = IR.AffineMap[get_map.(inputs_indices)..., get_map(output_indices)]

    iterator_types = IR.ArrayAttribute(iterator_types)
    indexing_maps = IR.Attribute.(API.mlirAffineMapAttrGet.(indexing_maps)) |> IR.ArrayAttribute
    return indexing_maps, iterator_types
end

struct Einsum{I, O} end
@generated function maps(::Einsum{I, O}) where {I, O}
    return maps(O, I)
end
function Einsum(desc::Pair{T}) where T
    return Einsum{desc.first, desc.second}()
end

@mlirfunction function (E::Einsum{I, O})(Y::T, XS::Vararg{tensor})::T where {I, O, T<:tensor}
    indexing_maps, iterator_types = maps(E)
    region = @nonoverlay Brutus.code_mlir(
        (xs, y)->linalgyield(y+prod(xs)),
        # (y, xs)->linalgyield(y+prod(xs)),
        Tuple{Tuple{eltype.(XS)...}, eltype(Y)},
        emit_region=true, ignore_returns=true
    )
    op = linalg.generic(
        XS,
        [Y];
        result_tensors=MLIRType[MLIRType(T)],
        indexing_maps,
        iterator_types,
        region
    )
    return tensor{T, 2}(IR.get_result(op))
end

Brutus.code_mlir(Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}}) do Y, A, B
    f = Einsum(((:i, :k), (:k, :j))=>(:i, :j))
    f(Y, A, B)
end

# f(Y, A, B) = Einsum(((:i, :k), (:k, :j))=>(:i, :j))(Y, A, B)
# Brutus.code_mlir(f, Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}})

Base.code_ircode(
        (args)->linalgyield(args[end]+prod(args[1:end-1])),
        Tuple{Tuple{eltype(tensor{i64}), eltype.((tensor{i64}, tensor{i64}))...}}
    )

###########################################################################

Base.promote_rule(::Type{T}, ::Type{I}) where {T<:Brutus.Types.MLIRInteger, I<:Integer} = T

@mlirfunction function Base.convert(::Type{T}, x::Integer)::T where {T <: Brutus.Types.MLIRInteger}
    op = arith.constant(value=Attribute(x), result=MLIRType(T))
    T(IR.get_result(op))
end

Brutus.code_mlir(Tuple{i64}) do a
    a+2
end


Base.promote_rule(::Type{Brutus.Types.MLIRIndex}, ::Type{I}) where {I<:Integer} = Brutus.Types.MLIRIndex

@mlirfunction function Base.convert(::Type{T}, x::Integer)::T where {T<:Brutus.Types.MLIRIndex}
    op = index.constant(value=Attribute(x, IR.IndexType()), result=MLIRType(T))
    T(IR.get_result(op))
end
@mlirfunction function Base.:+(a::T, b::T)::T where {T<:Brutus.Types.MLIRIndex}
    T(IR.get_result(index.add(a, b)))
end

Brutus.code_mlir(Tuple{i64, Types.index}) do a, b
    (a+2, b+1)
end

Base.code_ircode(Tuple{i64, Types.index}) do a, b
    (a+2, b+1)
end

###########################################################################

Base.promote_rule(::Type{Brutus.Types.MLIRInteger{A}}, y::Type{Brutus.Types.MLIRInteger{B}}) where {A, B} = Brutus.Types.MLIRInteger{max(A, B)}

@mlirfunction function Base.convert(::Type{T}, x::Brutus.Types.MLIRInteger{X})::T where {N, T<:Brutus.Types.MLIRInteger{N}, X}
    if (N > X)
        op = arith.extsi(x, out=MLIRType(T))
    else
        @warn "Converting from $(typeof(x)) to $T will unconditionally truncate the value. This differs from conversion between Julia integers."
        op = arith.trunci(x, out=MLIRType(T))
    end
    T(IR.get_result(op))
end

Base.code_ircode(Tuple{tensor{f64, 2}, tensor{f64, 2}, tensor{f64, 2}}) do Y, A, B
    f = Einsum(((:i, :k), (:k, :j))=>(:i, :j))
    f(Y, A, B)
end

op() = Brutus.code_mlir(Tuple{tensor{f32, 2}, tensor{f32, 2}, tensor{f32, 2}}, fname="f") do Y, A, B
    f = Einsum(((:i, :k), (:k, :j))=>(:i, :j))
    f(Y, A, B)
end

