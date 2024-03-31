using Brutus: @mlirfunction, generate, simplify
using Brutus.Library: tensor, i64
using MLIR.Dialects: linalg
using MLIR.IR: Context, Attribute, AffineMap, DenseArrayAttribute, Type, context
using MLIR.API: mlirAffineDimExprGet, mlirRegisterAllPasses, mlirRegisterAllLLVMTranslations, mlirAffineMapGet, mlirAffineMapAttrGet
includet("../../utils.jl")

ctx = Context()
registerAllDialects!()
mlirRegisterAllPasses()
mlirRegisterAllLLVMTranslations(ctx.context)

@mlirfunction function linalgyield(x::T)::Nothing where {T}
    linalg.yield([x])
    return nothing
end

function maps(output_indices, inputs_indices)
    parallel, reduction = parse.(Ref(Attribute), (
        "#linalg.iterator_type<parallel>",
        "#linalg.iterator_type<reduction>"
    ))
    
    # get all index symbols used in the inputs, the output can't contain any different symbols.
    symbols = Dict()
    iterator_types = Attribute[]
    for arg in inputs_indices
        map(arg) do i
            get!(symbols, i) do 
                # indices that don't occur in output have to be reduced
                push!(iterator_types, i ∉ output_indices ? reduction : parallel)
                
                return mlirAffineDimExprGet(context(), length(symbols))
            end
        end
    end
    
    # function to create affinemap
    function get_map(indices)
        exprs = map(indices) do i
            symbols[i]
        end
        mlirAffineMapGet(
            context(), length(symbols),
            0, length(indices),
            # [exprs...]
            collect(exprs)
        ) |> AffineMap
    end
    
    indexing_maps = AffineMap[get_map.(inputs_indices)..., get_map(output_indices)]
    
    iterator_types = Attribute(iterator_types)
    indexing_maps = Attribute.(mlirAffineMapAttrGet.(indexing_maps)) |> Attribute
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
    region = @nonoverlay generate(
        (xs, y)->linalgyield(y+prod(xs)),
        Tuple{Tuple{eltype.(XS)...}, eltype(Y)},
        emit_region=true, skip_return=true
    )
    op = linalg.generic(
        XS,
        [Y];
        result_tensors=IR.Type[Type(T)],
        indexing_maps,
        iterator_types,
        region
    )
    @info op
    return T(IR.result(op))
end

generate(Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}}) do Y, A, B
    f = Einsum(((:i, :k), (:k, :j))=>(:i, :j))
    f(Y, A, B)
end
