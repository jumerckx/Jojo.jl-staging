using Brutus: @intrinsic, generate, simplify
using Brutus.Library: tensor, i64
using MLIR.Dialects: linalg, scf
using MLIR.IR: Context, Attribute, AffineMap, DenseArrayAttribute, Type, context
using MLIR.API: mlirAffineDimExprGet, mlirRegisterAllPasses, mlirRegisterAllLLVMTranslations, mlirAffineMapGet, mlirAffineMapAttrGet
includet("../../utils.jl")

ctx = Context()
registerAllDialects!()
mlirRegisterAllPasses()
mlirRegisterAllLLVMTranslations(ctx.context)


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

struct Einsum{T}
    desc::Pair{T}
    @intrinsic function Einsum(desc::Pair{T}) where {T}
        return new{T}(desc)
    end
end
function maps(e::Einsum)
    return maps(e.desc.second, e.desc.first)
end

import Brutus: generate_return, generate_function, region, CodegenContext

abstract type ExecuteRegion end
generate_return(cg::CodegenContext{ExecuteRegion}, values; location) = scf.yield(values; location)
generate_function(cg::CodegenContext{ExecuteRegion}) = region(cg)

@intrinsic function execute_region(f, T)
    cg = CodegenContext{ExecuteRegion}(f, Tuple{})
    region = generate(cg)
    T(scf.execute_region(;
        region,
        result_0=[IR.Type(T)]
    ) |> IR.result)
end
execute_region(f) = execute_region(f, only(Base.return_types(f, Tuple{}, interp=Brutus.MLIRInterpreter())))

abstract type LinalgBody end
generate_return(cg::CodegenContext{LinalgBody}, values; location) = linalg.yield(values; location)
generate_function(cg::CodegenContext{LinalgBody}) = region(cg)

generate(
    CodegenContext{LinalgBody}(
        (xs, y)-> execute_region(i64) do 
            y+prod(xs)
        end,
        Tuple{Tuple{i64, i64}, i64}
    )
)

@intrinsic function _einsum(E::Einsum, Y::T, XS) where {T<:tensor}
    indexing_maps, iterator_types = maps(E)
    region = generate(CodegenContext{LinalgBody}(
        (xs, y)-> execute_region(i64) do 
            y+prod(xs)
        end,
        Tuple{Tuple{eltype.(XS)...}, eltype(Y)}
    ))
    op = linalg.generic(
        XS,
        [Y];
        result_tensors=IR.Type[Type(T)],
        indexing_maps,
        iterator_types,
        region
    )
    return T(IR.result(op))
end

function (E::Einsum)(Y, XS...)
    _einsum(E, Y, XS)
end

generate(Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}}) do Y, A, B
    f = Einsum(((:i, :k), (:k, :j))=>(:i, :j))
    f(Y, A, B)
end |> simplify
