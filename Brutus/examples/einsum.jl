include("utils.jl")

import MLIR: IR, API
import Brutus
import Brutus.Library: i64, f32, tensor
import MLIR.Dialects: scf, linalg

ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

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
                
                return API.mlirAffineDimExprGet(IR.context(), length(symbols))
            end
        end
    end
    
    # function to create affinemap
    function get_map(indices)
        exprs = map(indices) do i
            symbols[i]
        end
        API.mlirAffineMapGet(
            IR.context(), length(symbols),
            0, length(indices),
            collect(exprs)
        ) |> IR.AffineMap
    end
    
    indexing_maps = IR.AffineMap[get_map.(inputs_indices)..., get_map(output_indices)]
    
    iterator_types = IR.Attribute(iterator_types)
    indexing_maps = IR.Attribute.(API.mlirAffineMapAttrGet.(indexing_maps)) |> IR.Attribute
    return indexing_maps, iterator_types
end

struct Einsum{T}
    desc::Pair{T}
    Brutus.@intrinsic function Einsum(desc::Pair{T}) where {T}
        return new{T}(desc)
    end
end
function maps(e::Einsum)
    return maps(e.desc.second, e.desc.first)
end

# methods to be specialised:
import Brutus: generate_return, generate_function

abstract type ExecuteRegion end
generate_return(cg::Brutus.CodegenContext{ExecuteRegion}, values; location) = scf.yield(values; location)
generate_function(cg::Brutus.CodegenContext{ExecuteRegion}) = Brutus.region(cg)

Brutus.@intrinsic function execute_region(f, T)
    cg = Brutus.CodegenContext{ExecuteRegion}(f, Tuple{})
    region = Brutus.generate(cg)
    T(scf.execute_region(;
        region,
        result_0=[IR.Type(T)]
    ) |> IR.result)
end
execute_region(f) = execute_region(f, only(Base.return_types(f, Tuple{}, interp=Brutus.MLIRInterpreter())))

abstract type LinalgBody end
generate_return(cg::Brutus.CodegenContext{LinalgBody}, values; location) = linalg.yield(values; location)
generate_function(cg::Brutus.CodegenContext{LinalgBody}) = Brutus.region(cg)

Brutus.generate(
    Brutus.CodegenContext{LinalgBody}(
        (xs, y)-> execute_region(i64) do 
            y+prod(xs)
        end,
        Tuple{Tuple{i64, i64}, i64}
    )
)

Brutus.@intrinsic function _einsum(E::Einsum, Y::T, XS) where {T<:tensor}
    indexing_maps, iterator_types = maps(E)
    region = Brutus.generate(Brutus.CodegenContext{LinalgBody}(
        (xs, y)-> execute_region(eltype(T)) do 
            y+prod(xs)
        end,
        Tuple{Tuple{eltype.(XS)...}, eltype(Y)}
    ))
    op = linalg.generic(
        XS,
        [Y];
        result_tensors=IR.Type[IR.Type(T)],
        indexing_maps,
        iterator_types,
        region
    )
    return T(IR.result(op))
end

function (E::Einsum)(Y, XS...)
    _einsum(E, Y, XS)
end


# op = Brutus.generate(Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}}) do Y, A, B
#     f = Einsum(((:i, :k), (:k, :j))=>(:i, :j))
#     f(Y, A, B)
# end
    
function f(Y, A, B)
    Einsum(((:i, :k), (:k, :j))=>(:i, :j))(Y, A, B)
end
op = Brutus.generate(f, Tuple{tensor{f32, 2}, tensor{f32, 2}, tensor{f32, 2}})

# running simplification gets rid of the executeregion (since there's only one block in the region)
op = Brutus.simplify(op)

@show op

mod = IR.Module()
push!(IR.body(mod), op)

# simplest possible lowering just converts the linalg.generic op to nested loops:
mlir_opt(mod, "one-shot-bufferize{bufferize-function-boundaries=true}")
mlir_opt(mod, "convert-linalg-to-loops")

lowerModuleToLLVM(mod)

addr = jit(mod)("_mlir_ciface_f")

a = rand(Float32, 1024, 1024)
b = rand(Float32, 1024, 1024)
c = zeros(Float32, 1024, 1024)
@ccall $addr(
    Brutus.MemRef(c)::Ref{Brutus.MemRef}, # first argument is the output
    Brutus.MemRef(c)::Ref{Brutus.MemRef},
    Brutus.MemRef(a)::Ref{Brutus.MemRef},
    Brutus.MemRef(b)::Ref{Brutus.MemRef})::Nothing

@assert c ≈ a*b
