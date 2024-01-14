using MLIR
includet("utils.jl")
using Brutus
import Brutus: MemRef, @mlirfunction, @code_mlir
using Brutus.Types
using BenchmarkTools, MLIR, MacroTools

using MLIR.Dialects: arith, index, linalg
using MLIR.Dialects
using MLIR.IR
using MLIR.AffineUtils

using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

################################################################################################

@mlirfunction Base.:+(a::i64, b::i64)::i64 = i64(arith.addi(a, b))
@mlirfunction Base.:-(a::i64, b::i64)::i64 = i64(arith.subi(a, b))
@mlirfunction Base.:*(a::i64, b::i64)::i64 = i64(arith.muli(a, b))
@mlirfunction Base.:/(a::i64, b::i64)::i64 = i64(arith.divsi(a, b))
@mlirfunction Base.:>(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sgt))
@mlirfunction Base.:>=(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sge))
@mlirfunction function Base.getindex(A::memref{T}, i::Int)::T where T
    # this method can only be called with constant i since we assume arguments to `code_mlir`` to be MLIR types, not Julia types.
    i = Types.index(index.constant(; value=Attribute(i, IR.IndexType())) |> IR.get_result)
    A[i]
end
@mlirfunction function Base.getindex(A::memref{T, 1}, i::Types.index)::T where T
    oneoff = index.constant(; value=Attribute(1, IR.IndexType())) |> IR.get_result
    new_index = arith.subi(i, oneoff) |> IR.get_result
    T(Dialects.memref.load(A, [new_index]) |> IR.get_result)
end

square(a) = a*a
f(a, b) = (a>b) ? a+b : square(a)
g(a::AbstractVector) = a[2]
h(a, i) = a[i]

Base.code_ircode(f, (i64, i64))
@time op_f = Brutus.code_mlir(f, Tuple{i64, i64})
@assert IR.verify(op_f)

#region Running the code

# To run the code, we need to first:
#   put the operation in a MLIR module:
mod_f = IR.MModule(IR.Location())
push!(IR.get_body(mod_f), op_f);

#   lower all MLIR dialects to the llvm dialect:
pm_f = lowerModuleToLLVM(mod_f)

#   jit the function using an execution engine:
addr_f = jit(mod_f; opt=3)("_mlir_ciface_f")

# Finally, we call the function like a regular C-function:
@ccall $addr_f(3::Int, 10::Int)::Int

#endregion

# Base.code_ircode(g, (memref{i64, 1},))
op_g = Brutus.code_mlir(g, Tuple{memref{i64, 1}})
IR.verify(op_g)

op_h = Brutus.code_mlir(h, Tuple{memref{i64, 1}, Types.index})
IR.verify(op_h)

#region Running the code

mod_h = IR.MModule(IR.Location())
push!(IR.get_body(mod_h), op_h);

pm_h = lowerModuleToLLVM(mod_h)

addr_h = jit(mod_h; opt=3)("_mlir_ciface_h")

a = rand(Int, 3)

@ccall $addr_h(MemRef(a)::Ref{MemRef}, 1::Int)::Int

#endregion

################################################################################################

using MLIR: AffineUtils

@mlirfunction function linalgyield(x::T)::Nothing where {T}
    linalg.yield([x])
    return nothing
end

import LinearAlgebra.mul!
@mlirfunction function mul!(Y::tensor{T, 2}, A::tensor{T, 2}, B::tensor{T, 2})::tensor{T, 2} where T
    matmul_region = @nonoverlay Brutus.code_mlir((a, b, y)->linalgyield(y+(a*b)), Tuple{T, T, T}; emit_region=true, ignore_returns=true)
    indexing_maps = [
        (AffineUtils.@map (m, n, k)[] -> (m, k)),
        (AffineUtils.@map (m, n, k)[] -> (k, n)),
        (AffineUtils.@map (m, n, k)[] -> (m, n))
    ]
    indexing_maps = IR.Attribute.(API.mlirAffineMapAttrGet.(indexing_maps)) |> IR.ArrayAttribute
    iterator_types = IR.Attribute[parse(IR.Attribute, "#linalg.iterator_type<$type>") for type in ["parallel", "parallel", "reduction"]]
    iterator_types = IR.ArrayAttribute(iterator_types)
    op = linalg.generic(
        [A, B],
        [Y],
        result_tensors=MLIRType[MLIRType(typeof(Y))];
        indexing_maps,
        iterator_types,
        region=matmul_region
    )
    return tensor{T, 2}(IR.get_result(op))
end

f(Y, A, B) = mul!(Y, A, B)

Base.code_ircode(f, (tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}))
op = Brutus.code_mlir(f, Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}})

@assert IR.verify(op)

################################################################################################

@mlirfunction i64(i::Int)::i64 = i64(arith.constant(value=i) |> IR.get_result)

function f(a::AbstractArray)
    s = eltype(a)(0)
    for i in eachindex(IndexLinear(), a)
        s += a[i]
    end
    return s
end
Base.code_ircode(f, (memref{i64, 2},))

# compare: 
Base.code_ircode(:, (i64, i64))
Base.code_ircode(:, (Int, Int))

f() = print("non-overlaid")
@overlay MyMethodTable f() = print("overlaid")

function g()
    mypasss() do 
        f()
    end
end
