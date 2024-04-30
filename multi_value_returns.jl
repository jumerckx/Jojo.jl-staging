using MLIR
includet("utils.jl")
using Brutus
import Brutus: MemRef, @intrinsic, @code_mlir
using Brutus.Types
using BenchmarkTools, MLIR, MacroTools
using StaticArrays

using MLIR.Dialects: arith, index, linalg, transform, builtin
using MLIR.Dialects
using MLIR.IR
using MLIR.AffineUtils

using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

@intrinsic Base.:+(a::i64, b::i64)::i64 = i64(arith.addi(a, b))
@intrinsic Base.:-(a::i64, b::i64)::i64 = i64(arith.subi(a, b))
@intrinsic Base.:*(a::i64, b::i64)::i64 = i64(arith.muli(a, b))
@intrinsic Base.:/(a::i64, b::i64)::i64 = i64(arith.divsi(a, b))
@intrinsic Base.:>(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sgt))
@intrinsic Base.:>=(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sge))

###########################################################################################

f(a, b) = a*b

Base.code_ircode(f, Tuple{Complex{i64}, Complex{i64}})
op = Brutus.code_mlir(f, Tuple{Complex{i64}, Complex{i64}})
mod = IR.Module()
push!(IR.get_body(mod), op)
lowerModuleToLLVM(mod)
addr_f = jit(mod; opt=3)("_mlir_ciface_f")

result = Ref{Complex{Int}}();
@ccall $addr_f(result::Ref{Complex{Int}}, 1::Int, 2::Int, 3::Int, 1::Int)::Nothing
result[]

###########################################################################################

struct Point{T}
    a::T
    b::T
end
g(a, b) = Point(a, a+b)

Base.code_ircode(g, Tuple{Complex{i64}, Complex{i64}})
op = Brutus.code_mlir(g, Tuple{Complex{i64}, Complex{i64}})

mod = IR.Module()
push!(IR.get_body(mod), op)
lowerModuleToLLVM(mod)
addr_g = jit(mod; opt=3)("_mlir_ciface_g")

result = Ref{Point{Complex{Int}}}();
@ccall $addr_g(result::Ref{Point{Complex{Int}}}, 1::Int, 2::Int, 3::Int, 1::Int)::Nothing
result[]

###########################################################################################

Brutus.unpack(Point{Complex{i64}})
Brutus.unpack(Point{Complex{Int}})