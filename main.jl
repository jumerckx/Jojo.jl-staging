using MLIR
includet("utils.jl")
using Brutus
import Brutus: MemRef, @mlirfunction, @code_mlir
using Brutus.Types
using BenchmarkTools, MLIR, MacroTools

using MLIR.Dialects: arith
using MLIR.IR

using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

@mlirfunction Base.:+(a::i64, b::i64)::i64 = i64(arith.addi(a, b))
@mlirfunction Base.:>(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sge))
@mlirfunction Base.:+(a::i1, b::i1)::i64 = i64(arith.addi(a, b; result=MLIRType(i64)))
@noinline Types.i1(::Bool)::i1 = Brutus.new_intrinsic()

max(a, b) = (a>b) ? a : a+b

# Base.code_ircode(max, (Brutus.Types.i64, Brutus.Types.i64))
Brutus.code_mlir(max, Tuple{Brutus.Types.i64, Brutus.Types.i64})
