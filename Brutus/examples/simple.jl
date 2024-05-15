include("utils.jl")

import MLIR: IR, API
import Brutus
import Brutus.Library: i64

ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

f(a, b) = a>b ? a : a+b

# To see the Julia SSA IR from which the MLIR is generated:
display(Brutus.CodegenContext(f, Tuple{i64, i64}).ir)

op = Brutus.generate(f, Tuple{i64, i64});
display(op)

mod = IR.Module();
push!(IR.body(mod), op)

lowerModuleToLLVM(mod);
display(mod)

addr = jit(mod)("_mlir_ciface_f")
f_mlir(a, b) = @ccall $addr(a::Int, b::Int)::Int

@assert f_mlir(1, 2) == f(1, 2)
@assert f_mlir(10, 2) == f(10, 2)
