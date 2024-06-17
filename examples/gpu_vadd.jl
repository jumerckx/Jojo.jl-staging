include("utils.jl")

import MLIR: IR, API
import Jojo
using Jojo.Library.GPU: threadIdx, blockIdx, blockDim, GPUFunc, gpu_module
import Jojo.Library: f32, MLIRMemref

ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

function vadd(a, b, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end

T_in = MLIRMemref{f32, 1, nothing, nothing, nothing, 0}
Base.code_ircode(vadd, Tuple{T_in, T_in, T_in}, interp=Jojo.MLIRInterpreter())

gpu_mod_op = gpu_module([
    IR.attr!(
        Jojo.generate(Jojo.CodegenContext{GPUFunc}(vadd, Tuple{T_in, T_in, T_in})),
        "gpu.kernel", IR.UnitAttribute()),
    ]) |> Jojo.simplify

mod = IR.Module()
push!(IR.body(mod), gpu_mod_op)
IR.attr!(IR.Operation(mod), "gpu.container_module", IR.UnitAttribute())

# mlir_opt(mod, "gpu.module(strip-debuginfo,convert-gpu-to-rocdl), rocdl-attach-target,gpu-to-llvm")
mlir_opt(mod, "gpu.module(strip-debuginfo,convert-gpu-to-nvvm),nvvm-attach-target,gpu-to-llvm")
mlir_opt(mod, "reconcile-unrealized-casts")

gpu_mod_op

data = API.mlirSerializeGPUModuleOp(gpu_mod_op)
    
print(String(data))

import CUDA: CuPtr, CuModule, CuFunction, CuArray, cudacall

md = CuModule(data.data)
vadd_cu = CuFunction(md, "vadd")

a = rand(Float32, 10)
b = rand(Float32, 10)
a_d = CuArray(a)
b_d = CuArray(b)

c = zeros(Float32, 10)
c_d = CuArray(c)
null = CuPtr{Cfloat}(0);
cudacall(vadd_cu,
            (CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat},
            CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat},
            CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}),
            null, a_d, null, null, null,
            null, b_d, null, null, null,
            null, c_d, null, null, null;
            threads=10)
c = Array(c_d)
c ≈ a+b

