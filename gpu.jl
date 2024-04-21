using MLIR
includet("utils.jl")

using Brutus.Library: index, f32, i64, memref, MLIRMemref
using Brutus.Library.GPU: threadIdx, blockIdx, blockDim, GPUFunc, gpu_module
import Brutus: MemRef, @intrinsic, MLIRInterpreter, generate, unpack, entryblock, returntype, region, CodegenContext, simplify
using BenchmarkTools, MLIR, MacroTools

import MLIR.Dialects
using MLIR.Dialects: arith, gpu
using MLIR.IR: Context, @affinemap, Attribute, AffineMap, DenseArrayAttribute, Type, context
using MLIR.API: mlirRegisterAllPasses, mlirRegisterAllLLVMTranslations

ctx = IR.Context()
registerAllDialects!();
mlirRegisterAllPasses()
mlirRegisterAllLLVMTranslations(ctx.context)

function vadd(a, b, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end

Base.code_ircode(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}}, interp=MLIRInterpreter())

@time generate(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}})

T_in = MLIRMemref{f32, 1, nothing, nothing, nothing, 0}
@time generate(vadd, Tuple{T_in, T_in, T_in})

generate(CodegenContext{GPUFunc}(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}}))

gpu_mod_op = gpu_module([
    IR.attr!(
        generate(CodegenContext{GPUFunc}(vadd, Tuple{T_in, T_in, T_in})),
        "gpu.kernel", IR.UnitAttribute()),
    ]) |> simplify

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

nothing

# open("compiled.out", "w") do file
#     write(file, Base.unsafe_wrap(Vector{Int8}, pointer(data.data), Int(data.length), own=false))
# end

# @intrinsic function scf_yield(results)::Nothing
#     Dialects.scf.yield(results)
#     nothing
# end

# @intrinsic function scf_for(body, initial_value::T, lb::index, ub::index, step::index)::T where T
#     @info "body IR" @nonoverlay Base.code_ircode(body, Tuple{index, T}, interp=Brutus.MLIRInterpreter())
#     region = @nonoverlay generate(body, Tuple{index, T}, emit_region=true, skip_return=true)
#     op = Dialects.scf.for_(lb, ub, step, [initial_value]; results=IR.Type[IR.Type(T)], region)
#     return T(IR.result(op))
# end

# generate(Tuple{index, index, index, i64, i64}, do_simplify=false) do lb, ub, step, initial, cst

#     a = scf_for(initial, lb, ub, step) do i, carry
#         b = scf_for(initial, lb, ub, step) do j, carry2
#             scf_yield((carry2 * cst, ))
#         end
#         scf_yield((b + cst, ))
#     end
#     return a
# end

