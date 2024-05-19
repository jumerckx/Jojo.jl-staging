using MLIR
includet("utils.jl")

using Jojo.Library: index, f32, f16, i64, memref, MLIRMemref
using Jojo.Library.GPU: threadIdx, blockIdx, blockDim, GPUFunc, gpu_module
import Jojo: MemRef, @intrinsic, MLIRInterpreter, generate, unpack, entryblock, returntype, region, CodegenContext, simplify
using BenchmarkTools, MLIR, MacroTools

import MLIR.Dialects
using MLIR.Dialects: arith, gpu, linalg, scf
using MLIR.IR: Context, @affinemap, Attribute, AffineMap, DenseArrayAttribute, Type, context
using MLIR.API: mlirRegisterAllPasses, mlirRegisterAllLLVMTranslations

ctx = IR.Context()
registerAllDialects!();
mlirRegisterAllPasses()
mlirRegisterAllLLVMTranslations(ctx.context)

import LinearAlgebra: mul!

import Jojo: generate_return, generate_function, region, CodegenContext

abstract type LinalgBody end
generate_return(cg::CodegenContext{LinalgBody}, values; location) = linalg.yield(values; location)
generate_function(cg::CodegenContext{LinalgBody}) = region(cg)

@intrinsic function mul!(c::MLIRMemref{T}, a::MLIRMemref{T}, b::MLIRMemref{T})::MLIRMemref{T} where {T}
    reg = generate(CodegenContext{LinalgBody}((a, b, c) -> c + a*b, Tuple{T, T, T}))
    result_tensors = IR.Type[]
    linalg.matmul([a, b], [c]; result_tensors, region=reg)
    return c
end

import Core.Compiler
const CC = Core.Compiler

generate(Tuple{memref{f32, 2}, memref{f32, 2}, memref{f32, 2}}) do a, b, c
    mul!(c, a, b)
    return nothing
end

CodegenContext(Tuple{memref{f32, 2}, memref{f32, 2}, memref{f32, 2}}) do a, b, c
    return mul!(c, a, b)[1]
    return nothing
end.ir

const MMAOperand{T, S} = MLIRMemref{T, 2, S, 1, Tuple{16, 1}, 0}

generate(CodegenContext{GPUFunc}(Tuple{MMAOperand{f16, Tuple{16,16}}, MMAOperand{f16, Tuple{16,8}}, MMAOperand{f16, Tuple{16,8}}}) do a, b, c
    mul!(c, a, b)
    return nothing
end)

CodegenContext{GPUFunc}(Tuple{memref{f32, 2}, memref{f32, 2}, memref{f32, 2}}) do a, b, c
    mul!(c, a, b)
    return nothing
end |> generate

@intrinsic assume_alignment(m::MLIRMemref) = begin
    MLIR.Dialects.memref.assume_alignment(m, alignment=Int32(128))
    nothing
end

gpu_mod_op = gpu_module([
    IR.attr!(
        generate(CodegenContext{GPUFunc}(Tuple{MMAOperand{f16, Tuple{16,16}}, MMAOperand{f16, Tuple{16,8}}, MMAOperand{f16, Tuple{16,8}}}) do a, b, c
            assume_alignment(a)
            assume_alignment(b)
            assume_alignment(c)
            mul!(c, a, b)
            return nothing
        end),
        "gpu.kernel", IR.UnitAttribute()
    )
])

transform_mod = parse(IR.Module, """
module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
        transform.nvgpu.rewrite_matmul_as_mma_sync %matmul
        : (!transform.any_op) -> ()
        transform.yield
    }
}
""")

# gpu_mod_op = parse(IR.Module, """
# gpu.module @gpu_module {
#   func.func @f(%a: i64, %b: i64) -> i64 {
#     %c = arith.addi %a, %b : i64
#     return %c : i64
#   }
#   gpu.func @"abc"(%arg0: memref<16x16xf16, strided<[16, 1]>>, %arg1: memref<16x8xf16, strided<[16, 1]>>, %arg2: memref<16x8xf16, strided<[16, 1]>>) kernel {
#     %a = arith.constant 0 : i64
#     func.call @f(%a, %a) : (i64, i64) -> i64
#     linalg.matmul ins(%arg0, %arg1 : memref<16x16xf16, strided<[16, 1]>>, memref<16x8xf16, strided<[16, 1]>>) outs(%arg2 : memref<16x8xf16, strided<[16, 1]>>)
#     gpu.return
#   }
# }
# """) |> IR.Operation

mod = IR.Module()
# gpu_mod_op_copy = copy(first(IR.OperationIterator(first(IR.BlockIterator(first(IR.RegionIterator(gpu_mod_op)))))))
gpu_mod_op_copy = copy(gpu_mod_op)
push!(IR.body(mod), gpu_mod_op_copy)
push!(IR.body(mod), copy(IR.Operation(transform_mod)))
IR.attr!(IR.Operation(mod), "gpu.container_module", IR.UnitAttribute())

# mlir_opt(mod, "gpu-lower-to-nvvm-pipeline{cubin-chip=sm_90 cubin-format=isa}")
mlir_opt(mod, "transform-interpreter")
# mlir_opt(mod, "test-transform-dialect-erase-schedule")
mlir_opt(mod, "convert-nvgpu-to-nvvm,convert-vector-to-scf,convert-scf-to-cf,convert-nvvm-to-llvm,convert-func-to-llvm,expand-strided-metadata")
mlir_opt(mod, "convert-vector-to-scf")
mlir_opt(mod, "lower-affine,convert-arith-to-llvm,convert-index-to-llvm,canonicalize,cse")
mlir_opt(mod, "gpu.module(strip-debuginfo,convert-gpu-to-nvvm),nvvm-attach-target{chip=sm_90 O=3}")
mlir_opt(mod, "reconcile-unrealized-casts")
mlir_opt(mod, "gpu-to-llvm")

data = API.mlirSerializeGPUModuleOp(gpu_mod_op_copy)

print(String(data))

