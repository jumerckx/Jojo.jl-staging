include("utils.jl")

import MLIR: IR, API, Dialects
import Jojo
import Jojo.Library.GPU: GPUFunc, gpu_module
import Jojo.Library: MLIRMemref, f16
import MLIR.Dialects: linalg

ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

import LinearAlgebra: mul!

import Jojo: generate_return, generate_function, region, CodegenContext

abstract type LinalgBody end
generate_return(cg::Jojo.CodegenContext{LinalgBody}, values; location) = linalg.yield(values; location)
generate_function(cg::Jojo.CodegenContext{LinalgBody}) = region(cg)

Jojo.@intrinsic function mul!(c::MLIRMemref{T}, a::MLIRMemref{T}, b::MLIRMemref{T})::MLIRMemref{T} where {T}
    reg = Jojo.generate(Jojo.CodegenContext{LinalgBody}((a, b, c) -> c + a*b, Tuple{T, T, T}))
    result_tensors = IR.Type[]
    linalg.matmul([a, b], [c]; result_tensors, region=reg)
    return c
end

Jojo.@intrinsic assume_alignment(m::MLIRMemref) = begin
    Dialects.memref.assume_alignment(m, alignment=Int32(128))
    nothing
end

const T{N, M} = MLIRMemref{f16, 2, Tuple{N, M}, 1, Tuple{N, 1}, 0}

gpu_mod_op = gpu_module([
    IR.attr!(
        Jojo.generate(Jojo.CodegenContext{GPUFunc}(Tuple{T{16,16}, T{16,8}, T{16,8}}) do a, b, c
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

mod = IR.Module()
gpu_mod_op_copy = copy(gpu_mod_op)
push!(IR.body(mod), gpu_mod_op_copy)
push!(IR.body(mod), copy(IR.Operation(transform_mod)))
IR.attr!(IR.Operation(mod), "gpu.container_module", IR.UnitAttribute())

mlir_opt(mod, "transform-interpreter")
mlir_opt(mod, "convert-nvgpu-to-nvvm,convert-vector-to-scf,convert-scf-to-cf,convert-nvvm-to-llvm,convert-func-to-llvm,expand-strided-metadata")
mlir_opt(mod, "convert-vector-to-scf")
mlir_opt(mod, "lower-affine,convert-arith-to-llvm,convert-index-to-llvm,canonicalize,cse")
mlir_opt(mod, "gpu.module(strip-debuginfo,convert-gpu-to-nvvm),nvvm-attach-target{chip=sm_90 O=3}")
mlir_opt(mod, "reconcile-unrealized-casts")
mlir_opt(mod, "gpu-to-llvm")

data = API.mlirSerializeGPUModuleOp(gpu_mod_op_copy)

print(String(data))
#=
//
// Generated by LLVM NVPTX Back-End
//

.version 7.8
.target sm_90
.address_size 64

        // .globl       anon_f_29

.visible .entry anon_f_29(
        .param .u64 anon_f_29_param_0,
        .param .u64 anon_f_29_param_1,
        .param .u64 anon_f_29_param_2,
        .param .u64 anon_f_29_param_3,
        .param .u64 anon_f_29_param_4,
        .param .u64 anon_f_29_param_5,
        .param .u64 anon_f_29_param_6,
        .param .u64 anon_f_29_param_7,
        .param .u64 anon_f_29_param_8,
        .param .u64 anon_f_29_param_9,
        .param .u64 anon_f_29_param_10,
        .param .u64 anon_f_29_param_11,
        .param .u64 anon_f_29_param_12,
        .param .u64 anon_f_29_param_13,
        .param .u64 anon_f_29_param_14,
        .param .u64 anon_f_29_param_15,
        .param .u64 anon_f_29_param_16,
        .param .u64 anon_f_29_param_17,
        .param .u64 anon_f_29_param_18,
        .param .u64 anon_f_29_param_19,
        .param .u64 anon_f_29_param_20
)
{
        .reg .b16       %rs<5>;
        .reg .b32       %r<19>;
        .reg .b64       %rd<20>;

        ld.param.u64    %rd1, [anon_f_29_param_1];
        mov.u32         %r1, %tid.x;
        shr.s32         %r2, %r1, 31;
        xor.b32         %r3, %r2, %r1;
        shr.s32         %r4, %r3, 31;
        shr.u32         %r5, %r4, 30;
        add.s32         %r6, %r3, %r5;
        shr.s32         %r7, %r6, 2;
        xor.b32         %r8, %r7, %r2;
        mul.wide.s32    %rd2, %r1, 2;
        ld.param.u64    %rd3, [anon_f_29_param_8];
        mul.wide.s32    %rd4, %r8, 8;
        sub.s64         %rd5, %rd2, %rd4;
        mul.wide.s32    %rd6, %r8, 16;
        add.s64         %rd7, %rd5, %rd6;
        shl.b64         %rd8, %rd7, 1;
        add.s64         %rd9, %rd1, %rd8;
        ld.global.b32   %r9, [%rd9];
        ld.param.u64    %rd10, [anon_f_29_param_15];
        ld.global.b32   %r10, [%rd9+256];
        shl.b64         %rd11, %rd6, 1;
        add.s64         %rd12, %rd1, %rd11;
        shl.b64         %rd13, %rd5, 1;
        add.s64         %rd14, %rd12, %rd13;
        ld.global.b32   %r11, [%rd14+16];
        ld.global.b32   %r12, [%rd14+272];
        shl.b64         %rd15, %rd5, 5;
        add.s64         %rd16, %rd3, %rd15;
        mul.wide.s32    %rd17, %r8, 2;
        add.s64         %rd18, %rd16, %rd17;
        ld.global.b16   %rs1, [%rd18];
        ld.global.b16   %rs2, [%rd18+32];
        ld.global.b16   %rs3, [%rd18+256];
        ld.global.b16   %rs4, [%rd18+288];
        mov.b32         %r13, {%rs1, %rs2};
        mov.b32         %r14, {%rs3, %rs4};
        add.s64         %rd19, %rd10, %rd8;
        ld.global.b32   %r15, [%rd19];
        ld.global.b32   %r16, [%rd19+256];
        mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
                {%r17, %r18},
                {%r9, %r10, %r11, %r12},
                {%r13, %r14},
                {%r15, %r16};
        st.global.b32   [%rd19], %r17;
        st.global.b32   [%rd19+256], %r18;
        ret;

}
=#