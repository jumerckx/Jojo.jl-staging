using MLIR
includet("utils.jl")

using Brutus.Library: index, f32, f16, i64, memref, MLIRMemref
using Brutus.Library.GPU: threadIdx, blockIdx, blockDim, GPUFunc, gpu_module
import Brutus: MemRef, @mlirfunction, MLIRInterpreter, generate, unpack, entryblock, returntype, region, CodegenContext, simplify
using BenchmarkTools, MLIR, MacroTools

import MLIR.Dialects
using MLIR.Dialects: arith, gpu
using MLIR.IR: Context, @affinemap, Attribute, AffineMap, DenseArrayAttribute, Type, context
using MLIR.API: mlirRegisterAllPasses, mlirRegisterAllLLVMTranslations

ctx = IR.Context()
registerAllDialects!();
mlirRegisterAllPasses()
mlirRegisterAllLLVMTranslations(ctx.context)

# import Brutus
# test = Brutus.__new__(MLIRMemref{f16, 2, Tuple{16, 16}, 1, nothing, 0}, Brutus.__new__(IR.Value, Brutus.__new__(API.MlirValue, Ptr{Nothing}(0))))

import Brutus.Library.GPU: MMA_Matrix, OperandType, AOp, BOp, COp

function _mma_load(src::MLIRMemref{T, 2}, operandtype::OperandType, I::Tuple{index, index}) where {T<:Union{f32, f16}}
    I = I .- 1
    T_out = MMA_Matrix{f16, operandtype}
    return T_out(
        IR.result(Dialects.gpu.subgroup_mma_load_matrix(
            src, I;
            res=IR.Type(T_out),
            leadDimension=IR.Attribute(16, IR.Type(index)))
            )
    )
end
_mma_load(src, operandtype, I) = _mma_load(src, operandtype, (index(I[1]), index(I[2])))

@mlirfunction (mma_load_A(src::MLIRMemref{T, 2}, I=(index(1), index(1)))::MMA_Matrix{T, AOp}) where T = _mma_load(src, AOp, I)
@mlirfunction (mma_load_B(src::MLIRMemref{T, 2}, I=(index(1), index(1)))::MMA_Matrix{T, BOp}) where T = _mma_load(src, BOp, I)
@mlirfunction (mma_load_C(src::MLIRMemref{T, 2}, I=(index(1), index(1)))::MMA_Matrix{T, COp}) where T = _mma_load(src, COp, I)

@mlirfunction function mma_store(dest::D, src::S, I::Tuple{index, index}=(index(1), index(1)))::Nothing where {T, D<:MLIRMemref{T}, S<:MMA_Matrix{T}}
    I = I .- 1
    Dialects.gpu.subgroup_mma_store_matrix(
        src, dest, I;
        leadDimension=IR.Attribute(16, IR.Type(index)))
    return nothing
end
mma_store(dest, src, I) = mma_store(dest, src, (index(I[1]), index(I[2])))

@mlirfunction function mma_compute(a::A, b::B, c::C)::C where {T, A<:MMA_Matrix{T, AOp}, B<:MMA_Matrix{T, BOp}, C<:MMA_Matrix{T, COp}}
    C(
        IR.result(Dialects.gpu.subgroup_mma_compute(
            a, b, c;
            )
        )
    )
end

# @inline function f(a, b)
#     if (a > b)
#         return 2+(a+b)
#     else
#         return a*b
#     end
# end

# CodegenContext(Tuple{i64, i64}) do a, b
#     @inline f(a, b)
# end.ir


# generate(Tuple{i64, i64}) do a, b
#     f(a, b)
# end

@noinline function mma(a, b, c)
    a_mma = mma_load_A(a)
    b_mma = mma_load_B(b)
    c_mma = mma_load_C(c)
    c_mma = mma_compute(a_mma, b_mma, c_mma)
    mma_store(c, c_mma)
    return nothing
end

T_in = MLIRMemref{f16, 2, Tuple{16, 16}, nothing, Tuple{16, 1}, 0}

gpu_mod_op = gpu_module([
    IR.attr!(
        generate(CodegenContext{GPUFunc}(mma, Tuple{T_in, T_in, T_in})),
        "gpu.kernel", IR.UnitAttribute())
])  |> simplify


mod = IR.Module()
push!(IR.body(mod), gpu_mod_op)
IR.attr!(IR.Operation(mod), "gpu.container_module", IR.UnitAttribute())

# mlir_opt(mod, "gpu-lower-to-nvvm-pipeline{cubin-chip=sm_70 cubin-format=isa}")

mlir_opt(mod, "gpu.module(strip-debuginfo,convert-gpu-to-nvvm),nvvm-attach-target{chip=sm_90 O=3},gpu-to-llvm")
mlir_opt(mod, "reconcile-unrealized-casts")
data = API.mlirSerializeGPUModuleOp(gpu_mod_op)

print(String(data))
