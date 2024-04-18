module GPU

import MLIR.IR
using MLIR.IR: Value, Attribute, get_value, result, Operation, Convertible, context, IndexType, MLIRValueTrait
import MLIR.Dialects
using MLIR.API: mlirMemRefTypeGet, mlirStridedLayoutAttrGet, mlirRankedTensorTypeGet, mlirIntegerTypeGet, mlirShapedTypeGetDynamicSize, mlirF64TypeGet, mlirF32TypeGet
using Brutus: @intrinsic, Boollike, CodegenContext, unpack
import Brutus: BoolTrait, generate_return, generate_function, region, entryblock, returntype
using Brutus.Library: index, MLIRMemref, i8, i32, f16, f32

abstract type OperandType end
struct AOp <: OperandType end
struct BOp <: OperandType end
struct COp <: OperandType end

struct MMA_Matrix{T, OT}
    value::Value
end
MLIRValueTrait(::Type{<:MMA_Matrix}) = Convertible()
IR.Type(::Type{MMA_Matrix{f16, OT}}) where OT = parse(IR.Type, "!gpu.mma_matrix<16x16xf16,\"$(string(OT))\">")
IR.Type(::Type{MMA_Matrix{f32, OT}}) where OT = parse(IR.Type, "!gpu.mma_matrix<16x16xf32,\"$(string(OT))\">")

abstract type GPUFunc end
function generate_return(cg::CodegenContext{GPUFunc}, values; location)
    if (length(values) != 0)
        error("GPU kernel should return Nothing, got values of type $(typeof(values))")
    end
    return Dialects.gpu.return_(values; location)
end
function generate_function(cg::CodegenContext{GPUFunc})
    body = region(cg)
    input_types = IR.Type[
        IR.type(IR.argument(entryblock(cg), i))
        for i in 1:IR.nargs(entryblock(cg))]
    result_types = IR.Type[IR.Type.(unpack(returntype(cg)))...]
    ftype = IR.FunctionType(input_types, result_types)
    op = Dialects.gpu.func(;
        function_type=ftype,
        body
    )
    IR.attr!(op, "sym_name", IR.Attribute(replace(String(nameof(cg.f)), "#"=>"anon_f_")))
end

function gpu_module(funcs::Vector{IR.Operation}, name="gpu_module")
    block = IR.Block()
    for f in funcs
        push!(block, f)
    end
    push!(block, Dialects.gpu.module_end())
    bodyRegion = IR.Region()
    push!(bodyRegion, block)
    op = Dialects.gpu.module_(;
        bodyRegion,
    )
    IR.attr!(op, "sym_name", IR.Attribute(name))
    op
end

@intrinsic function threadIdx(dim::Symbol)::index
    oneoff = index(Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result)
    
    dimension = parse(IR.Attribute, "#gpu<dim $dim>")
    i = index(Dialects.gpu.thread_id(; dimension) |> IR.result)
    i + oneoff
end

function threadIdx()::@NamedTuple{x::index, y::index, z::index}
    (; x=threadIdx(:x), y=threadIdx(:y), z=threadIdx(:z))
end

@intrinsic function blockIdx()::@NamedTuple{x::index, y::index, z::index}
    oneoff = index(Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result)
    indices = map(('x', 'y', 'z')) do d
        dimension = parse(IR.Attribute, "#gpu<dim $d>")
        i = index(Dialects.gpu.block_id(; dimension) |> IR.result)
        i + oneoff
    end
    return (; x=indices[1], y=indices[2], z=indices[3])
end

@intrinsic function blockDim()::@NamedTuple{x::index, y::index, z::index}
    oneoff = Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result
    indices = map(('x', 'y', 'z')) do d
        dimension = parse(IR.Attribute, "#gpu<dim $d>")
        i = Dialects.gpu.block_dim(; dimension) |> IR.result
        index(Dialects.arith.addi(i, oneoff) |> IR.result)
    end
    return (; x=indices[1], y=indices[2], z=indices[3])
end


@intrinsic function mma_load(src::T, I::Tuple{index, index})::MLIRMemref{f16, 2, Tuple{16, 16}, nothing, Tuple{16, 1}, 0} where {T<:MLIRMemref{f16, 2}}
    T_out = MLIRMemref{f16, 2, Tuple{16, 16}, nothing, Tuple{16, 1}, 0}
    r = T_out(
        IR.result(Dialects.gpu.subgroup_mma_load_matrix(
            src, I;
            res=IR.Type(T_out),
            leadDimension=16)
            )
        )
    return r
end

mma_load(src, I) = mma_load(src, (index(I[1]), index(I[2])))
mma_load(src) = mma_load(src, (index(1), index(1)))

end # GPU
