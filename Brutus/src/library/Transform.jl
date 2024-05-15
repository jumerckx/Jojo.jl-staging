module Transform

import MLIR: IR, API
using MLIR.IR: Value, Attribute, get_value, result, Operation, Convertible, context, IndexType, MLIRValueTrait
import MLIR.Dialects: transform
using MLIR.API: mlirMemRefTypeGet, mlirStridedLayoutAttrGet, mlirRankedTensorTypeGet, mlirIntegerTypeGet, mlirShapedTypeGetDynamicSize, mlirF64TypeGet, mlirF32TypeGet
using Brutus: @intrinsic, Boollike, CodegenContext, unpack
import Brutus: BoolTrait, generate_return, generate_function, region, entryblock, returntype, generate

struct AnyOp
    value::IR.Value
end
IR.MLIRValueTrait(::Type{<:AnyOp}) = IR.Convertible()
IR.Type(::Type{AnyOp}) = IR.parse(IR.Type, "!transform.any_op")

abstract type NoReturn end

function generate_return(cg::CodegenContext{NoReturn}, values; location)
    if (length(values) != 0)
        error("Expected nothing to be returned, got values of type $(typeof(values))")
    end
    nothing
end
function generate_function(cg::CodegenContext{NoReturn})
    region(cg)
end

@intrinsic function apply_patterns(f, target)
    patterns = generate(CodegenContext{NoReturn}(f, Tuple{}))
    transform.apply_patterns(
        target;
        patterns
    )
end

abstract type MatchInterface end
abstract type TilingInterface <: MatchInterface end
abstract type LoopLikeInterface <: MatchInterface end

@intrinsic function _structured_match(parent::AnyOp, name::Union{Nothing, String}, interface::Union{Nothing, Type{<:MatchInterface}})
    if !isnothing(interface)
        if interface == TilingInterface
            interface = 1
        elseif interface == LoopLikeInterface
            interface = 2
        end
    end

    return IR.result(transform.structured_match(
        parent;
        results = IR.Type(AnyOp),
        ops = isnothing(name) ? name : IR.Attribute([IR.Attribute(name)]),
        interface = isnothing(interface) ? interface : IR.Attribute(Int32(interface))
    )) |> AnyOp
end
function structured_match(parent; name=nothing, interface=nothing)
    _structured_match(parent, name, interface)
end

@intrinsic function structured_tile_using_for(op::AnyOp, tilesizes::NTuple{N, Int}) where N
    loops = fill(IR.Type(AnyOp), count(tilesizes != 0))
    static_sizes = IR.Attribute(API.mlirDenseI64ArrayGet(IR.context(), length(tilesizes), [tilesizes...]))
    scalable_sizes = IR.Attribute(API.mlirDenseBoolArrayGet(IR.context(), length(tilesizes), fill(Int32(false), length(tilesizes))))
    op = transform.structured_tile_using_for(
        op, [];
        tiled_linalg_op = IR.Type(AnyOp),
        loops,
        static_sizes,
        scalable_sizes
    )
    return AnyOp(IR.result(op, 1))
end
@intrinsic function structured_tile_using_forall(target::AnyOp, tilesizes::NTuple{N, Int}) where N
    tiled_op = IR.Type(AnyOp)
    forall_op = IR.Type(AnyOp)
    static_tile_sizes = IR.Attribute(API.mlirDenseI64ArrayGet(IR.context(), length(tilesizes), [tilesizes...]))
    op = transform.structured_tile_using_forall(
        target, [], [];
        tiled_op, forall_op,
        static_tile_sizes,
    )
    return AnyOp(IR.result(op, 1)), AnyOp(IR.result(op, 2))
end
@intrinsic function structured_tile_reduction_using_for(target::AnyOp, tilesizes::NTuple{N, Int}) where N
    fill_op = IR.Type(AnyOp)
    split_linalg_op = IR.Type(AnyOp)
    combining_linalg_op = IR.Type(AnyOp)
    for_op = IR.Type(AnyOp)
    tile_sizes = IR.Attribute(API.mlirDenseI64ArrayGet(IR.context(), length(tilesizes), [tilesizes...]))
    op = transform.structured_tile_reduction_using_for(
        target;
        fill_op, split_linalg_op, combining_linalg_op, for_op,
        tile_sizes
    )
    return AnyOp(IR.result(op, 1)), AnyOp(IR.result(op, 2)), AnyOp(IR.result(op, 3)), AnyOp(IR.result(op, 4))
end

abstract type TileStrategy end
abstract type ForTiling <: TileStrategy end
abstract type ForAllTiling <: TileStrategy end

function tile(::Type{<:ForAllTiling}, target, tilesizes)
    return structured_tile_using_forall(target, tilesizes)
end
function tile(::Type{<:ForTiling}, target, tilesizes)
    return structured_tile_using_for(target, tilesizes)
end
function tile_reduction(::Type{<:ForTiling}, target, tilesizes)
    return structured_tile_reduction_using_for(target, tilesizes)
end

@intrinsic function yield(results::NTuple{N}) where N
    transform.yield(collect(results))
    nothing
end
yield() = yield(())

@intrinsic function apply_registered_pass(target, pass_name)
    AnyOp(IR.result(transform.apply_registered_pass(target; result=IR.Type(AnyOp), pass_name)))
end

@intrinsic function match(root, op_names)
    op = IR.result(transform.match_operation_name(root, op_names=IR.ArrayAttribute(collect(IR.Attribute.(op_names)))))
    return AnyOp(op)
end

@intrinsic function split_handle(handle, N)
    op = transform.split_handle(handle, results=fill(IR.Type(AnyOp), N))
    return Tuple([AnyOp(IR.result(op, i)) for i in 1:N])
end

@intrinsic function fuse_into(target, op)
    fused_op = IR.Type(AnyOp)
    new_containing_op = IR.Type(AnyOp)
    op = transform.structured_fuse_into_containing_op(
            op, target;
            fused_op, new_containing_op
        )
    return AnyOp(IR.result(op, 1)), AnyOp(IR.result(op, 2))
end

@intrinsic function vectorize_and_apply(target)
    AnyOp(IR.result(transform.structured_vectorize_children_and_apply_patterns(target; transformed=IR.Type(AnyOp))))
end

@intrinsic function one_shot_bufferize(target, bufferize_function_boundaries::Bool, function_boundary_type_conversion::Int32)
    AnyOp(IR.result(transform.bufferization_one_shot_bufferize(target; transformed=IR.Type(AnyOp), function_boundary_type_conversion, bufferize_function_boundaries)))
end
one_shot_bufferize(target) = one_shot_bufferize(target, false, 1)

end # Transform
