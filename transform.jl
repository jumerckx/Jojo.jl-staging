includet("utils.jl")
using MLIR
using MLIR: IR
using MLIR.Dialects: transform, arith
import Brutus: @mlirfunction, code_mlir
using CassetteOverlay

ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

struct AnyOp
    value::IR.Value
end
IR.MLIRValueTrait(::Type{<:AnyOp}) = IR.Convertible()
IR.MLIRType(::Type{AnyOp}) = IR.MLIRType(IR.parse(IR.MLIRType, "!transform.any_op"))

@mlirfunction function structured_match(parent::AnyOp, name::String)::AnyOp
    return IR.get_result(transform.structured_match(
        parent;
        results=IR.MLIRType(AnyOp),
        ops = IR.ArrayAttribute([IR.Attribute(name)])
    )) |> AnyOp
end
@mlirfunction function structured_tile_using_for(op::AnyOp, tilesizes::NTuple{N, Int})::AnyOp where N
    loops = fill(IR.MLIRType(AnyOp), count(tilesizes != 0))
    static_sizes = IR.Attribute(API.mlirDenseI64ArrayGet(IR.context(), length(tilesizes), [tilesizes...]))
    scalable_sizes = IR.Attribute(API.mlirDenseBoolArrayGet(IR.context(), length(tilesizes), fill(Int32(false), length(tilesizes))))
    op = transform.structured_tile_using_for(
        op, [];
        tiled_linalg_op = IR.MLIRType(AnyOp),
        loops,
        static_sizes,
        scalable_sizes
    )
    return AnyOp(IR.get_result(op, 1))
end
@mlirfunction function yield(results::NTuple{N})::Nothing where N
    transform.yield(collect(results))
    nothing
end
@inline yield() = yield(())

@mlirfunction function named_sequence(body, name="__transform_main")::Nothing
    body = @nonoverlay code_mlir(body, Tuple{AnyOp}, emit_region=true)
    transform.named_sequence(;
        body,
        sym_name=name,
# not working because collect throws an error in CassetteOverlay
# https://github.com/JuliaDebug/CassetteOverlay.jl/issues/39
        # function_type=IR.MLIRType((AnyOp, )=>())

        function_type=IR.MLIRType(API.mlirFunctionTypeGet(IR.context(),
            1, [IR.MLIRType(AnyOp)],
            0, []))
    )
    nothing
end

Base.code_ircode((AnyOp, )) do op
    matched = structured_match(op, "linalg.generic")
    tiled = structured_tile_using_for(matched, (0, 0, 1))
    tiled = structured_tile_using_for(tiled, (0, 1, 0))
    yield()
end

code_mlir(Tuple{}) do 
    named_sequence() do op
        matched = structured_match(op, "linalg.generic")
        tiled = structured_tile_using_for(matched, (0, 0, 1))
        tiled = structured_tile_using_for(tiled, (0, 1, 0))
        yield()
    end
end
