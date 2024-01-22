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
@mlirfunction function structured_tile_using_for(op::AnyOp, tilesizes::NTuple{N, Int})::Nothing where N
    loops = IR.MLIRType[]
    for _ in 1:length(tilesizes)
        push!(loops, IR.MLIRType(AnyOp))
    end
    static_sizes = IR.Attribute(API.mlirDenseI64ArrayGet(IR.context(), length(tilesizes), [tilesizes...]))
    scalable_sizes = IR.Attribute(API.mlirDenseBoolArrayGet(IR.context(), length(tilesizes), fill(Int32(false), length(tilesizes))))
    transform.structured_tile_using_for(
        op, [];
        tiled_linalg_op = IR.MLIRType(AnyOp),
        loops,
        static_sizes,
        scalable_sizes
    )
    nothing
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

Base.code_ircode(()) do 
    named_sequence() do op
        matched = structured_match(op, "linalg.generic")
        structured_tile_using_for(matched, (2, ))
        yield()
    end
    1
end

code_mlir(Tuple{}) do 
    named_sequence() do op
        matched = structured_match(op, "linalg.generic")
        structured_tile_using_for(matched, (16, 16, 16))
        yield()
    end
    1
end


# function matmul_and_schedule(Y, A, B)
#     named_sequence() do op
#         matched = structured_match(op, "linalg.generic")
#         structured_tile_using_for(matched, (2, 4, 4, 10))
#         yield()
#     end
#     mul!(Y, A, B)
#     1
# end

# Base.code_ircode(matmul_and_schedule, (tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}))
# code_mlir(matmul_and_schedule, Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}})








begin
ops = []
for op in IR.OperationIterator(first(IR.BlockIterator(region)))
    push!(ops, op)
end
ops
end

b = IR.Block()
arg1 = IR.push_argument!(b, parse(IR.MLIRType, "!transform.any_op"), IR.Location())

matched = transform.structured_match(
    arg1;
    results = parse(IR.MLIRType, "!transform.any_op"),
    ops = IR.ArrayAttribute([Attribute("linalg.generic")])
)
@show matched
tiled = transform.structured_tile_using_for(
    IR.get_result(matched), [];
    tiled_linalg_op=parse(IR.MLIRType, "!transform.any_op"),
    loops = [parse(IR.MLIRType, "!transform.any_op") for _ in 1:3],
    static_sizes=IR.Attribute(API.mlirDenseI64ArrayGet(IR.context(), 3, [2, 3, 4])),
    scalable_sizes=IR.Attribute(API.mlirDenseBoolArrayGet(IR.context(), 3, Int32[false, false, false]))
)
@show tiled
yield = transform.yield([])

push!.(Ref(b),[matched, tiled, yield])
body = Region()
push!(body, b)

ns = transform.named_sequence(;
    sym_name=Attribute("__transform_main"),
    function_type=MLIRType((parse(MLIRType, "!transform.any_op"),)=>()),
    body
)

bodyRegion = IR.Region()
b_module = IR.Block()
push!(b_module, ns)
push!(b_module, op)
push!(bodyRegion, b_module)
mod_op = builtin.module_(;
    additional_attributes=[NamedAttribute("transform.with_named_sequence", IR.Attribute(API.mlirUnitAttrGet(IR.context())))],
    bodyRegion
)
