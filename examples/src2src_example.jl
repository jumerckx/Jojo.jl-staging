includet("utils.jl")

using MLIR: IR, API, Dialects
import Jojo
using Jojo.Library: i64

ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

# pirate!
# (this function is the same as the original implementation except for the pushing the created operation to the current block.)
import MLIR.IR: create_operation
import MLIR: API
function create_operation(
    name, loc;
    results=nothing,
    operands=nothing,
    owned_regions=nothing,
    successors=nothing,
    attributes=nothing,
    result_inference=isnothing(results)
)
    GC.@preserve name loc begin
        state = Ref(API.mlirOperationStateGet(name, loc))
        if !isnothing(results)
            if result_inference
                error("Result inference and provided results conflict")
            end
            API.mlirOperationStateAddResults(state, length(results), results)
        end
        if !isnothing(operands)
            API.mlirOperationStateAddOperands(state, length(operands), operands)
        end
        if !isnothing(owned_regions)
            IR.lose_ownership!.(owned_regions)
            GC.@preserve owned_regions begin
                mlir_regions = Base.unsafe_convert.(API.MlirRegion, owned_regions)
                API.mlirOperationStateAddOwnedRegions(state, length(mlir_regions), mlir_regions)
            end
        end
        if !isnothing(successors)
            GC.@preserve successors begin
                mlir_blocks = Base.unsafe_convert.(API.MlirBlock, successors)
                API.mlirOperationStateAddSuccessors(
                    state,
                    length(mlir_blocks),
                    mlir_blocks,
                )
            end
        end
        if !isnothing(attributes)
            API.mlirOperationStateAddAttributes(state, length(attributes), attributes)
        end
        if result_inference
            API.mlirOperationStateEnableResultTypeInference(state)
        end
        op = API.mlirOperationCreate(state)
        if IR.mlirIsNull(op)
            error("Create Operation '$name' failed")
        end
        op = IR.Operation(op, true)
    end
    if (Jojo.currentblockindex(Jojo.codegencontext()) != 0)
        push!(Jojo.currentblock(Jojo.codegencontext()), op)
    end
    op
end

function run(f, type)
    cg = Jojo.CodegenContext(f, type)
    @info "original IR" cg.ir
    Jojo.codegencontext!(cg) do
        f = Jojo.source2source(cg.ir)
        f(cg.args[2:end]...)
        Jojo.region(cg)
        Jojo.setcurrentblockindex!(cg, 0) # hacky way to signal that we don't want to newly created operations to any block.
        Jojo.generate_function(cg)
    end
end

# simple example:
run(Tuple{i64, i64}) do a, b
    a+b
end

# now with control flow and structured types:
run(Tuple{i64, i64}) do a, b
    if (Complex(a, b)*Complex(a, b)).re + b > b
        return b
    end
    return a
end

# structured argument and return type:
run(Tuple{Complex{i64}}) do a
    Complex(a.re + a.im, a.re - a.im)
end

function max(a, b)
    if (a >= b)
        c = a
    else
        c = b
    end
    return c
end
run(max, Tuple{i64, i64})
