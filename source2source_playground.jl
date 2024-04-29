using MLIR
includet("utils.jl")
import Brutus
using Brutus.Library: index, f32, i64, memref, MLIRMemref
import Brutus.Library.Transform
import Brutus: MemRef, @intrinsic, MLIRInterpreter, generate, unpack, entryblock, returntype, region, CodegenContext, simplify
using BenchmarkTools, MLIR, MacroTools

import MLIR.Dialects
using MLIR.Dialects: arith, gpu, transform
using MLIR.IR: Context, @affinemap, Attribute, AffineMap, DenseArrayAttribute, Type, context
using MLIR.API: mlirRegisterAllPasses, mlirRegisterAllLLVMTranslations

ctx = IR.Context()
registerAllDialects!();
mlirRegisterAllPasses()
mlirRegisterAllLLVMTranslations(ctx.context)

# pirate!
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
    push!(Brutus.currentblock(Brutus.codegencontext()), op)
    op
end

cg = Brutus.CodegenContext(Tuple{i64, i64}) do a, b
    # if (Complex(a, b)*Complex(a, b)).re + b > b
    #     return b
    # end
    # return a

    # currently fails because of bug in phi-node conversion:
    if (a>b)
        d = a
        e = b
    else
        d = b
        e = b
    end
    return d+e
end

ir = Brutus.source2source(cg.ir)

@time Brutus.codegencontext!(cg) do
    f = Brutus.source2source(cg.ir)
    f(cg.args[2:end]...)
    # Brutus.generate_function(cg)
end

function run(f, type)
    cg = Brutus.CodegenContext(f, type)
    Brutus.codegencontext!(cg) do
        f = Brutus.source2source(cg.ir)
        f(cg.args[2:end]...)
    end
end

run(Tuple{i64, i64}) do a, b
    if (Complex(a, b)*Complex(a, b)).re + b > b
        return b
    end
    return a
end
