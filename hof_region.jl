using MLIR
includet("utils.jl")

using Brutus.Library: index, f32, i64, memref, MLIRMemref
import Brutus.Library.Transform
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

# @intrinsic function scf_yield(results)
#     Dialects.scf.yield(results)
# end

import Brutus: generate_return, generate_function

abstract type LoopBody end

generate_return(cg::CodegenContext{LoopBody}, values; location) = Dialects.scf.yield(values)
generate_function(cg::CodegenContext{LoopBody}) = region(cg)


@intrinsic function scf_for(body, initial_value::T, start::index, stop::index, step::index) where T
    region = generate(CodegenContext{LoopBody}(body, Tuple{index, T}))
    op = Dialects.scf.for_(start, stop, step, [initial_value]; results=IR.Type[IR.Type(T)], region)
    return T(IR.result(op))
end
@intrinsic function scf_for(body, start::index, stop::index, step::index)
    region = generate(CodegenContext{LoopBody}(body, Tuple{index}))
    op = Dialects.scf.for_(start, stop, step, []; results=IR.Type[], region)
    return nothing
end


function f(N)
    val = i64(0)
    scf_for(val, index(0), N, index(1)) do i, val
        val + i64(i)
    end
end

generate(f, Tuple{index})

CodegenContext(cumsum, Tuple{index})

generate(Tuple{index, index, index, i64, i64}) do start, stop, step, initial, cst

    a = scf_for(initial, start, stop, step) do i, carry
        b = scf_for(initial, start, stop, step) do j, carry2
            carry2 * cst
        end
        b + cst
    end
    return a
end |> display

function fibonacci(n)
    # a = i64(0)
    # b = i64(1)
    result = scf_for((i64(0), i64(1)), index(0), n-2, index(1)) do i, (a, b)
        return (b, a+b)
    end
    result[2]
    # return b
end


generate(fibonacci, Tuple{index})
CodegenContext(fibonacci, Tuple{index}).ir

CodegenContext{LoopBody}(Tuple{index, Tuple{i64, i64}}) do i, (a, b)
    b, a+b
end
