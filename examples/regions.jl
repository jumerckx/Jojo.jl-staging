include("utils.jl")

import MLIR: IR, API
import Jojo
import Jojo.Library: index, i64
import MLIR.Dialects

ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

import Jojo: generate_return, generate_function

abstract type LoopBody end

generate_return(cg::Jojo.CodegenContext{LoopBody}, values; location) = Dialects.scf.yield(values)
generate_function(cg::Jojo.CodegenContext{LoopBody}) = Jojo.region(cg)

Jojo.@intrinsic function scf_for(body, initial_value::T, start::index, stop::index, step::index) where T
    region = Jojo.generate(Jojo.CodegenContext{LoopBody}(body, Tuple{index, T}))
    op = Dialects.scf.for_(start, stop, step, [initial_value]; results=IR.Type[IR.Type(T)], region)
    return T(IR.result(op))
end
Jojo.@intrinsic function scf_for(body, start::index, stop::index, step::index)
    region = generate(Jojo.CodegenContext{LoopBody}(body, Tuple{index}))
    op = Dialects.scf.for_(start, stop, step, []; results=IR.Type[], region)
    return nothing
end

function f(N)
    val = i64(0)
    scf_for(val, index(0), N, index(1)) do i, val
        val + i64(i)
    end
end

Jojo.generate(f, Tuple{index})
