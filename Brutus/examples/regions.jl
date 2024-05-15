include("utils.jl")

import MLIR: IR, API
import Brutus
import Brutus.Library: index, i64
import MLIR.Dialects

ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

import Brutus: generate_return, generate_function

abstract type LoopBody end

generate_return(cg::Brutus.CodegenContext{LoopBody}, values; location) = Dialects.scf.yield(values)
generate_function(cg::Brutus.CodegenContext{LoopBody}) = Brutus.region(cg)

Brutus.@intrinsic function scf_for(body, initial_value::T, start::index, stop::index, step::index) where T
    region = Brutus.generate(Brutus.CodegenContext{LoopBody}(body, Tuple{index, T}))
    op = Dialects.scf.for_(start, stop, step, [initial_value]; results=IR.Type[IR.Type(T)], region)
    return T(IR.result(op))
end
Brutus.@intrinsic function scf_for(body, start::index, stop::index, step::index)
    region = generate(Brutus.CodegenContext{LoopBody}(body, Tuple{index}))
    op = Dialects.scf.for_(start, stop, step, []; results=IR.Type[], region)
    return nothing
end

function f(N)
    val = i64(0)
    scf_for(val, index(0), N, index(1)) do i, val
        val + i64(i)
    end
end

Brutus.generate(f, Tuple{index})

