module Brutus

using MLIR.IR
using MLIR: API
using MLIR.Dialects: arith, func, cf, memref, index, builtin, llvm, ub
using Core: PhiNode, GotoNode, GotoIfNot, SSAValue, Argument, ReturnNode, PiNode

include("MemRef.jl")
include("pass.jl")
include("overlay.jl")
include("abstract.jl")
include("codegencontext.jl")
include("generate.jl")
include("src2src.jl")
include("library/Library.jl")

end # module Brutus
