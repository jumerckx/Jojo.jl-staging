### Setup ###
using MLIR
includet("utils.jl")
using MLIR: IR, API
using MLIR.IR: Value, MLIRType, NamedAttribute, Location
using MLIR.Dialects: arith, cf
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

new_intrinsic = () -> Base.compilerbarrier(:const, error("Intrinsics should be compiled to MLIR!"))
using MacroTools
using CassetteOverlay
# @MethodTable MLIRCompilation
MLIRCompilation = MLIR.Dialects.MLIRCompilation
mlircompilationpass = @overlaypass MLIRCompilation;


# from https://github.com/JuliaLang/julia/blob/1b183b93f4b78f567241b1e7511138798cea6a0d/base/experimental.jl#L345C1-L357C4
function overlay_def!(mt, @nospecialize ex)
    arg1 = ex.args[1]
    if isexpr(arg1, :call)
        arg1.args[1] = Expr(:overlay, mt, arg1.args[1])
    elseif isexpr(arg1, :(::))
        overlay_def!(mt, arg1)
    elseif isexpr(arg1, :where)
        overlay_def!(mt, arg1)
    else
        error("@overlay requires a function definition")
    end
    return ex
end

function mlirfunction_(expr)
    dict = splitdef(expr)
    if !(:rtype in keys(dict))
        error("Result type should be specified in function definition, `f(args)::T = ...` or `function f(args)::T ...`")
    end

    expr = overlay_def!(:MLIRCompilation, expr)
    return quote
        $expr

        function $(dict[:name])($(dict[:args]...); $(dict[:kwargs]...))::$(dict[:rtype]) where {$(dict[:whereparams]...)}
            new_intrinsic()
        end
    end
end

"""
    @mlirfunction [function def]

Will define the provided function definition twice. Once replacing the body with a call to
`new_intrinsic()`, and once with the body as-is, but registered in the `MLIRCompilation` method table.
"""
macro mlirfunction(f)
    f = macroexpand(__module__, f)
    Base.is_function_def(f) || error("@mlirfunction requires a function definition")
    mlirfunction_(f)
end


# function mlirop_(expr)
#     dict = splitdef(expr)
#     rtype = get(dict, :rtype, :Any)

#     modified = :($(dict[:name])($(dict[:args]...); $(dict[:kwargs]...)))
#     modified = nonoverlay(modified)
#     @info modified
#     modified = :(
#         function $(dict[:name])($(dict[:args]...); $(dict[:kwargs]...))::$rtype where {$(dict[:whereparams]...)}
#             $modified
#             println("overlayed!!!!")
#             push!(currentblock(), op)
#         end
#     )

#     modified = overlay_def!(:MLIRCompilation, modified)
#     return quote
#         $(expr)
#         $modified
#     end |> esc
# end

# macro mlirop(f)
#     f = macroexpand(__module__, f)
#     Base.is_function_def(f) || error("@mlirop requires a function definition")
#     mlirop_(f)
# end

# (@macroexpand @mlirop function addi(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
#     results = MLIRType[]
#     operands = Value[lhs, rhs, ]
#     owned_regions = Region[]
#     successors = Block[]
#     attributes = NamedAttribute[]
#     (result != nothing) && push!(results, result)
    
#     create_operation(
#         "arith.addi", location;
#         operands, owned_regions, successors, attributes,
#         results=(length(results) == 0 ? nothing : results),
#         result_inference=(length(results) == 0 ? true : false)
#     )
# end) |> prettify

# @mlirop function arith.addi(lhs::Value, rhs::Value; result=nothing::Union{Nothing, MLIRType}, location=Location())
#     results = MLIRType[]
#     operands = Value[lhs, rhs, ]
#     owned_regions = Region[]
#     successors = Block[]
#     attributes = NamedAttribute[]
#     (result != nothing) && push!(results, result)
    
#     create_operation(
#         "arith.addi", location;
#         operands, owned_regions, successors, attributes,
#         results=(length(results) == 0 ? nothing : results),
#         result_inference=(length(results) == 0 ? true : false)
#     )
# end

# Each function to generate an operation has an additional version with the
# same name in the MLIRCompilation context that will also push the created
# operation to the current block.
# @overlay MLIRCompilation function MLIR.Dialects.arith.addi(
#     lhs::Value,
#     rhs::Value;
#     result=nothing::Union{Nothing, MLIRType},
#     location=Location())
#     println("overlayed!")
#     op = @nonoverlay MLIR.Dialects.arith.addi(
#         lhs::Value,
#         rhs::Value;
#         result=nothing::Union{Nothing, MLIRType},
#         location=Location())
#     push!(current_block, op)
#     return op
# end

current_block = IR.Block() # Fixed block for demonstration purposes.


#= 
User code
=#
struct MyNumber
    value::Value
end

@mlirfunction function Base.:+(a::MyNumber, b::MyNumber)::MyNumber
    op = arith.addi(a.value, b.value)

    return MyNumber(IR.get_result(op))
end

f(a, b, c) = a + b + c

Base.code_ircode(f, Tuple{MyNumber,MyNumber,MyNumber})
"""
1 ─ %1 = invoke Base.:+(_2::MyNumber, _3::MyNumber)::MyNumber
│   %2 = invoke Base.:+(%1::MyNumber, _4::MyNumber)::MyNumber
└──      return %2
 => MyNumber

""";

#=
Following code is called behind-the-scene, in Brutus.code_mlir
=#
# dummy arguments, in Brutus.code_mlir, this will be the actual function arguments/ssa values:
placeholder = IR.Block()
a = MyNumber(MLIR.IR.push_argument!(placeholder, IR.MLIRType(Int), IR.Location()))
b = MyNumber(MLIR.IR.push_argument!(placeholder, IR.MLIRType(Int), IR.Location()))
c = MLIR.IR.push_argument!(placeholder, parse(IR.MLIRType, "!shape<shape>"), IR.Location())



op = mlircompilationpass((a, b)) do (a, b)
    Base.:+(a, b)
end # MyNumber(%0 = arith.addi <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>> : i64)
