using CassetteOverlay, MacroTools
import MLIR.IR.create_operation

@MethodTable MLIRCompilation
mlircompilationpass = @overlaypass MLIRCompilation

@overlay MLIRCompilation function IR.create_operation(
        name, loc;
        results=nothing,
        operands=nothing,
        owned_regions=nothing,
        successors=nothing,
        attributes=nothing,
        result_inference=isnothing(results))
    @info "Overlayed!!!"
    op = @nonoverlay create_operation(
        name, loc;
        results,
        operands,
        owned_regions,
        successors,
        attributes,
        result_inference)
    push!(currentblock(codegencontext()), op)
    return op
end

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
    rtype = dict[:rtype]

    # since the methodtable shouldn't be escaped, we create an alias that escapes and binds to the real deal.
    methodtable = gensym(:MLIRCompilation)

    # hacky: in the MLIRCompilation context, we get rid of the return type as this could be something else than what should be in the Julia IR.
    delete!(dict, :rtype)
    expr = overlay_def!(methodtable, combinedef(dict))

    return quote
        # TODO: is this allowed? Maybe the gensym'd name can collide with a method in the caller's module?
        # https://discourse.julialang.org/t/can-i-unescape-a-variable-name-within-an-esc-node/102396/7
        $(esc(methodtable)) = MLIRCompilation

        function $(esc(dict[:name]))($(esc.(dict[:args])...); $(esc.(dict[:kwargs])...))::$(esc(rtype)) where {$(esc.(dict[:whereparams])...)}
            new_intrinsic()
        end

        $(esc(expr))
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