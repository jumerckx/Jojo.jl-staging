using CassetteOverlay, MacroTools
import MLIR.IR.create_operation

@MethodTable MLIRCompilation
mlircompilationpass = @overlaypass MLIRCompilation
@inline (::typeof(mlircompilationpass))(::typeof(Base.typename), @nospecialize args...) = 
    Base.typename(args...)

@overlay MLIRCompilation function IR.create_operation(
        name, loc;
        results=nothing,
        operands=nothing,
        owned_regions=nothing,
        successors=nothing,
        attributes=nothing,
        result_inference=isnothing(results))
    @debug "overlaid create_operation for $name"
    op = @nonoverlay IR.create_operation(
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

"""
    @intrinsic [function def]

Mark a function definition as an intrinsic function. This will make sure the function is not inlined.

!!!note
    Intrinsic functions can't have keyword arguments. Define a regular function with kwargs that calls the intrinsic instead.

!!!note
    Currently, it's possible to confuse the intrinsic system by defining regular function methods with a signature that is more specific than the intrinsic.
    e.g. in this case:

    ```julia
    @intrinsic foo(x::Integer) = ...
    foo(x::Int32) = ...
    ```

    foo(x::Int32) will also be seen as an intrinsic, which could lead to unexpected behavior.
"""
macro intrinsic(f)
    f = macroexpand(__module__, f)
    Base.is_function_def(f) || error("@intrinsic requires a function definition")
    intrinsic_(f)
end

function intrinsic_(expr)
    dict = splitdef(expr)
    length(dict[:kwargs]) == 0 || error("Intrinsic functions can't have keyword arguments\nDefine a regular function with kwargs that calls the intrinsic instead.")
    argtypes = map(dict[:args]) do arg
        if arg isa Symbol
            return :Any
        elseif arg.head == :(::)
            return arg.args[end]
        else
            error("Don't know how to handle argument type $arg")
        end
    end

    return quote
        Jojo.reset_cache!()
        $(esc(expr))
        Jojo.is_intrinsic(::Type{<:Tuple{Jojo._typeof($(esc(dict[:name]))), $(esc.(argtypes)...)}}) where {$(esc.(dict[:whereparams])...)} = true
    end
end