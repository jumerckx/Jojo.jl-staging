using CassetteOverlay
@MethodTable MyTable
mypass = @overlaypass MyTable
world = Base.get_world_counter()

const CC = Core.Compiler
@inline function signature_type_by_tt(ft::Type, tt::Type)
    u = Base.unwrap_unionall(tt)
    return Base.rewrap_unionall(Tuple{ft, elim_free_typevars.(u.parameters)...}, tt)
end

function elim_free_typevars(@nospecialize t)
    if CC.has_free_typevars(t)
        return CC.isType(t) ? Type : Any
    else
        return t
    end
end

function methodinstance(ft::Type, tt::Type)
    sig = signature_type_by_tt(ft, tt)

    match, _ = CC._findsup(sig, nothing, world)
    
    match === nothing && throw(MethodError(ft, tt, world))

    mi = CC.specialize_method(match)

    return mi::CC.MethodInstance
end

ft = typeof(mypass)
tt = Tuple{Type{UnionAll}, TypeVar, Type{Array{TypeVar(:T,Integer)}}}
mi = methodinstance(ft, tt)::CC.MethodInstance

# ft = typeof(mypass)
# tt = Tuple{Type{UnionAll}, TypeVar, Type{Array{TypeVar(:T,Integer)}}}
# mi = methodinstance(ft, tt)::CC.MethodInstance

# # inference fails, resulting in the interpreter firing
# try
#     mypass() do
#         UnionAll(TypeVar(:T,Integer), Array{TypeVar(:T,Integer)})
#     end
# catch err
#     Base.showerror(stdout, err)
#     Base.show_backtrace(stdout, catch_backtrace())
# end

# println()

# inference asserts
@show @ccall jl_type_infer(mi::Any, world::Csize_t, false::Cint)::Any