@noinline function intrinsic(T)::T
    # prevent type inference:
    invokelatest(error, "MLIR intrinsics can't be executed in a regular Julia context.")

    Base.inferencebarrier(nothing)::T
end

abstract type BoolTrait end
struct NonBoollike <: BoolTrait end
struct Boollike <: BoolTrait end
BoolTrait(T) = NonBoollike()

mlir_bool_conversion(x::Bool) = x
@inline mlir_bool_conversion(x::T) where T = mlir_bool_conversion(BoolTrait(T), x)
@noinline mlir_bool_conversion(::Boollike, x)::Bool = intrinsic(Bool)
mlir_bool_conversion(::NonBoollike, x::T) where T = error("Type $T is not marked as Boollike.")

using Core: MethodInstance, CodeInstance, OpaqueClosure
const CC = Core.Compiler
using CodeInfoTools

## code instance cache

struct CodeCache
    dict::IdDict{MethodInstance,Vector{CodeInstance}}

    CodeCache() = new(Dict{MethodInstance,Vector{CodeInstance}}())
end

Base.empty!(cc::CodeCache) = empty!(cc.dict)

function CC.setindex!(cache::CodeCache, ci::CodeInstance, mi::MethodInstance)
    cis = get!(cache.dict, mi, CodeInstance[])
    push!(cis, ci)
end


## world view of the cache

using Core.Compiler: WorldView

function CC.haskey(wvc::WorldView{CodeCache}, mi::MethodInstance)
    CC.get(wvc, mi, nothing) !== nothing
end

function CC.get(wvc::WorldView{CodeCache}, mi::MethodInstance, default)
    # check the cache
    # Core.println("$(mi.def)")
    # display.(stacktrace())
    # Core.println("")
    for ci in get!(wvc.cache.dict, mi, CodeInstance[])
        if ci.min_world <= wvc.worlds.min_world && wvc.worlds.max_world <= ci.max_world
            
            # Core.println("Getting: ", mi.def)
            # Core.println("\tcodeinstance range:\t(", ci.min_world, ", ", ci.max_world, ")")
            # Core.println("\tworld range:\t\t(", wvc.worlds.min_world, ", ", wvc.worlds.max_world, ")")

            # TODO: if (code && (code == jl_nothing || jl_ir_flag_inferred((jl_array_t*)code)))
            src = if ci.inferred isa Vector{UInt8}
                ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any),
                       mi.def, C_NULL, ci.inferred)
            else
                ci.inferred
            end
            return ci
        end
    end
    return default
end

function CC.getindex(wvc::WorldView{CodeCache}, mi::MethodInstance)
    r = CC.get(wvc, mi, nothing)
    r === nothing && throw(KeyError(mi))
    return r::CodeInstance
end

function CC.setindex!(wvc::WorldView{CodeCache}, ci::CodeInstance, mi::MethodInstance)
    src = if ci.inferred isa Vector{UInt8}
        ccall(:jl_uncompress_ir, Any, (Any, Ptr{Cvoid}, Any),
                mi.def, C_NULL, ci.inferred)
    else
        ci.inferred
    end
    CC.setindex!(wvc.cache, ci, mi)
end


## custom interpreter

struct MLIRInterpreter <: CC.AbstractInterpreter
    world::UInt

    code_cache::CodeCache
    inf_cache::Vector{CC.InferenceResult}

    inf_params::CC.InferenceParams
    opt_params::CC.OptimizationParams
end

function MLIRInterpreter(world::UInt;
                           code_cache::CodeCache,
                           inf_params::CC.InferenceParams,
                           opt_params::CC.OptimizationParams)
    @assert world <= Base.get_world_counter()

    # inf_params.ipo_constant_propagation = false
    inf_params = CC.InferenceParams(ipo_constant_propagation=false)

    inf_cache = Vector{CC.InferenceResult}()

    return MLIRInterpreter(world,
                             code_cache, inf_cache,
                             inf_params, opt_params)
end
MLIRInterpreter(opt_params=CC.OptimizationParams(#= inline_cost_threshold=... =#)) = MLIRInterpreter(
    Base.get_world_counter();
    code_cache=global_ci_cache,
    inf_params=CC.InferenceParams(),
    opt_params
)

CC.InferenceParams(interp::MLIRInterpreter) = interp.inf_params
CC.OptimizationParams(interp::MLIRInterpreter) = interp.opt_params
CC.get_world_counter(interp::MLIRInterpreter) = interp.world
CC.get_inference_cache(interp::MLIRInterpreter) = interp.inf_cache
CC.code_cache(interp::MLIRInterpreter) = WorldView(interp.code_cache, interp.world)
CC.get_inference_world(interp::MLIRInterpreter) = interp.world
CC.cache_owner(interp::MLIRInterpreter) = nothing

# No need to do any locking since we're not putting our results into the runtime cache
CC.lock_mi_inference(interp::MLIRInterpreter, mi::MethodInstance) = nothing
CC.unlock_mi_inference(interp::MLIRInterpreter, mi::MethodInstance) = nothing

function CC.add_remark!(interp::MLIRInterpreter, sv::CC.InferenceState, msg)
    @debug "Inference remark during compilation of MethodInstance of $(sv.linfo): $msg"
end

CC.may_optimize(interp::MLIRInterpreter) = true
CC.may_compress(interp::MLIRInterpreter) = true
CC.may_discard_trees(interp::MLIRInterpreter) = true
CC.verbose_stmt_info(interp::MLIRInterpreter) = false

struct MLIRIntrinsicCallInfo <: CC.CallInfo
    info::CC.CallInfo
    MLIRIntrinsicCallInfo(@nospecialize(info::CC.CallInfo)) = new(info)
end
CC.nsplit_impl(info::MLIRIntrinsicCallInfo) = CC.nsplit(info.info)
CC.getsplit_impl(info::MLIRIntrinsicCallInfo, idx::Int) = CC.getsplit(info.info, idx)
CC.getresult_impl(info::MLIRIntrinsicCallInfo, idx::Int) = CC.getresult(info.info, idx)


function CC.abstract_call_gf_by_type(interp::MLIRInterpreter, @nospecialize(f), arginfo::CC.ArgInfo, si::CC.StmtInfo, @nospecialize(atype),
    sv::CC.AbsIntState, max_methods::Int)

    cm = @invoke CC.abstract_call_gf_by_type(interp::CC.AbstractInterpreter, f::Any,
    arginfo::CC.ArgInfo, si::CC.StmtInfo, atype::Any, sv::CC.InferenceState, max_methods::Int)
    
    Core.println(f)
    tt = Core.Compiler.argtypes_to_type(arginfo.argtypes)
    world = CC.get_inference_world(interp)
    method_table = CC.method_table(interp)
    caller = sv

    add_intrinsic_backedges(tt; world, method_table, caller)
    
    argtype_tuple = Tuple{map(_type, arginfo.argtypes)...}
    
    if is_primitive(argtype_tuple)
        return CC.CallMeta(cm.rt, cm.exct, cm.effects, MLIRIntrinsicCallInfo(cm.info))
    else
        return cm
    end
end


"""
    _typeof(x)

Central definition of typeof, which is specific to the use-required in this package.
"""
_typeof(x) = Base._stable_typeof(x)
_typeof(x::Tuple) = Tuple{map(_typeof, x)...}
_typeof(x::NamedTuple{names}) where {names} = NamedTuple{names, _typeof(Tuple(x))}

_type(x) = x
_type(x::CC.Const) = _typeof(x.val)
_type(x::CC.PartialStruct) = x.typ
_type(x::CC.Conditional) = Union{x.thentype, x.elsetype}

is_primitive(::Any) = false

macro is_primitive(sig)
    return esc(:(Brutus.is_primitive(::Type{<:$sig}) = true))
end

is_primitive(::Type{<:Tuple{typeof(*), Integer, Integer}}) = true

# @is_primitive Tuple{typeof(*), Int, Int}

function CC.inlining_policy(interp::MLIRInterpreter,
    @nospecialize(src), @nospecialize(info::CC.CallInfo), stmt_flag::UInt32)

    if isa(info, MLIRIntrinsicCallInfo)
        return nothing
    else
        return src
    end
end

function add_intrinsic_backedges(@nospecialize(tt);
                      world::UInt=Base.get_world_counter(),
                      method_table::Union{Nothing,Core.Compiler.MethodTableView}=nothing,
                      caller::CC.AbsIntState)
    sig = Base.signature_type(Brutus.is_primitive, Tuple{Type{tt}})
    mt = ccall(:jl_method_table_for, Any, (Any,), sig)
    mt isa Core.MethodTable || return false
    if method_table === nothing
        method_table = Core.Compiler.InternalMethodTable(world)
    end
    Core.println("Finding all methods for $(sig)")
    result = Core.Compiler.findall(sig, method_table; limit=-1)
    @assert !(result === nothing || result === missing)
    @static if isdefined(Core.Compiler, :MethodMatchResult)
        (; matches) = result
    else
        matches = result
    end
    fullmatch = Core.Compiler._any(match::Core.MethodMatch->match.fully_covers, matches)
    if caller !== nothing
        fullmatch || add_mt_backedge!(caller, mt, sig)
    end
    if Core.Compiler.isempty(matches)
        return false
    else
        if caller !== nothing
            for i = 1:Core.Compiler.length(matches)
                match = Core.Compiler.getindex(matches, i)::Core.MethodMatch
                edge = Core.Compiler.specialize_method(match)::Core.MethodInstance
                Core.println("Adding backedge from $(caller) to $(edge)")
                CC.add_backedge!(caller, edge)
                # Core.println("\t$(edge.backedges)")
            end
        end
        return true
    end
end

function add_backedge!(caller::Core.MethodInstance, callee::Core.MethodInstance, @nospecialize(sig))
    ccall(:jl_method_instance_add_backedge, Cvoid, (Any, Any, Any), callee, sig, caller)
    return nothing
end

function add_mt_backedge!(caller::Core.MethodInstance, mt::Core.MethodTable, @nospecialize(sig))
    ccall(:jl_method_table_add_backedge, Cvoid, (Any, Any, Any), mt, sig, caller)
    return nothing
end


## utils

# create a MethodError from a function type
# TODO: fix upstream
function unsafe_function_from_type(ft::Type)
    if isdefined(ft, :instance)
        ft.instance
    else
        # HACK: dealing with a closure or something... let's do somthing really invalid,
        #       which works because MethodError doesn't actually use the function
        Ref{ft}()[]
    end
end
function MethodError(ft::Type{<:Function}, tt::Type, world::Integer=typemax(UInt))
    Base.MethodError(unsafe_function_from_type(ft), tt, world)
end
MethodError(ft, tt, world=typemax(UInt)) = Base.MethodError(ft, tt, world)

const global_ci_cache = CodeCache()

import Core.Compiler: retrieve_code_info, maybe_validate_code, InferenceState, InferenceResult
# Replace usage sites of `retrieve_code_info`, OptimizationState is one such, but in all interesting use-cases
# it is derived from an InferenceState. There is a third one in `typeinf_ext` in case the module forbids inference.
function InferenceState(result::InferenceResult, cache_mode::UInt8, interp::MLIRInterpreter)
    src = retrieve_code_info(result.linfo, interp.world)
    src === nothing && return nothing
    maybe_validate_code(result.linfo, src, "lowered")
    src = transform(interp, result.linfo, src)
    maybe_validate_code(result.linfo, src, "transformed")

    return InferenceState(result, src, cache_mode, interp)
end

struct DestinationOffsets
    indices::Vector{Int}
    DestinationOffsets() = new([])
end
function Base.insert!(d::DestinationOffsets, insertion::Int)
    candidateindex = d[insertion]+1
    if (length(d.indices) == 0)
        push!(d.indices, insertion)
    elseif candidateindex == length(d.indices)+1
        push!(d.indices, insertion)
    elseif (candidateindex == 1) || (d.indices[candidateindex-1] != insertion)
        insert!(d.indices, candidateindex, insertion)
    end
    return d
end
Base.getindex(d::DestinationOffsets, i::Int) = searchsortedlast(d.indices, i, lt= <=)

function insert_bool_conversions_pass(mi, src)
    offsets = DestinationOffsets()

    b = CodeInfoTools.Builder(src)
    for (v, st) in b
        if st isa Core.GotoIfNot
            arg = st.cond isa Core.SSAValue ? var(st.cond.id + offsets[st.cond.id]) : st.cond
            b[v] = Statement(Expr(:call, GlobalRef(Brutus, :mlir_bool_conversion), arg))
            push!(b, Core.GotoIfNot(v, st.dest))
            insert!(offsets, v.id)
        elseif st isa Core.GotoNode
            b[v] = st
        end
    end

    # fix destinations and conditions
    for i in 1:length(b.to)
        st = b.to[i].node
        if st isa Core.GotoNode
            b.to[i] = Core.GotoNode(st.label + offsets[st.label])
        elseif st isa Core.GotoIfNot
            b.to[i] = Statement(Core.GotoIfNot(st.cond, st.dest + offsets[st.dest]))
        end
    end
    finish(b)
end

function transform(interp, mi, src)
    src = insert_bool_conversions_pass(mi, src)
    return src
end
