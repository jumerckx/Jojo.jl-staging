using MLIR: IR, API
using MLIR.Dialects: func

unpack(T) = unpack(IR.MLIRValueTrait(T), T)
unpack(::IR.Convertible, T) = (T, )
function unpack(::IR.NonConvertible, T)
    @assert isbitstype(T) "Cannot unpack type $T that is not `isbitstype`"
    fc = fieldcount(T)
    if (fc == 0)
        if (sizeof(T) == 0)
            return []
        else
            error("Unable to unpack NonConvertible type $T any further")
        end
    end
    unpacked = []
    for i in 1:fc
        ft = fieldtype(T, i)
        append!(unpacked, unpack(ft))
    end
    return unpacked
end

abstract type AbstractCodegenContext end
Base.values(cg::T) where {T<:AbstractCodegenContext} = error("values not implemented for type $T")
args(cg::T) where {T<:AbstractCodegenContext} = error("args not implemented for type $T")
blocks(cg::T) where {T<:AbstractCodegenContext} = error("blocks not implemented for type $T")
entryblock(cg::T) where {T<:AbstractCodegenContext} = error("entryblock not implemented for type $T")
region(cg::T) where {T<:AbstractCodegenContext} = error("region not implemented for type $T")
returntype(cg::T) where {T<:AbstractCodegenContext} = error("returntype not implemented for type $T")
ir(cg::T) where {T<:AbstractCodegenContext} = error("ir not implemented for type $T")
currentblockindex(cg::T) where {T<:AbstractCodegenContext} = error("currentblockindex not implemented for type $T")
setcurrentblockindex!(cg::T, i) where {T<:AbstractCodegenContext} = error("setcurrentblockindex! not implemented for type $T")
currentblock(cg::T) where {T<:AbstractCodegenContext} = error("currentblock not implemented for type $T")

generate_return(cg::T, values; location) where {T<:AbstractCodegenContext} = error("generate_return not implemented for type $T")
generate_goto(cg::T, args, dest; location) where {T<:AbstractCodegenContext} = error("generate_goto not implemented for type $T")
generate_gotoifnot(cg::T, cond; true_args, false_args, true_dest, false_dest, location) where {T<:AbstractCodegenContext} = error("generate_gotoifnot not implemented for type $T")
generate_function(cg::T, region) where {T<:AbstractCodegenContext} = error("generate_function not implemented for type $T")

abstract type Default end

mutable struct CodegenContext{T} <: AbstractCodegenContext
    f::Function
    region::Region
    const blocks::Vector{Block}
    const entryblock::Block
    currentblockindex::Int
    const ir::Core.Compiler.IRCode
    const ret::Type
    const values::Vector
    const args::Vector


    function CodegenContext{T}(;
            f::Function,
            region::Region,
            blocks::Vector{Block},
            entryblock::Block,
            currentblockindex::Int,
            ir::Core.Compiler.IRCode,
            ret::Type,
            values::Vector,
            args::Vector) where {T}
        cg = new{T}(f, region, blocks, entryblock, currentblockindex, ir, ret, values, args)
        return cg
    end
end

CodegenContext(f, types) = CodegenContext{Default}(f, types)

function CodegenContext{T}(f, types) where T
    ir, ret = Core.Compiler.code_ircode(f, types; interp=MLIRInterpreter()) |> only

    types = ir.argtypes[begin+1:end]
    values = Vector(undef, length(ir.stmts))
    args = Vector(undef, length(types)+1)

    blocks = [
        prepare_block(ir, bb)
        for bb in ir.cfg.blocks
    ]
    entryblock = blocks[begin]
    
    args[1] = f # first argument is special in that it doesn't show up in the MLIR
    for (i, argtype) in enumerate(types)
        temp = []
        for t in unpack(argtype)
            arg = IR.push_argument!(entryblock, IR.Type(t))
            push!(temp, t(arg))
        end
        # TODO: what to do with padding?
        args[i+1] = reinterpret(argtype, Tuple(temp))
    end

    CodegenContext{T}(;
        f,
        region=Region(),
        blocks,
        entryblock=entryblock,
        currentblockindex=0,
        ir,
        ret,
        values,
        args
    )
end

Base.values(cg::CodegenContext) = cg.values
args(cg::CodegenContext) = cg.args
blocks(cg::CodegenContext) = cg.blocks
entryblock(cg::CodegenContext) = cg.entryblock
region(cg::CodegenContext) = cg.region
currentblockindex(cg::CodegenContext) = cg.currentblockindex
setcurrentblockindex!(cg::CodegenContext, i) = cg.currentblockindex = i
returntype(cg::CodegenContext) = cg.ret
ir(cg::CodegenContext) = cg.ir
currentblock(cg::CodegenContext) = cg.blocks[cg.currentblockindex]
generate_return(cg::CodegenContext, values; location) = func.return_(values; location)
generate_goto(cg::CodegenContext, args, dest; location) = cf.br(args; dest, location)
generate_gotoifnot(
    cg::CodegenContext, cond;
    true_args, false_args, true_dest, false_dest, location
    )= cf.cond_br(
        cond, true_args, false_args;
        trueDest=true_dest, falseDest=false_dest, location
        )
function generate_function(cg::CodegenContext)
    body = region(cg)
    input_types = IR.Type[IR.type(IR.argument(entryblock(cg), i)) for i in 1:IR.nargs(entryblock(cg))]

    # getting result types requires return type to be fully inferred and convertible to MLIR:
    result_types = IR.Type[IR.Type.(unpack(returntype(cg)))...]
    function_type = IR.FunctionType(input_types, result_types)
    op = func.func_(;
        sym_name=string(nameof(cg.f)),
        function_type,
        body,
    )
    IR.attr!(op, "llvm.emit_c_interface", API.mlirUnitAttrGet(IR.context()))
    IR.verify(op)
    return op
end

_has_context() = haskey(task_local_storage(), :CodegenContext) &&
                 !isempty(task_local_storage(:CodegenContext))

function codegencontext(; throw_error::Core.Bool=true)
    if !_has_context()
        throw_error && error("No CodegenContext is active")
        return nothing
    end
    last(task_local_storage(:CodegenContext))
end

function activate(cg::AbstractCodegenContext)
    @debug "activating $(hash(Ref(cg)))"
    stack = get!(task_local_storage(), :CodegenContext) do
        AbstractCodegenContext[]
    end
    push!(stack, cg)
    return
end

function deactivate(cg::AbstractCodegenContext)
    @debug "deactivating $(hash(Ref(cg)))"
    codegencontext() == cg || error("Deactivating wrong CodegenContext")
    pop!(task_local_storage(:CodegenContext))
end

function codegencontext!(f, cg::AbstractCodegenContext)
    activate(cg)
    try
        f()
    finally
        deactivate(cg)
    end
end

function get_value(cg::AbstractCodegenContext, x)
    if x isa Core.SSAValue
        @assert isassigned(values(cg), x.id) "value $x was not assigned"
        return values(cg)[x.id]
    elseif x isa Core.Argument
        @assert isassigned(args(cg), x.n) "value $x was not assigned"
        return args(cg)[x.n]
        # return IR.argument(cg.entryblock, x.n - 1)
    elseif x == GlobalRef(Main, :nothing) # This might be something else than Main sometimes?
        return IR.Type(Nothing)
    else
        # error("could not use value $x inside MLIR")
        @debug "Value could not be converted to MLIR: $x, of type $(typeof(x))."
        return x
    end
end
