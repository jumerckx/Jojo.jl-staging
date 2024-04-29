const CC = Core.Compiler

function my_goto(label, args)
    cg = codegencontext()
    dest = blocks(cg)[label]
    @info typeof(args)
    generate_goto(cg, args, dest; location=IR.Location())
end
function my_gotoifnot(cond, true_dest, false_dest, true_args, false_args)
    cg = codegencontext()
    true_dest = blocks(cg)[true_dest]
    false_dest = blocks(cg)[false_dest]
    generate_gotoifnot(cg, cond; true_args, false_args, true_dest, false_dest, location=IR.Location())
end
function my_phi(T, i)
    cg = codegencontext()
    T(IR.argument(currentblock(cg), i))
end
function my_return(val::T) where T
    cg = codegencontext()
    if isnothing(val)
        returnvalues = []
    else
        returnvalues = reinterpret(Tuple{unpack(T)...}, val)
    end
    generate_return(cg, returnvalues; location=IR.Location())
end
function start_block()
    cg = codegencontext()
    setcurrentblockindex!(cg, currentblockindex(cg)+1)
    push!(region(cg), currentblock(cg))
end


#Helpers:
"Given some IR generates a MethodInstance suitable for passing to infer_ir!, if you don't already have one with the right argument types"
function get_toplevel_mi_from_ir(ir, _module::Module)
    mi = ccall(:jl_new_method_instance_uninit, Ref{Core.MethodInstance}, ());
    ft = typeof(first(ir.argtypes) isa Core.Const ? first(ir.argtypes).val : first(ir.argtypes))
    mi.specTypes = Tuple{ft, ir.argtypes[2:end]...}
    # mi.specTypes = Tuple{typeof(first(ir.argtypes).val), ir.argtypes[2:end]...}
    mi.def = _module
    return mi
end

"run type inference and constant propagation on the ir"
function infer_ir!(ir, interp::CC.AbstractInterpreter, mi::CC.MethodInstance)
    method_info = CC.MethodInfo(#=propagate_inbounds=#true, nothing)
    min_world = world = CC.get_world_counter()
    max_world = Base.get_world_counter()
    irsv = CC.IRInterpretationState(interp, method_info, ir, mi, ir.argtypes, world, min_world, max_world)
    rt = CC._ir_abstract_constant_propagation(interp, irsv)
    return ir
end


# add overloads from Core.Compiler into Base
# Diffractor has a bunch of these, we need to make a library for them
# https://github.com/JuliaDiff/Diffractor.jl/blob/b23337a4b12d21104ff237cf0c72bcd2fe13a4f6/src/stage1/hacks.jl
# https://github.com/JuliaDiff/Diffractor.jl/blob/b23337a4b12d21104ff237cf0c72bcd2fe13a4f6/src/stage1/recurse.jl#L238-L247
# https://github.com/JuliaDiff/Diffractor.jl/blob/b23337a4b12d21104ff237cf0c72bcd2fe13a4f6/src/stage1/compiler_utils.jl

Base.iterate(compact::CC.IncrementalCompact, state) = CC.iterate(compact, state)
Base.iterate(compact::CC.IncrementalCompact) = CC.iterate(compact)
Base.getindex(c::CC.IncrementalCompact, args...) = CC.getindex(c, args...)
Base.setindex!(c::CC.IncrementalCompact, args...) = CC.setindex!(c, args...)
Base.setindex!(i::CC.Instruction, args...) = CC.setindex!(i, args...)

make_old(x) = x
make_old(x::Core.SSAValue) = x#CC.OldSSAValue(x.id)
make_old(x::Core.PhiNode) = Core.PhiNode(x.edges, Any[make_old.(x.values)...])

function collect_phi_nodes(ir)
    phi_nodes = [Core.PhiNode[] for _ in ir.cfg.blocks]
    for (i, stmt) in enumerate(ir.stmts)
        inst = stmt[:inst]
        if inst isa Core.PhiNode
            push!(
                phi_nodes[CC.block_for_inst(ir.cfg.blocks, i)],
                make_old(inst))
        end
    end
    return phi_nodes
end

function get_block_args(phi_nodes, from, to)
    values = []
    for phi in phi_nodes[to]
        i = findfirst(isequal(from), phi.edges)
        val = isnothing(i) ? nothing : phi.values[i]

        push!(values, val)
    end
    return values
end

function source2source(ir::CC.IRCode)
    phi_nodes = collect_phi_nodes(ir)
    @info phi_nodes
    compact = CC.IncrementalCompact(ir, #=allow_cfg_transforms=# true)

    # insert calls to start_block at the beginning of each block and add explicit gotos if they are missing.
    #   first block is handled separately:
    CC.insert_node_here!(
        compact,
        CC.NewInstruction(
            Expr(:call, Brutus.start_block),
            Any,
            CC.NoCallInfo(),
            Int32(1),
            CC.IR_FLAG_REFINED
        ))
    #   remainder of the blocks:
    for i in ir.cfg.index
        CC.insert_node!(
            compact,
            CC.OldSSAValue(i),
            CC.NewInstruction(
                Expr(:call, Brutus.start_block),
                Any,
                CC.NoCallInfo(),
                Int32(1),
                CC.IR_FLAG_REFINED
            )
        )
        if !(ir.stmts[i-1][:inst] isa Union{Core.GotoIfNot, Core.GotoNode, Core.ReturnNode})
            CC.insert_node!(
                compact,
                CC.OldSSAValue(i-1),
                CC.NewInstruction(
                    Core.GotoNode(CC.block_for_inst(ir.cfg.blocks, i)),
                    Any,
                    CC.NoCallInfo(),
                    Int32(1),
                    CC.IR_FLAG_REFINED
                ),
                true
            )
        end 
    end

    prev_bb = compact.active_bb
    current_phi = 1
    # Core of the transformation:
    # replace GotoIfNot, GotoNode, PhiNode, ReturnNode with calls to MLIR generation functions.
    for ((original_idx, idx), inst) in compact
        ssa = Core.SSAValue(idx)
        @info original_idx, idx inst
    
        if inst isa Union{Core.GotoIfNot, Core.GotoNode, Core.PhiNode, Core.ReturnNode}
            if inst isa Core.GotoIfNot
                compact[ssa][:inst] = Expr(:call, Core.tuple, get_block_args(phi_nodes, compact.active_bb-1, inst.dest)...)
                
                false_dest = inst.dest
                true_dest = compact.active_bb
                
                false_args = ssa
                # when cond is true, branch to next block
                true_args = Core.Compiler.insert_node_here!(
                    compact,
                    Core.Compiler.NewInstruction(
                        Expr(:call, Core.tuple, get_block_args(phi_nodes, compact.active_bb-1, compact.active_bb)...),
                        Any,
                        Core.Compiler.NoCallInfo(),
                        Int32(1),
                        Core.Compiler.IR_FLAG_REFINED
                    ),
                    true
                )
                Core.Compiler.insert_node_here!(
                    compact,
                    Core.Compiler.NewInstruction(
                        Expr(:call, Brutus.my_gotoifnot, inst.cond, true_dest, false_dest, true_args, false_args),
                        Any,
                        Core.Compiler.NoCallInfo(),
                        Int32(1),
                        Core.Compiler.IR_FLAG_REFINED
                    ),
                    true # insert within the current basic block, not at the start of the next one
                )    
            elseif inst isa Core.GotoNode
                compact[ssa][:inst] = Expr(:call, Core.tuple, get_block_args(phi_nodes, compact.active_bb-1, inst.label)...)
                Core.Compiler.insert_node_here!(
                    compact,
                    Core.Compiler.NewInstruction(
                        Expr(:call, Brutus.my_goto, inst.label, ssa),
                        Any,
                        Core.Compiler.NoCallInfo(),
                        Int32(1),
                        Core.Compiler.IR_FLAG_REFINED
                    ),
                    true
                )
            elseif inst isa Core.PhiNode
                # determine how many phi nodes came before in this block. 
                if prev_bb == compact.active_bb
                    current_phi += 1
                else
                    current_phi = 1
                    prev_bb = compact.active_bb
                end
    
                compact[ssa][:inst] = Expr(:call, Brutus.my_phi, compact[ssa][:type], current_phi)
            elseif inst isa Core.ReturnNode
                if isdefined(inst, :val)
                    compact[ssa][:inst] = Expr(:call, Brutus.my_return, inst.val)
                else
                    compact[ssa][:inst] = :nothing
                end
            end

            # Set general type Any and set flag to allow re-inferring type.
            compact[ssa][:type] = Any
            compact[ssa][:flag] = CC.IR_FLAG_REFINED
        elseif Meta.isexpr(inst, :invoke)
            _, called_func, args... = inst.args
            if called_func == Brutus.bool_conversion_intrinsic
                compact[ssa][:inst] = only(args)
            end
        end
    end
    @info "compacted IR" compact

    # Since ReturnNodes have disappeared, add an explicit `return nothing` at the end.
    CC.insert_node_here!(
        compact,
        CC.NewInstruction(
            Core.ReturnNode(nothing),
            Nothing,
            Int32(1)
        ))
    
    ir = CC.finish(compact)
    
    # manually set CFG to be a straight line, there might be a better way to do this.
    for (i, b) in enumerate(ir.cfg.blocks)
        deleteat!(b.preds, 1:length(b.preds))
        deleteat!(b.succs, 1:length(b.succs))
        insert!(b.preds, 1, i-1)
        insert!(b.succs, 1, i+1)
    end
    deleteat!(ir.cfg.blocks[1].preds, 1:length(ir.cfg.blocks[1].preds))
    deleteat!(ir.cfg.blocks[end].succs, 1:length(ir.cfg.blocks[end].succs))
    
    ir = CC.compact!(ir)
    
    
    # type inference
    interp = CC.NativeInterpreter()
    mi = get_toplevel_mi_from_ir(ir, @__MODULE__);
    ir = infer_ir!(ir, interp, mi)
    @info "transformed IR: " ir
    
    # inlining
    inline_state = CC.InliningState(interp)
    ir = CC.ssa_inlining_pass!(ir, inline_state, #=propagate_inbounds=#true)
    ir = CC.compact!(ir)

    # SROA + DCE
    ir = CC.sroa_pass!(ir, inline_state)
    ir = first(CC.adce_pass!(ir, inline_state))
    ir = CC.compact!(ir)

    # errors if invalid
    CC.verify_ir(ir)
    
    return Core.OpaqueClosure(ir; do_compile=true)
end

