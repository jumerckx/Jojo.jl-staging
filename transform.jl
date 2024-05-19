using MLIR
includet("utils.jl")

using Jojo.Library: index, f32, i64, memref, MLIRMemref
import Jojo.Library.Transform
import Jojo: MemRef, @intrinsic, MLIRInterpreter, generate, unpack, entryblock, returntype, region, CodegenContext, simplify
using BenchmarkTools, MLIR, MacroTools

import MLIR.Dialects
using MLIR.Dialects: arith, gpu, transform
using MLIR.IR: Context, @affinemap, Attribute, AffineMap, DenseArrayAttribute, Type, context
using MLIR.API: mlirRegisterAllPasses, mlirRegisterAllLLVMTranslations

ctx = IR.Context()
registerAllDialects!();
mlirRegisterAllPasses()
mlirRegisterAllLLVMTranslations(ctx.context)

@intrinsic function _as_intrinsic(f, args, kwargs)
    f(args...; kwargs...)
end
function as_intrinsic(f, args...; kwargs...)
    _as_intrinsic(f, args, kwargs)
end

function lower(op::Transform.AnyOp)
    f00 = Transform.structured_match(op, name="func.func")

    Transform.apply_patterns(f00) do
        as_intrinsic(Dialects.transform.apply_patterns_canonicalization)
        as_intrinsic(Dialects.transform.apply_patterns_linalg_tiling_canonicalization)
        return
    end

    as_intrinsic(Dialects.transform.apply_cse, f00)

    all_loops = Transform.structured_match(op, interface=Transform.LoopLikeInterface)

    as_intrinsic(Dialects.transform.apply_licm, all_loops)

    Transform.apply_patterns(f00) do 
        as_intrinsic(Dialects.transform.apply_patterns_linalg_fold_unit_extent_dims_via_reshapes)
        return
    end

    fv = Transform.vectorize_and_apply(f00)

    Transform.apply_patterns(fv) do 
        as_intrinsic(Dialects.transform.apply_patterns_canonicalization),
        as_intrinsic(Dialects.transform.apply_patterns_tensor_fold_tensor_subset_ops_into_vector_transfers)
        return
    end

    as_intrinsic(Dialects.transform.apply_cse, fv)
    as_intrinsic(Dialects.transform.structured_hoist_redundant_vector_transfers, fv; transformed=IR.Type(Transform.AnyOp))

    op = Transform.one_shot_bufferize(op, true, Int32(1)) # todo: bufferize_boundaries, function_boundary_type_conversion

    f = Transform.structured_match(op, name="func.func")
    Transform.apply_registered_pass(f, "buffer-deallocation-pipeline")

    fb = Transform.structured_match(op, name="func.func")
    Transform.apply_patterns(fb) do
        as_intrinsic(Dialects.transform.apply_patterns_canonicalization)
        return
    end
    as_intrinsic(Dialects.transform.apply_cse, fb)

    Transform.apply_patterns(fb) do 
        as_intrinsic(Dialects.transform.apply_patterns_vector_lower_contraction; lowering_strategy=Int32(3))
        as_intrinsic(Dialects.transform.apply_patterns_vector_lower_transfer; max_transfer_rank=1)
        as_intrinsic(Dialects.transform.apply_patterns_vector_lower_transpose)
        as_intrinsic(Dialects.transform.apply_patterns_vector_lower_shape_cast)
        return
    end

    Transform.apply_patterns(fb) do 
        as_intrinsic(Dialects.transform.apply_patterns_vector_transfer_to_scf)
        as_intrinsic(Dialects.transform.apply_patterns_memref_alloc_to_alloca)
        return
    end
    as_intrinsic(Dialects.transform.bufferization_buffer_loop_hoisting, fb)


    Transform.apply_patterns(fb) do 
        as_intrinsic(Dialects.transform.apply_patterns_memref_fold_memref_alias_ops)
        as_intrinsic(Dialects.transform.apply_patterns_canonicalization)
        return        
    end

    as_intrinsic(Dialects.transform.apply_cse, fb)
    

    return
end

abstract type NamedSequence end
import Jojo: generate_function, generate_return
generate_return(cg::CodegenContext{NamedSequence}, values; location) = Dialects.transform.yield(values; location)
function generate_function(cg::CodegenContext{NamedSequence})
    body = region(cg)
    input_types = IR.Type[
        IR.type(IR.argument(entryblock(cg), i))
        for i in 1:IR.nargs(entryblock(cg))]
    result_types = IR.Type[IR.Type.(unpack(returntype(cg)))...]
    ftype = IR.FunctionType(input_types, result_types)
    op = Dialects.transform.named_sequence(;
        sym_name="__transform_main",
        function_type=ftype,
        body
    )
end

named_sequence_op = CodegenContext{NamedSequence}(Tuple{Transform.AnyOp}) do op
    bias = Transform.structured_match(op; name="linalg.broadcast")
    generics = Transform.structured_match(op; name="linalg.generic")
    conv, relu = Transform.split_handle(generics, 2)

    relu, co = Transform.tile(Transform.ForAllTiling, relu, (0, 0, 0, 64))
    relu, n_y_xo = Transform.tile(Transform.ForAllTiling, relu, (1, 1, 5, 0))

    conv, co = Transform.fuse_into(co, conv)
    conv, n_y_xo = Transform.fuse_into(n_y_xo, conv)

    bias, co = Transform.fuse_into(co, bias)
    bias, n_y_xo = Transform.fuse_into(n_y_xo, bias)

    # to clean up IR:
    Transform.apply_patterns(Transform.structured_match(op, name="func.func")) do 
        nothing
    end

    red_fill, conv, combining, rz_ry_rx = Transform.tile_reduction(Transform.ForTiling, conv, (0, 0, 0, 0, 1, 1, 1))

    as_intrinsic(Dialects.transform.structured_generalize, bias; transformed=IR.Type(Transform.AnyOp))

    lower(op)

    return
end |> generate

named_sequence_mod = IR.Module()

IR.attr!(IR.Operation(named_sequence_mod), "transform.with_named_sequence", IR.UnitAttribute())
push!(IR.body(named_sequence_mod), named_sequence_op)

mod = parse(IR.Module, """
    !tinput = tensor<5x82x102x128xf32>
    !tfilter = tensor<128x3x3x128xf32>
    !tbias = tensor<128xf32>
    !toutput = tensor<5x80x100x128xf32>

    // Function containing the convolution. Note that its arguments and results are
    // tensors annotated with attributes from the `bufferization` dialect. These
    // attributes hint the bufferization pass to assume buffers can be directly
    // used for these tensors without reshaping.
    func.func @conv(
        %input: !tinput {bufferization.writable = false,
                        bufferization.access = "read",
                        bufferization.buffer_layout =
                            affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>},
        %filter: !tfilter {bufferization.writable = false,
                        bufferization.access = "read",
                        bufferization.buffer_layout =
                            affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>},
        %bias: !tbias {bufferization.writable = false,
                    bufferization.access = "read",
                    bufferization.buffer_layout = affine_map<(d0)->(d0)>},
        %output: !toutput {bufferization.writable = true,
                        bufferization.buffer_layout =
                            affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                        bufferization.access = "write"}) -> !toutput
    // This requests a C-compatible interface to be emitted for the function
    // when translating to LLVM IR.
    attributes { llvm.emit_c_interface }
    {
    // Bias. Using a named Linalg operation for brevity.
    %bias_init = tensor.empty() : !toutput
    %biased = linalg.broadcast ins(%bias : !tbias)
        outs(%bias_init : !toutput) dimensions = [0, 1, 2]

    // Convolution proper. While Linalg has named operations for 2D convolutions,
    // the one in the Halide example has an uncommon order of filter dimensions
    // and is not supported. It also takes the fitler as first argument. This
    // code recreates it faithfully using the generic form.
    %convolved = linalg.generic {
        iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"],
        indexing_maps = [
        affine_map<(n, y, x, c, rz, ry, rx) -> (rx, rz, ry, c)>,
        affine_map<(n, y, x, c, rz, ry, rx) -> (n, y+rz, x+ry, rx)>,
        affine_map<(n, y, x, c, rz, ry, rx) -> (n, y, x, c)>
        ]
    } ins(%filter, %input: !tfilter, !tinput) outs(%biased : !toutput) {
    ^bb0(%in: f32, %f: f32, %b: f32):
        // Note the fastmath attributes that allow operations to be recombined into
        //   %0 = math.fma %in, %f, %b : f32
        // later on and to reorder reductions.
        %m1 = arith.mulf %in, %f  {fastmath = #arith.fastmath<fast>} : f32
        %0 = arith.addf %b, %m1  {fastmath = #arith.fastmath<fast>} : f32
        linalg.yield %0 : f32
    } -> !toutput

    // ReLU is just a max(0, x).
    %c0 = arith.constant 0.0 : f32
    %relued = linalg.generic {
        iterator_types = ["parallel", "parallel", "parallel", "parallel"],
        indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> ()>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ]
    } ins(%c0, %convolved : f32, !toutput)
        outs(%output : !toutput) {
    ^bb0(%cst: f32, %in: f32, %out: f32):
        %0 = llvm.intr.maxnum(%cst, %in) : (f32, f32) -> f32
        linalg.yield %0 : f32
    } -> !toutput

    return %relued : !toutput
    }
""")

push!(IR.body(mod), IR.rmfromparent!(IR.Operation(named_sequence_mod)))

mlir_opt(mod, "transform-interpreter")
IR.rmfromparent!(IR.Operation(named_sequence_mod))
IR.Operation(mod) |> simplify
mlir_opt(mod, "math-uplift-to-fma")
IR.Operation(mod) |> simplify
mlir_opt(mod, "convert-bufferization-to-memref")
IR.Operation(mod) |> simplify
lowerModuleToLLVM(mod)
IR.Operation(mod) |> simplify

input = rand(Float32, (5, 82, 102, 128));
filter = rand(Float32, (128, 3, 3, 128));
bias = rand(Float32, 128);
output = zeros(Float32, (5, 80, 100, 128));

input, filter, bias, output = map((input, filter, bias, output)) do x
    out = MemRef(x)
    out.strides = reverse(Tuple([1, cumprod(reverse(size(x)))[1:end-1]...]))
    out
end


addr = jit(mod; opt=3)("_mlir_ciface_conv")
@ccall $addr(output::Ref{MemRef}, input::Ref{MemRef}, filter::Ref{MemRef}, bias::Ref{MemRef}, output::Ref{MemRef})::Nothing

mod = parse(IR.Module, """
    !tinput = tensor<5x82x102x128xf32>
    !tfilter = tensor<128x3x3x128xf32>
    !tbias = tensor<128xf32>
    !toutput = tensor<5x80x100x128xf32>

    // Function containing the convolution. Note that its arguments and results are
    // tensors annotated with attributes from the `bufferization` dialect. These
    // attributes hint the bufferization pass to assume buffers can be directly
    // used for these tensors without reshaping.
    func.func @conv(
        %input: !tinput {bufferization.writable = false,
                        bufferization.access = "read",
                        bufferization.buffer_layout =
                            affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>},
        %filter: !tfilter {bufferization.writable = false,
                        bufferization.access = "read",
                        bufferization.buffer_layout =
                            affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>},
        %bias: !tbias {bufferization.writable = false,
                    bufferization.access = "read",
                    bufferization.buffer_layout = affine_map<(d0)->(d0)>},
        %output: !toutput {bufferization.writable = true,
                        bufferization.buffer_layout =
                            affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>,
                        bufferization.access = "write"}) -> !toutput
    // This requests a C-compatible interface to be emitted for the function
    // when translating to LLVM IR.
    attributes { llvm.emit_c_interface }
    {
    // Bias. Using a named Linalg operation for brevity.
    %bias_init = tensor.empty() : !toutput
    %biased = linalg.broadcast ins(%bias : !tbias)
        outs(%bias_init : !toutput) dimensions = [0, 1, 2]

    // Convolution proper. While Linalg has named operations for 2D convolutions,
    // the one in the Halide example has an uncommon order of filter dimensions
    // and is not supported. It also takes the fitler as first argument. This
    // code recreates it faithfully using the generic form.
    %convolved = linalg.generic {
        iterator_types = ["parallel", "parallel", "parallel", "parallel",
                        "reduction", "reduction", "reduction"],
        indexing_maps = [
        affine_map<(n, y, x, c, rz, ry, rx) -> (rx, rz, ry, c)>,
        affine_map<(n, y, x, c, rz, ry, rx) -> (n, y+rz, x+ry, rx)>,
        affine_map<(n, y, x, c, rz, ry, rx) -> (n, y, x, c)>
        ]
    } ins(%filter, %input: !tfilter, !tinput) outs(%biased : !toutput) {
    ^bb0(%in: f32, %f: f32, %b: f32):
        // Note the fastmath attributes that allow operations to be recombined into
        //   %0 = math.fma %in, %f, %b : f32
        // later on and to reorder reductions.
        %m1 = arith.mulf %in, %f  {fastmath = #arith.fastmath<fast>} : f32
        %0 = arith.addf %b, %m1  {fastmath = #arith.fastmath<fast>} : f32
        linalg.yield %0 : f32
    } -> !toutput

    // ReLU is just a max(0, x).
    %c0 = arith.constant 0.0 : f32
    %relued = linalg.generic {
        iterator_types = ["parallel", "parallel", "parallel", "parallel"],
        indexing_maps = [
        affine_map<(d0, d1, d2, d3) -> ()>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
        ]
    } ins(%c0, %convolved : f32, !toutput)
        outs(%output : !toutput) {
    ^bb0(%cst: f32, %in: f32, %out: f32):
        %0 = llvm.intr.maxnum(%cst, %in) : (f32, f32) -> f32
        linalg.yield %0 : f32
    } -> !toutput

    return %relued : !toutput
    }
""")

mlir_opt(mod, "one-shot-bufferize{bufferize-function-boundaries=true}")
mlir_opt(mod, "convert-linalg-to-loops")
lowerModuleToLLVM(mod)

output.data[:, 1:10, 2, 2]