using MLIR
includet("utils.jl")
using Brutus
import Brutus: MemRef, @mlirfunction, @code_mlir
using Brutus.Types
using BenchmarkTools, MLIR, MacroTools

using MLIR.Dialects: arith, index, linalg, transform, builtin
using MLIR.Dialects
using MLIR.IR
using MLIR.AffineUtils

using MLIR: IR, API
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

################################################################################################

@mlirfunction Base.:+(a::i64, b::i64)::i64 = i64(arith.addi(a, b))
@mlirfunction Base.:-(a::i64, b::i64)::i64 = i64(arith.subi(a, b))
@mlirfunction Base.:*(a::i64, b::i64)::i64 = i64(arith.muli(a, b))
@mlirfunction Base.:/(a::i64, b::i64)::i64 = i64(arith.divsi(a, b))
@mlirfunction Base.:>(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sgt))
@mlirfunction Base.:>=(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sge))
@mlirfunction function Base.getindex(A::memref{T}, i::Int)::T where T
    # this method can only be called with constant i since we assume arguments to `code_mlir`` to be MLIR types, not Julia types.
    i = Types.index(index.constant(; value=Attribute(i, IR.IndexType())) |> IR.get_result)
    A[i]
end
@mlirfunction function Base.getindex(A::memref{T, 1}, i::Types.index)::T where T
    oneoff = index.constant(; value=Attribute(1, IR.IndexType())) |> IR.get_result
    new_index = arith.subi(i, oneoff) |> IR.get_result
    T(Dialects.memref.load(A, [new_index]) |> IR.get_result)
end

square(a) = a*a
f(a, b) = (a>b) ? a+b : square(a)
g(a::AbstractVector) = a[2]
h(a, i) = a[i]

Base.code_ircode(f, (i64, i64))
@time op_f = Brutus.code_mlir(f, Tuple{i64, i64})
@assert IR.verify(op_f)

#region Running the code

# To run the code, we need to first:
#   put the operation in a MLIR module:
mod_f = IR.MModule(IR.Location())
push!(IR.get_body(mod_f), op_f);

#   lower all MLIR dialects to the llvm dialect:
pm_f = lowerModuleToLLVM(mod_f)

#   jit the function using an execution engine:
addr_f = jit(mod_f; opt=3)("_mlir_ciface_f")

# Finally, we call the function like a regular C-function:
@ccall $addr_f(3::Int, 2::Int)::Int

#endregion

Base.code_ircode(g, (memref{i64, 1},))
op_g = Brutus.code_mlir(g, Tuple{memref{i64, 1}})
IR.verify(op_g)

op_h = Brutus.code_mlir(h, Tuple{memref{i64, 1}, Types.index})
IR.verify(op_h)

#region Running the code

mod_h = IR.MModule(IR.Location())
push!(IR.get_body(mod_h), op_h);

pm_h = lowerModuleToLLVM(mod_h)

addr_h = jit(mod_h; opt=3)("_mlir_ciface_h")

a = [42, 43, 44]

@ccall $addr_h(MemRef(a)::Ref{MemRef}, 1::Int)::Int

#endregion

################################################################################################

using MLIR: AffineUtils

@mlirfunction function linalgyield(x::T)::Nothing where {T}
    linalg.yield([x])
    return nothing
end

import LinearAlgebra.mul!
@mlirfunction function mul!(Y::tensor{T, 2}, A::tensor{T, 2}, B::tensor{T, 2})::tensor{T, 2} where T
    indexing_maps = [
        (AffineUtils.@map (m, n, k)[] -> (m, k)),
        (AffineUtils.@map (m, n, k)[] -> (k, n)),
        (AffineUtils.@map (m, n, k)[] -> (m, n))
        ]
    indexing_maps = IR.Attribute.(API.mlirAffineMapAttrGet.(indexing_maps)) |> IR.ArrayAttribute
    iterator_types = IR.Attribute[parse(IR.Attribute, "#linalg.iterator_type<$type>") for type in ["parallel", "parallel", "reduction"]]
    iterator_types = IR.ArrayAttribute(iterator_types)
    matmul_region = @nonoverlay Brutus.code_mlir((a, b, y)->linalgyield(y+(a*b)), Tuple{T, T, T}; emit_region=true, ignore_returns=true)
    op = linalg.generic(
        [A, B],
        [Y],
        result_tensors=MLIRType[MLIRType(typeof(Y))];
        indexing_maps,
        iterator_types,
        region=matmul_region
    )
    return tensor{T, 2}(IR.get_result(op))
end

f(Y, A, B) = mul!(Y, A, B)

Base.code_ircode(f, (tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}))
op = Brutus.code_mlir(f, Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}})

@assert IR.verify(op)

################################################################################################

b = IR.Block()
arg1 = IR.push_argument!(b, parse(IR.MLIRType, "!transform.any_op"), IR.Location())

matched = transform.structured_match(
    arg1;
    results = parse(IR.MLIRType, "!transform.any_op"),
    ops = IR.ArrayAttribute([Attribute("linalg.generic")])
)
@show matched
tiled = transform.structured_tile_using_for(
    IR.get_result(matched), [];
    tiled_linalg_op=parse(IR.MLIRType, "!transform.any_op"),
    loops = [parse(IR.MLIRType, "!transform.any_op") for _ in 1:3],
    static_sizes=IR.Attribute(API.mlirDenseI64ArrayGet(IR.context(), 3, [2, 3, 4])),
    scalable_sizes=IR.Attribute(API.mlirDenseBoolArrayGet(IR.context(), 3, Int32[false, false, false]))
)
@show tiled
yield = transform.yield([])

push!.(Ref(b),[matched, tiled, yield])
body = Region()
push!(body, b)

ns = transform.named_sequence(;
    sym_name=Attribute("__transform_main"),
    function_type=MLIRType((parse(MLIRType, "!transform.any_op"),)=>()),
    body
)

bodyRegion = IR.Region()
b_module = IR.Block()
push!(b_module, ns)
push!(b_module, op)
push!(bodyRegion, b_module)
mod_op = builtin.module_(;
    additional_attributes=[NamedAttribute("transform.with_named_sequence", IR.Attribute(API.mlirUnitAttrGet(IR.context())))],
    bodyRegion
)

mod = MModule(mod_op)

# mlir_opt(mod, "one-shot-bufferize{bufferize-function-boundaries=true}")
mlir_opt(mod, "transform-interpreter")
mlir_opt(mod, "test-transform-dialect-erase-schedule")
# mlir_opt(mod, "one-shot-bufferize{function-boundary-type-conversion=infer-layout-map bufferize-function-boundaries}, expand-realloc, ownership-based-buffer-deallocation, canonicalize, buffer-deallocation-simplification, bufferization-lower-deallocations, cse, canonicalize")
mlir_opt(mod, "convert-linalg-to-loops")
mlir_opt(mod, "
  func.func(convert-vector-to-scf{
      full-unroll=false lower-tensors=false target-rank=1
  }),
  one-shot-bufferize{bufferize-function-boundaries=true},
  mem2reg,
  convert-linalg-to-loops,
  loop-invariant-code-motion,
  lower-affine,
  convert-scf-to-cf,
  canonicalize{ max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
  cse,
  func.func(convert-math-to-llvm{approximate-log1p=true}),
  expand-strided-metadata,
  lower-affine,
  finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},
  convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},
  convert-index-to-llvm{index-bitwidth=0}
")
@show mod
pm = lowerModuleToLLVM(mod)
mod

#   jit the function using an execution engine:
addr = jit(mod; opt=3)("_mlir_ciface_f")

@warn addr

# Finally, we call the function like a regular C-function:
@ccall $addr_f(3::Int, 2::Int)::Int

################################################################################################

mod = parse(IR.MModule, """
llvm.func @f(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %3 = llvm.insertvalue %arg0, %2[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.insertvalue %arg1, %3[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.insertvalue %arg2, %4[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.insertvalue %arg3, %5[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.insertvalue %arg5, %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.insertvalue %arg4, %7[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.insertvalue %arg6, %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.br ^bb1(%1 : i64)
  ^bb1(%10: i64):  // 2 preds: ^bb0, ^bb6
    %11 = llvm.icmp "slt" %10, %arg10 : i64
    llvm.cond_br %11, ^bb2(%1 : i64), ^bb7
  ^bb2(%12: i64):  // 2 preds: ^bb1, ^bb5
    %13 = llvm.icmp "slt" %12, %arg18 : i64
    llvm.cond_br %13, ^bb3(%1 : i64), ^bb6
  ^bb3(%14: i64):  // 2 preds: ^bb2, ^bb4
    %15 = llvm.icmp "slt" %14, %arg11 : i64
    llvm.cond_br %15, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %16 = llvm.getelementptr %arg8[%arg9] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %17 = llvm.mul %10, %arg12  : i64
    %18 = llvm.mul %14, %arg13  : i64
    %19 = llvm.add %17, %18  : i64
    %20 = llvm.getelementptr %16[%19] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %21 = llvm.load %20 : !llvm.ptr -> i64
    %22 = llvm.getelementptr %arg15[%arg16] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %23 = llvm.mul %14, %arg19  : i64
    %24 = llvm.mul %12, %arg20  : i64
    %25 = llvm.add %23, %24  : i64
    %26 = llvm.getelementptr %22[%25] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %27 = llvm.load %26 : !llvm.ptr -> i64
    %28 = llvm.getelementptr %arg1[%arg2] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %29 = llvm.mul %10, %arg5  : i64
    %30 = llvm.mul %12, %arg6  : i64
    %31 = llvm.add %29, %30  : i64
    %32 = llvm.getelementptr %28[%31] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %33 = llvm.load %32 : !llvm.ptr -> i64
    %34 = llvm.mul %21, %27  : i64
    %35 = llvm.add %33, %34  : i64
    llvm.store %35, %32 : i64, !llvm.ptr
    %36 = llvm.add %14, %0  : i64
    llvm.br ^bb3(%36 : i64)
  ^bb5:  // pred: ^bb3
    %37 = llvm.add %12, %0  : i64
    llvm.br ^bb2(%37 : i64)
  ^bb6:  // pred: ^bb2
    %38 = llvm.add %10, %0  : i64
    llvm.br ^bb1(%38 : i64)
  ^bb7:  // pred: ^bb1
    llvm.return %9 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  }
llvm.func @_mlir_ciface_f(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %8 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.call @f(%1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    llvm.store %24, %arg0 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
    llvm.return
}
""")

mod = parse(IR.MModule, """
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {transform.with_named_sequence} {
  "func.func"() <{function_type = (tensor<?x?xi64>, tensor<?x?xi64>, tensor<?x?xi64>) -> i64, sym_name = "f", llvm.emit_c_interface}> ({
    ^bb0(%arg0: tensor<?x?xi64>, %arg1: tensor<?x?xi64>, %arg2: tensor<?x?xi64>):
        %0 = "linalg.generic"(%arg1, %arg2, %arg0) <{indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
            ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
                %1 = "arith.muli"(%arg3, %arg4) : (i64, i64) -> i64
                %2 = "arith.addi"(%arg5, %1) : (i64, i64) -> i64
                "linalg.yield"(%2) : (i64) -> ()
        }) : (tensor<?x?xi64>, tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
//        "func.return"(%0) : (tensor<?x?xi64>) -> ()
        %a = arith.constant 1 : i64
        func.return %a : i64
    }) {llvm.emit_c_interface} : () -> ()
}
""")
mlir_opt(mod, "one-shot-bufferize{bufferize-function-boundaries=true}, convert-linalg-to-loops")
mlir_opt(mod, "
  func.func(convert-vector-to-scf{
      full-unroll=false lower-tensors=false target-rank=1
  }),
  one-shot-bufferize{bufferize-function-boundaries=true},
  mem2reg,
  convert-linalg-to-loops,
  loop-invariant-code-motion,
  lower-affine,
  convert-scf-to-cf,
  canonicalize{ max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},
  cse,
  func.func(convert-math-to-llvm{approximate-log1p=true}),
  expand-strided-metadata,
  lower-affine,
  finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},
  convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},
  convert-index-to-llvm{index-bitwidth=0}
")
pm = lowerModuleToLLVM(mod)


addr = jit(mod; opt=3)("_mlir_ciface_f")

struct ReturnType
    allocated_pointer::Ptr{Int}
    aligned_pointer::Ptr{Int}
    offset::Int
    sizes::NTuple{2, Int}
    strides::NTuple{2, Int}
end
function ReturnType(a::Array{Int, N}, size) where {N}
    allocated_pointer = a.ref.mem.ptr
    aligned_pointer = a.ref.ptr_or_offset
    offset = Int((aligned_pointer - allocated_pointer)//sizeof(Int))
    @show offset
    @assert offset == 0 "Arrays with Memoryref offset are, as of yet, unsupported."
    strides = Tuple([1, cumprod(size)[1:end-1]...])

    return ReturnType(
        allocated_pointer,
        aligned_pointer,
        offset,
        size,
        strides,
    )
end
ReturnType(a::Array) = ReturnType(a, size(a))

a = ones(Int64, 4, 3)
b = ones(Int64, 3, 4) .* 2
y = similar(a, (30, 30)) .* 0

a_, b_, y_ = ReturnType(a, (3, 2)), ReturnType(b, (2, 3)), ReturnType(y, (3, 3))


@ccall $addr(y_::Ref{ReturnType}, a_::Ref{ReturnType}, b_::Ref{ReturnType})::Int

@show y

################################################################################################

schedules = [
    # no transformation:
    """
    // transform.yield
    """,
    # permute loops: (1, 2, 3) -> (2, 3, 1):
    """
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %0[0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %tiled_linalg_op_0, %loops_1 = transform.structured.tile_using_for %tiled_linalg_op[0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield 
    """,
    # tile loops: (6, 1, 1):
    """
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %0[6] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
    """,
    # tile loops: (6, 6, 1):
    """
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:2 = transform.structured.tile_using_for %0[6, 6] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield 
    """,
    # tile loops: (6, 6, 6):
    """
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0[6, 6, 6] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield 
    """,
    # tile loops: (8, 8, 8):
    """
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_linalg_op, %loops:3 = transform.structured.tile_using_for %0[8, 8, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield 
    """,
]

function mlir_bench(schedule::String)
    mod = parse(IR.MModule, """
    #map = affine_map<(d0, d1, d2) -> (d0, d2)>
    #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
    #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

    module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
        $(schedule)
        %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op

    // Simplify the code as tiling and fusion may have produced a lot of
    // operations computing tensor subsets and loop ranges, some of which may be
    // duplicated or excessively complex. Simplification involving
    // canonicalization, common subexpression elimination, loop invariant code
    // motion and various rewrite patterns can be applied directly from the
    // transform dialect. Furthermore, an arbitrary combination of rewrite
    // patterns can be applied in one sweep to a given scope, a functionality
    // that cannot be achieved with conventional compiler passes that apply each
    // group of patterns separately (at least without creating a new pass for
    // each combination of pattern groups).
    transform.apply_patterns to %f00 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %f00 : !transform.any_op
    %all_loops = transform.structured.match interface{LoopLikeInterface}
      in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op

    // Tiling-by-one as a way of materializing loops produced operations
    // processing 4+D types where only a handful of dimension isn’t unit-sized,
    // e.g., tensor<1x1x1x5x64xf32> where 5 and 64 are tile sizes. Remove such
    // unit dimensions before vectorization, for clarity.
    transform.apply_patterns to %f00 {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op

    }

    "func.func"() <{function_type = (tensor<?x?xi64>, tensor<?x?xi64>, tensor<?x?xi64>) -> i64, sym_name = "f"}> ({
        ^bb0(%arg0: tensor<?x?xi64>, %arg1: tensor<?x?xi64>, %arg2: tensor<?x?xi64>):
            %0 = "linalg.generic"(%arg1, %arg2, %arg0) <{indexing_maps = [#map, #map1, #map2], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
                ^bb0(%arg3: i64, %arg4: i64, %arg5: i64):
                    %1 = "arith.muli"(%arg3, %arg4) : (i64, i64) -> i64
                    %2 = "arith.addi"(%arg5, %1) : (i64, i64) -> i64
                    "linalg.yield"(%2) : (i64) -> ()
            }) : (tensor<?x?xi64>, tensor<?x?xi64>, tensor<?x?xi64>) -> tensor<?x?xi64>
            %a = arith.constant 1 : i64
            func.return %a : i64
        }) {llvm.emit_c_interface} : () -> ()
    }
    """)
    mlir_opt(mod, "one-shot-bufferize{bufferize-function-boundaries=true}")
    mlir_opt(mod, "transform-interpreter")
    mlir_opt(mod, "test-transform-dialect-erase-schedule")
    mlir_opt(mod, "convert-linalg-to-loops")
    mlir_opt(mod, "mem2reg")
    display(mod)
    pm = lowerModuleToLLVM(mod)
    addr = jit(mod; opt=3)("_mlir_ciface_f")
    f_mlir(y, a, b) = @ccall $addr(MemRef(y)::Ref{MemRef}, MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef})::Int
    return f_mlir
end


results = []
for (i, schedule) in enumerate(schedules)
    a = rand(1024, 1024)
    b = rand(1024, 1024)
    y = rand(1024, 1024)
    println("### Schedule $i ###")
    f = mlir_bench(schedule)
    push!(results, @benchmark $f($y, $a, $b))
    display(results[end])
    println("")
end


# @ccall $addr(MemRef(y)::Ref{MemRef}, MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef})::Int

################################################################################################

@mlirfunction i64(i::Int)::i64 = i64(arith.constant(value=i) |> IR.get_result)

function f(a::AbstractArray)
    s = eltype(a)(0)
    for i in eachindex(IndexLinear(), a)
        s += a[i]
    end
    return s
end
Base.code_ircode(f, (memref{i64, 2},))

# compare: 
Base.code_ircode(:, (i64, i64))
Base.code_ircode(:, (Int, Int))

################################################################################################

mod = parse(IR.MModule, """
func.func @f(%arg0: memref<?x?xi64, strided<[?, ?], offset: ?>>, %arg1: memref<?x?xi64, strided<[?, ?], offset: ?>>, %arg2: memref<?x?xi64, strided<[?, ?], offset: ?>>) -> i64 attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_i64 = arith.constant 1 : i64
    %dim = memref.dim %arg1, %c0 : memref<?x?xi64, strided<[?, ?], offset: ?>>
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xi64, strided<[?, ?], offset: ?>>
    %dim_1 = memref.dim %arg2, %c1 : memref<?x?xi64, strided<[?, ?], offset: ?>>
    scf.for %arg3 = %c0 to %dim step %c1 {
      scf.for %arg4 = %c0 to %dim_1 step %c1 {
        scf.for %arg5 = %c0 to %dim_0 step %c1 {
          %0 = memref.load %arg1[%arg3, %arg5] : memref<?x?xi64, strided<[?, ?], offset: ?>>
          %1 = memref.load %arg2[%arg5, %arg4] : memref<?x?xi64, strided<[?, ?], offset: ?>>
          %2 = memref.load %arg0[%arg3, %arg4] : memref<?x?xi64, strided<[?, ?], offset: ?>>
          %3 = arith.muli %0, %1 : i64
          %4 = arith.addi %2, %3 : i64
          memref.store %4, %arg0[%arg3, %arg4] : memref<?x?xi64, strided<[?, ?], offset: ?>>
        }
      }
    }
    return %c1_i64 : i64
  }
""")
lowerModuleToLLVM(mod)
addr = jit(mod; opt=3)("_mlir_ciface_f")

a = rand(Int64, 4, 3)
b = rand(Int64, 3, 4) .* 2
y = similar(a, (4, 4)) .* 0


@ccall $addr(MemRef(y)::Ref{MemRef}, MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef})::Int
