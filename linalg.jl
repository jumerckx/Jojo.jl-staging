### Setup ###
using MLIR
includet("utils.jl")
using MLIR: IR, API
using MLIR.IR: Value, NamedAttribute, Location
using MLIR.Dialects: arith, cf
ctx = IR.Context()
registerAllDialects!();
API.mlirRegisterAllPasses()
API.mlirRegisterAllLLVMTranslations(ctx.context)

import MLIR.Dialects: linalg, arith, transform

mod = IR.parse(IR.Module, """
!T = f64
!matrix_type_A = tensor<1024x1024x!T>
!matrix_type_B = tensor<1024x1024x!T>
!matrix_type_C = tensor<1024x1024x!T>

#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmul_trait = {
  iterator_types = ["parallel", "parallel", "reduction"],
  indexing_maps = #matmul_accesses
}

func.func @mm(%A: memref<?x?xf32, strided<[1, ?]>>, %B: memref<?x?xf32, strided<[1, ?]>>, %C: memref<?x?xf32, strided<[1, ?]>>) {
  linalg.generic #matmul_trait
  ins(%A, %B : memref<?x?xf32, strided<[1, ?]>>, memref<?x?xf32, strided<[1, ?]>>)
  outs(%C : memref<?x?xf32, strided<[1, ?]>>)
  {
    ^bb0(%a: f32, %b: f32, %c: f32) :
      %d = arith.mulf %a, %b: f32
      %e = arith.addf %c, %d: f32
      linalg.yield %e : f32
  }
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %module_op
      : (!transform.any_op) -> !transform.op<"linalg.generic">
    transform.structured.pack_greedily %matmul
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [0, 1, 2]
      : (!transform.op<"linalg.generic">) -> !transform.op<"linalg.generic">
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
  
    %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.unpack">
    transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
      -> (!transform.op<"tensor.empty">,
          !transform.op<"linalg.transpose">,
          !transform.op<"tensor.collapse_shape">,
          !transform.op<"tensor.extract_slice">)

    transform.yield
  }
}

""")


ops = []
for op in IR.OperationIterator(first(IR.BlockIterator(first(IR.RegionIterator(first(IR.OperationIterator(first(IR.BlockIterator(first(IR.RegionIterator(IR.get_operation(mod))))))))))))
    push!(ops, op)
end
ops[1]


m = API.mlirAffineDimExprGet(IR.context().context, 0)
n = API.mlirAffineDimExprGet(IR.context().context, 1)
k = API.mlirAffineDimExprGet(IR.context().context, 2)

map = API.mlirAffineMapGet(IR.context().context, 3, 0, 2, [m, k])
attr_a = IR.Attribute(API.mlirAffineMapAttrGet(map))

map = API.mlirAffineMapGet(IR.context().context, 3, 0, 2, [k, n])
attr_b = IR.Attribute(API.mlirAffineMapAttrGet(map))

map = API.mlirAffineMapGet(IR.context().context, 3, 0, 2, [m, n])
attr_c = IR.Attribute(API.mlirAffineMapAttrGet(map))

indexing_maps = IR.ArrayAttribute([attr_a, attr_b, attr_c])
iterator_types = IR.ArrayAttribute(parse.(Ref(IR.Attribute), [
  "#linalg.iterator_type<parallel>",
  "#linalg.iterator_type<parallel>",
  "#linalg.iterator_type<reduction>"]))

dummy = IR.Block()
TensorType = parse(IR.Type, "tensor<?x?xf64>")
a = IR.push_argument!(dummy, TensorType)
b = IR.push_argument!(dummy, TensorType)
c = IR.push_argument!(dummy, TensorType)

# a = IR.push_argument!(dummy, IR.Type(Array{Float64, 2}))
# b = IR.push_argument!(dummy, IR.Type(Array{Float64, 2}))
# c = IR.push_argument!(dummy, IR.Type(Array{Float64, 2}))

matmul_block = IR.Block()
a_el, b_el, c_el = IR.push_argument!.(Ref(matmul_block), [IR.Type(Float64) for _ in 1:3])

d = IR.result(push!(matmul_block, arith.mulf(a_el, b_el)))
e = IR.result(push!(matmul_block, arith.addf(c_el, d)))
push!(matmul_block, linalg.yield([e]));

matmul_region = IR.Region()
push!(matmul_region, matmul_block)

mm = push!(dummy, linalg.generic(
  [a, b],
  [c];
  # result_tensors=IR.Type[],
  result_tensors=IR.Type[TensorType],
  indexing_maps,
  iterator_types,
  region=matmul_region
))

IR.verify(mm)

x = [(:n, :k), (:k, :m)]=>(:n, :m)
typeof(x)

# function f(description::Pair)
#   inputs, output = description

#   dims = Symbol[]
#   for x in inputs
#     map(x) do d
#       i = findfirst(isequal(d), dims)
#       if (i == nothing)
#         push!(dims, d)
#         i = length(dims)
#       end
#     end
      

#     push!.(Ref(dims), x)
#   end
#   push!.(Ref(dims), output)

#   return dims
# end

# f(x)

parse(IR.Module, """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
      : (!transform.any_op) -> !transform.op<"linalg.matmul">
    transform.structured.pack_greedily %matmul
        matmul_packed_sizes = [8, 16, 32] matmul_inner_dims_order = [0, 1, 2]
      : (!transform.op<"linalg.matmul">) -> !transform.op<"linalg.generic">
    %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.pack">
    transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
  
    %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"tensor.unpack">
    transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
      -> (!transform.op<"tensor.empty">,
          !transform.op<"linalg.transpose">,
          !transform.op<"tensor.collapse_shape">,
          !transform.op<"tensor.extract_slice">)

    transform.yield
  }
}
""")

transform_body = IR.Region()
b0 = push!(transform_body, IR.Block())
any_op = IR.push_argument!(b0, IR.Type(IR.parse(IR.Type, "!transform.any_op")))

matched_op = push!(b0, transform.structured_match(any_op, results=parse(IR.Type, "!transform.op<\"linalg.generic\">"), ops=IR.ArrayAttribute([IR.Attribute("linalg.generic")]))) |> IR.result


# transform.named_sequence(
#   body=transform_body,
#   sym_name=IR.Attribute("__transform_main"),
#   function_type=IR.Attribute(IR.Type((parse(IR.Type, "!transform.any_op"), )=>())),
#   arg_attrs=IR.ArrayAttribute([parse(IR.Attribute, "{transform.readonly}")]),
# )

IR.Type((parse(IR.Type, "!transform.any_op"), )=>())