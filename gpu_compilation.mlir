// mlir-opt ./gpu_compilation.mlir --transform-interpreter -test-transform-dialect-erase-schedule -gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-features=+ptx76 cubin-format=isa"



// gpu.module @main_kernel {
//     gpu.func @main_kernel(%arg0: memref<16x16xf16>, %arg1: memref<16x8xf16>, %arg2: memref<16x8xf16>) kernel attributes {gpu.known_block_size = array<i32: 32, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
//             %block_id_x = gpu.block_id  x
//             %block_id_y = gpu.block_id  y
//             %block_id_z = gpu.block_id  z
//             %thread_id_x = gpu.thread_id  x
//             %thread_id_y = gpu.thread_id  y
//             %thread_id_z = gpu.thread_id  z
//             %grid_dim_x = gpu.grid_dim  x
//             %grid_dim_y = gpu.grid_dim  y
//             %grid_dim_z = gpu.grid_dim  z
//             %block_dim_x = gpu.block_dim  x
//             %block_dim_y = gpu.block_dim  y
//             %block_dim_z = gpu.block_dim  z
//             cf.br ^bb1
//         ^bb1:  // pred: ^bb0
//             linalg.matmul ins(%arg0, %arg1 : memref<16x16xf16>, memref<16x8xf16>) outs(%arg2 : memref<16x8xf16>)
//             gpu.return
//     }
// }

gpu.module @gpu_module {
  func.func @f(%a: i64, %b: i64) -> i64 {
    %c = arith.addi %a, %b : i64
    return %c : i64
  }
  gpu.func @"abc"(%arg0: memref<16x16xf16, strided<[16, 1]>>, %arg1: memref<16x8xf16, strided<[16, 1]>>, %arg2: memref<16x8xf16, strided<[16, 1]>>) kernel {
    %a = arith.constant 0 : i64
    func.call @f(%a, %a) : (i64, i64) -> i64
    linalg.matmul ins(%arg0, %arg1 : memref<16x16xf16, strided<[16, 1]>>, memref<16x8xf16, strided<[16, 1]>>) outs(%arg2 : memref<16x8xf16, strided<[16, 1]>>)
    gpu.return
  }
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
        transform.nvgpu.rewrite_matmul_as_mma_sync %matmul
        : (!transform.any_op) -> ()
        transform.yield
    }
}

// !lhs_memref_type = memref<16x16xf16>
// !rhs_memref_type = memref<16x8xf16>
// !res_memref_type = memref<16x8xf16>
// module attributes {gpu.container_module} {
//     gpu.module @gpu_module {
//         gpu.func @mma(%lhs: !lhs_memref_type, %rhs: !rhs_memref_type, %res: !res_memref_type) kernel {
//             linalg.matmul ins(%lhs, %rhs: !lhs_memref_type, !rhs_memref_type)
//                  outs(%res: !res_memref_type)
//             gpu.return
//         }
//     }
// }

// mlir-opt %s \
// | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm),nvvm-attach-target)' \
// | mlir-opt -gpu-to-llvm -gpu-module-to-binary \
// | mlir-cpu-runner \
//   --shared-libs=/home/jumerckx/masterthesis/llvm-project/llvm/install/relwithdebinfo/lib/libmlir_cuda_runtime.so \
//   --shared-libs=/home/jumerckx/masterthesis/llvm-project/llvm/install/relwithdebinfo/lib/libmlir_runner_utils.so \
//   --entry-point-result=void \

