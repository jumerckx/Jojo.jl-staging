- [Overview of the Repository](#overview-of-the-repository)
  - [Examples](#examples)
- [Installation Instructions](#installation-instructions)
  - [Building MLIR](#building-mlir)


## Overview of the Repository

`generate.jl` contains the main MLIR code generation interpreter loop.
`codegenContext.jl` contains the `AbstractCodegenContext` and default `CodegenContext`.

Inlining policy changes and Boolean conversions are handled by the MLIRInterpreter defined in `abstrat.jl`.

`src2src.jl` is an experiment to generate MLIR code by transforming the Julia IR code and executing it, instead manual abstract interpretation over the regular Julia SSA IR.

### Examples

|Filename|Description|
|----------------|---|
| `simple.jl`    |Illustrates the basic MLIR code generation process + execution on a simple function with control flow.|
| `regions.jl`   |Example of a custom `CodegenContext` and intrinsic functions that generate operations with regions through the use of higher-order functions.|
| `einsum.jl`    | An example of a DSL for generating `linalg.generic` operations given an einsum description. |
| `gpu_vadd.jl`  |`vadd` kernel similar to CUDA.jl example.|
| `gpu_wmma.jl`  |Vendor-neutral WMMA operations.|
| `gpu_mma_from_linalg.jl` | Example of transforming a `m16n8k16` `linalg.matmul` into hardware-specific NVPTX (`mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`).|
| `transform.jl` |Port of the [MLIR transform tutorial](https://mlir.llvm.org/docs/Tutorials/transform/ChH/) that reproduces a Halide schedule.|
|`src2src_example.jl` | Demonstration of the approach of generating MLIR code by transforming and executing Julia SSA IR. |


## Installation Instructions
The code depends on Julia compiler internals and therefore won't work with many different Julia versions.
Development and testing was done with `1.11.0-beta1`.
```
Julia Version 1.11.0-beta1
Commit 08e1fc0abb9 (2024-04-10 08:40 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 12 × Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
  WORD_SIZE: 64
  LLVM: libLLVM-16.0.6 (ORCJIT, skylake)
```
Instantiate the packages in `Manifest.toml`:
```jl
] instantiate
```

This will install forks with small changes that haven't yet been upstreamed for:
* [CassetteOverlay](https://github.com/jumerckx/CassetteOverlay.jl.git)
* [CodeInfoTools](https://github.com/jumerckx/CodeInfoTools.jl.git)
* [MLIR.jl](https://github.com/jumerckx/MLIR.jl.git)

### Building MLIR

To run the examples, a path to the MLIR C API library `libMLIR-C.so` needs to be provided.
This can be done by creating `LocalPreferences.toml` file in the root directory containing:
```
[MLIR_jll]
mlir_c_path = "[LLVM install directory]/lib/libMLIR-C.so"
```

A sufficiently recent version of LLVM/MLIR might work, but development was done on commit [8a237ab7d9022d24441544ba25be480f0c944f5a](https://github.com/llvm/llvm-project/commits/8a237ab7d9022d24441544ba25be480f0c944f5a).

Alternatively, my fork of LLVM includes a few additional commits that work around a build error, as well as support for extracting generated PTX to run with CUDA.jl, and some commits from a stale pull-request upstream that allows lowering WMMA operations for AMD: https://github.com/jumerckx/llvm-project/

LLVM has to be built with the following flags:
```sh
LLVM_INSTALL_UTILS=ON
LLVM_BUILD_LLVM_DYLIB=ON
LLVM_EXTERNAL_MLIR_SOURCE_DIR=/home/jumerckx/llvm-project/mlir
LLVM_TARGETS_TO_BUILD=host;NVPTX;AMDGPU # NVPTX and AMDGPU are only needed for NVIDIA and AMD GPUs respectively.
LLVM_TOOL_MLIR_BUILD=ON
MLIR_BUILD_MLIR_C_DYLIB=ON
MLIR_ENABLE_CUDA_RUNNER=ON
```