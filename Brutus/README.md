# Brutus

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
### MLIR
