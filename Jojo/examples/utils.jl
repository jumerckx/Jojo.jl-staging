using InteractiveUtils, CassetteOverlay
import MLIR: IR, API

const mlir_runner_utils = joinpath(splitpath(API.MLIR_jll.mlir_c)[1:end-1]..., "libmlir_runner_utils.so")
const mlir_c_runner_utils = joinpath(splitpath(API.MLIR_jll.mlir_c)[1:end-1]..., "libmlir_c_runner_utils.so")

macro code_ircode(ex0...)
    thecall = InteractiveUtils.gen_call_with_extracted_types_and_kwargs(@__MODULE__, :(Base.code_ircode), ex0)
    quote
        local results = $thecall
        length(results) == 1 ? results[1] : results
    end
end

function registerAllDialects!()
    ctx = IR.context()
    registry = API.mlirDialectRegistryCreate()
    API.mlirRegisterAllDialects(registry)
    API.mlirContextAppendDialectRegistry(ctx, registry)
    API.mlirDialectRegistryDestroy(registry)

    API.mlirContextLoadAllAvailableDialects(ctx)
    return registry
end

function mlir_opt(mod::IR.Module, pipeline::String)
    pm = IR.PassManager()
    IR.add_pipeline!(IR.OpPassManager(pm), pipeline)
    status = API.mlirPassManagerRunOnOp(pm, IR.Operation(mod).operation)
    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return mod
end

function lowerModuleToLLVM(mod::IR.Module)
    pm = IR.PassManager()

    IR.add_pipeline!(
        IR.OpPassManager(pm), 
        "func.func(convert-vector-to-scf{full-unroll=false lower-tensors=false target-rank=1}),func.func(convert-linalg-to-loops),lower-affine,convert-scf-to-cf,canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=true test-convergence=false top-down=true},cse,convert-vector-to-llvm{enable-amx=false enable-arm-neon=false enable-arm-sve=false enable-x86vector=false force-32bit-vector-indices=true reassociate-fp-reductions=false},func.func(convert-math-to-llvm{approximate-log1p=true}),expand-strided-metadata,lower-affine,finalize-memref-to-llvm{index-bitwidth=0 use-aligned-alloc=false use-generic-functions=false},convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},convert-index-to-llvm{index-bitwidth=0}"
    )

    IR.add_owned_pass!(pm, API.mlirCreateConversionReconcileUnrealizedCasts())
    status = API.mlirPassManagerRunOnOp(pm, IR.Operation(mod).operation)

    if status.value == 0
        error("Unexpected failure running pass failure")
    end
    return mod
end
  
function jit(mod::IR.Module; opt=0)
    paths = Base.unsafe_convert.(Ref(API.MlirStringRef), [mlir_c_runner_utils, mlir_runner_utils])
    jit = API.mlirExecutionEngineCreate(
        mod,
        opt,
        length(paths), # numPaths
        paths, # libPaths
        true # enableObjectDump
    )
    function lookup(name)
        addr = API.mlirExecutionEngineLookup(jit, name)
        (addr == C_NULL) && error("Lookup failed.")
        return addr
    end
    return lookup
end

function jit(op::IR.Operation; opt=0)
    mod = IR.Module(IR.Location())
    push!(IR.get_body(mod), op)
    lowerModuleToLLVM(mod)
    jit(mod; opt)
end
