using MLIR
includet("utils.jl")

using Brutus.Library: index, f32, i64, memref, MLIRMemref
import Brutus: MemRef, @mlirfunction, MLIRInterpreter, generate, unpack, entryblock, returntype, region, CodegenContext, simplify
using BenchmarkTools, MLIR, MacroTools

import MLIR.Dialects
using MLIR.Dialects: arith, gpu
using MLIR.IR: Context, @affinemap, Attribute, AffineMap, DenseArrayAttribute, Type, context
using MLIR.API: mlirRegisterAllPasses, mlirRegisterAllLLVMTranslations

ctx = IR.Context()
registerAllDialects!();
mlirRegisterAllPasses()
mlirRegisterAllLLVMTranslations(ctx.context)

@mlirfunction function threadIdx(dim::Symbol)::index
    oneoff = index(Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result)
    
    dimension = parse(IR.Attribute, "#gpu<dim $dim>")
    i = index(gpu.thread_id(; dimension) |> IR.result)
    i + oneoff
end

function threadIdx()::@NamedTuple{x::index, y::index, z::index}
    (; x=threadIdx(:x), y=threadIdx(:y), z=threadIdx(:z))
end

@mlirfunction function blockIdx()::@NamedTuple{x::index, y::index, z::index}
    oneoff = index(Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result)
    indices = map(('x', 'y', 'z')) do d
        dimension = parse(IR.Attribute, "#gpu<dim $d>")
        i = index(gpu.block_id(; dimension) |> IR.result)
        i + oneoff
    end
    return (; x=indices[1], y=indices[2], z=indices[3])
end

@mlirfunction function blockDim()::@NamedTuple{x::index, y::index, z::index}
    oneoff = Dialects.index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result
    indices = map(('x', 'y', 'z')) do d
        dimension = parse(IR.Attribute, "#gpu<dim $d>")
        i = gpu.block_dim(; dimension) |> IR.result
        index(arith.addi(i, oneoff) |> IR.result)
    end
    return (; x=indices[1], y=indices[2], z=indices[3])
end

function vadd(a, b, c)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]
    return
end

Base.code_ircode(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}}, interp=MLIRInterpreter())

@time generate(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}})

T_in = MLIRMemref{f32, 1, nothing, 1, nothing, 0}
@time generate(vadd, Tuple{T_in, T_in, T_in})

import Brutus: generate_return, generate_function

abstract type GPUFunc end
function generate_return(cg::CodegenContext{GPUFunc}, values; location)
    if (length(values) != 0)
        error("GPU kernel should return Nothing, got values of type $(typeof(values))")
    end
    return Dialects.gpu.return_(values; location)
end
function generate_function(cg::CodegenContext{GPUFunc})
    body = region(cg)
    input_types = IR.Type[
        IR.type(IR.argument(entryblock(cg), i))
        for i in 1:IR.nargs(entryblock(cg))]
    result_types = IR.Type[IR.Type.(unpack(returntype(cg)))...]
    ftype = IR.FunctionType(input_types, result_types)
    op = Dialects.gpu.func(;
        function_type=ftype,
        body
    )
    IR.attr!(op, "sym_name", IR.Attribute(String(nameof(cg.f))))
end

generate(CodegenContext{GPUFunc}(vadd, Tuple{memref{i64, 1}, memref{i64, 1}, memref{i64, 1}}))

function gpu_module(funcs::Vector{IR.Operation}, name="gpu_module")
    block = IR.Block()
    for f in funcs
        push!(block, f)
    end
    push!(block, Dialects.gpu.module_end())
    bodyRegion = IR.Region()
    push!(bodyRegion, block)
    op = Dialects.gpu.module_(;
        bodyRegion,
    )
    IR.attr!(op, "sym_name", IR.Attribute(name))
    op
end

gpu_mod_op = gpu_module([
    IR.attr!(
        generate(CodegenContext{GPUFunc}(vadd, Tuple{T_in, T_in, T_in})),
        "gpu.kernel", IR.UnitAttribute()),
    ]) |> simplify

mod = IR.Module()
push!(IR.body(mod), gpu_mod_op)
IR.attr!(IR.Operation(mod), "gpu.container_module", IR.UnitAttribute())

mlir_opt(mod, "gpu.module(strip-debuginfo,convert-gpu-to-nvvm),nvvm-attach-target,gpu-to-llvm")
mlir_opt(mod, "reconcile-unrealized-casts")

gpu_mod_op

data = API.mlirSerializeGPUModuleOp(gpu_mod_op)

print(String(data))

import CUDA: CuPtr, CuModule, CuFunction, CuArray, cudacall

md = CuModule(data.data)
vadd_cu = CuFunction(md, "vadd")

a = rand(Float32, 10)
b = rand(Float32, 10)
ad = CuArray(a)
bd = CuArray(b)

c = zeros(Float32, 10)
c_d = CuArray(c)
null = CuPtr{Cfloat}(0);
cudacall(vadd_cu,
            (CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat},
            CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat},
            CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}),
            null, ad, null, null, null,
            null, bd, null, null, null,
            null, c_d, null, null, null;
            threads=10)
c = Array(c_d)
c ≈ a+b

nothing

# open("compiled.out", "w") do file
#     write(file, Base.unsafe_wrap(Vector{Int8}, pointer(data.data), Int(data.length), own=false))
# end

# @mlirfunction function scf_yield(results)::Nothing
#     Dialects.scf.yield(results)
#     nothing
# end

# @mlirfunction function scf_for(body, initial_value::T, lb::index, ub::index, step::index)::T where T
#     @info "body IR" @nonoverlay Base.code_ircode(body, Tuple{index, T}, interp=Brutus.MLIRInterpreter())
#     region = @nonoverlay Brutus.generate(body, Tuple{index, T}, emit_region=true, skip_return=true)
#     op = Dialects.scf.for_(lb, ub, step, [initial_value]; results=IR.Type[IR.Type(T)], region)
#     return T(IR.result(op))
# end

# Brutus.generate(Tuple{index, index, index, i64, i64}, do_simplify=false) do lb, ub, step, initial, cst

#     a = scf_for(initial, lb, ub, step) do i, carry
#         b = scf_for(initial, lb, ub, step) do j, carry2
#             scf_yield((carry2 * cst, ))
#         end
#         scf_yield((b + cst, ))
#     end
#     return a
# end

