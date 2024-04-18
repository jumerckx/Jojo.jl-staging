using MLIR
includet("utils.jl")
using Brutus
import Brutus: MemRef, @intrinsic, @code_mlir
using Brutus.Types
using Brutus.Types: MLIRFloat
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

@intrinsic (Base.:+(a::T, b::T)::T) where {T <: MLIRFloat} = T(arith.addf(a, b) |> IR.result)
@intrinsic (Base.:-(a::T, b::T)::T) where {T <: MLIRFloat} = T(arith.subf(a, b) |> IR.result)
@intrinsic (Base.:*(a::T, b::T)::T) where {T <: MLIRFloat} = T(arith.mulf(a, b) |> IR.result)
@intrinsic (Base.:/(a::T, b::T)::T) where {T <: MLIRFloat} = T(arith.divf(a, b) |> IR.result)

@intrinsic Base.:+(a::i64, b::i64)::i64 = i64(arith.addi(a, b))
@intrinsic Base.:-(a::i64, b::i64)::i64 = i64(arith.subi(a, b))
@intrinsic Base.:*(a::i64, b::i64)::i64 = i64(arith.muli(a, b))
@intrinsic Base.:/(a::i64, b::i64)::i64 = i64(arith.divsi(a, b))
@intrinsic Base.:>(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sgt))
@intrinsic Base.:>=(a::i64, b::i64)::Bool = i1(arith.cmpi(a, b, predicate=arith.Predicates.sge))
@intrinsic function Base.:+(a::T, b::T)::T where {I<:Types.MLIRInteger, T<:Union{I, tensor{I}}}
    T(IR.result(arith.addi(a, b)))
end
@intrinsic function Base.:*(a::T, b::T)::T where {I<:Types.MLIRInteger, T<:Union{I, tensor{I}}}
    T(IR.result(arith.muli(a, b)))
end
@intrinsic function Base.getindex(A::memref{T}, i::Int)::T where T
    # this method can only be called with constant i since we assume arguments to `code_mlir` to be MLIR types, not Julia types.
    i = Types.index(index.constant(; value=Attribute(i, IR.IndexType())) |> IR.result)
    A[i]
end
@intrinsic function Base.getindex(A::memref{T, 1}, i::Types.index)::T where T
    oneoff = index.constant(; value=Attribute(1, IR.IndexType())) |> IR.result
    new_index = arith.subi(i, oneoff) |> IR.result
    T(Dialects.memref.load(A, [new_index]) |> IR.result)
end

@intrinsic function linalgyield(x::T)::Nothing where {T}
    linalg.yield([x])
    return nothing
end

function maps(output_indices, inputs_indices)
    parallel, reduction = parse.(Ref(IR.Attribute), (
        "#linalg.iterator_type<parallel>",
        "#linalg.iterator_type<reduction>"
    ))

    # get all index symbols used in the inputs, the output can't contain any different symbols.
    symbols = Dict()
    iterator_types = IR.Attribute[]
    for arg in inputs_indices
        map(arg) do i
            get!(symbols, i) do 
                # indices that don't occur in output have to be reduced
                push!(iterator_types, i ∉ output_indices ? reduction : parallel)
                
                return API.mlirAffineDimExprGet(context(), length(symbols))
            end
        end
    end

    # function to create affinemap
    function get_map(indices)
        exprs = map(indices) do i
            symbols[i]
        end
        API.mlirAffineMapGet(
            context(), length(symbols),
            0, length(indices),
            # [exprs...]
            collect(exprs)
        ) |> IR.AffineMap
    end
    indexing_maps = IR.AffineMap[get_map.(inputs_indices)..., get_map(output_indices)]

    iterator_types = IR.ArrayAttribute(iterator_types)
    indexing_maps = IR.Attribute.(API.mlirAffineMapAttrGet.(indexing_maps)) |> IR.ArrayAttribute
    return indexing_maps, iterator_types
end

struct Einsum{I, O} end
@generated function maps(::Einsum{I, O}) where {I, O}
    return maps(O, I)
end
function Einsum(desc::Pair{T}) where T
    return Einsum{desc.first, desc.second}()
end

@intrinsic function (E::Einsum{I, O})(Y::T, XS::Vararg{tensor})::T where {I, O, T<:tensor}
    indexing_maps, iterator_types = maps(E)
    region = @nonoverlay Brutus.code_mlir(
        (xs, y)->linalgyield(y+prod(xs)),
        # (y, xs)->linalgyield(y+prod(xs)),
        Tuple{Tuple{eltype.(XS)...}, eltype(Y)},
        emit_region=true, ignore_returns=true
    )
    op = linalg.generic(
        XS,
        [Y];
        result_tensors=IR.Type[IR.Type(T)],
        indexing_maps,
        iterator_types,
        region
    )
    return tensor{T, 2}(IR.result(op))
end

Brutus.code_mlir(Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}}) do Y, A, B
    f = Einsum(((:i, :k), (:k, :j))=>(:i, :j))
    f(Y, A, B)
end

# f(Y, A, B) = Einsum(((:i, :k), (:k, :j))=>(:i, :j))(Y, A, B)
# Brutus.code_mlir(f, Tuple{tensor{i64, 2}, tensor{i64, 2}, tensor{i64, 2}})

Base.code_ircode(
        (args)->linalgyield(args[end]+prod(args[1:end-1])),
        Tuple{Tuple{eltype(tensor{i64}), eltype.((tensor{i64}, tensor{i64}))...}}
    )

###########################################################################

Base.promote_rule(::Type{T}, ::Type{I}) where {T<:Brutus.Types.MLIRInteger, I<:Integer} = T

@intrinsic function Base.convert(::Type{T}, x::Integer)::T where {T <: Brutus.Types.MLIRInteger}
    op = arith.constant(value=Attribute(x), result=IR.Type(T))
    T(IR.result(op))
end

Brutus.code_mlir(Tuple{i64}) do a
    a+2
end


Base.promote_rule(::Type{Brutus.Types.MLIRIndex}, ::Type{I}) where {I<:Integer} = Brutus.Types.MLIRIndex

@intrinsic function Base.convert(::Type{T}, x::Integer)::T where {T<:Brutus.Types.MLIRIndex}
    op = index.constant(value=Attribute(x, IR.IndexType()), result=IR.Type(T))
    T(IR.result(op))
end
@intrinsic function Base.:+(a::T, b::T)::T where {T<:Brutus.Types.MLIRIndex}
    T(IR.result(index.add(a, b)))
end

Brutus.code_mlir(Tuple{i64, Types.index}) do a, b
    (a+2, b+1)
end

Base.code_ircode(Tuple{i64, Types.index}) do a, b
    (a+2, b+1)
end

###########################################################################

Base.promote_rule(::Type{Brutus.Types.MLIRInteger{A}}, y::Type{Brutus.Types.MLIRInteger{B}}) where {A, B} = Brutus.Types.MLIRInteger{max(A, B)}

@intrinsic function Base.convert(::Type{T}, x::Brutus.Types.MLIRInteger{X})::T where {N, T<:Brutus.Types.MLIRInteger{N}, X}
    if (N > X)
        op = arith.extsi(x, out=IR.Type(T))
    else
        @warn "Converting from $(typeof(x)) to $T will unconditionally truncate the value. This differs from conversion between Julia integers."
        op = arith.trunci(x, out=IR.Type(T))
    end
    T(IR.result(op))
end

Base.code_ircode(Tuple{tensor{f64, 2}, tensor{f64, 2}, tensor{f64, 2}}) do Y, A, B
    f = Einsum(((:i, :k), (:k, :j))=>(:i, :j))
    f(Y, A, B)
end

op() = Brutus.code_mlir(Tuple{tensor{f32, 2}, tensor{f32, 2}, tensor{f32, 2}}, fname="f") do Y, A, B
    f = Einsum(((:i, :k), (:k, :j))=>(:i, :j))
    f(Y, A, B)
end

includet("transform.jl")
using .Transform

Base.code_ircode(Tuple{Transform.AnyOp}) do op
    matched = Transform.structured_match(op, "linalg.generic")
    matched = Transform.structured_tile_using_for(matched, (0, 1, 0))
    matched = Transform.structured_tile_using_for(matched, (0, 0, 1))
    matched = Transform.structured_tile_using_for(matched, (1, 0, 0))
    func = Transform.match(op, ("func.func", ))
    Transform.apply_registered_pass(func, pass_name="convert-linalg-to-affine-loops")
    Transform.yield()


end

begin
    function ns()
        region = Brutus.code_mlir(Tuple{}, ignore_returns=true, emit_region=true) do 
            Transform.named_sequence() do op
                matched = Transform.structured_match(op, "linalg.generic")

                # Transform.structured_tile_using_for(matched, (128, 0, 128))

                matched = Transform.structured_tile_using_for(matched, (1, 0, 0))
                matched = Transform.structured_tile_using_for(matched, (0, 0, 1))
                matched = Transform.structured_tile_using_for(matched, (0, 1, 0))

                # func = Transform.structured_match(op, "func.func")
                # Transform.apply_registered_pass(func, pass_name="convert-linalg-to-affine-loops")
                Transform.yield()
            end
        end
        return region
        IR.get_first_block(region)
    end
    builtin.module_(;
        bodyRegion=ns()) |> display
end

function op()
    mod_op = parse(IR.Module, """
    #map = affine_map<(d0, d1, d2) -> (d0, d1)>
    #map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
    #map2 = affine_map<(d0, d1, d2) -> (d0, d2)>

    !F = f32
    // !T = memref<?x?x!F>
    !T = memref<?x?x!F, strided<[?, 1]>>

    func.func @f(%arg0: !T, %arg1: !T, %arg2: !T) attributes {llvm.emit_c_interface} {
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg1, %arg2 : !T, !T) outs(%arg0 : !T) {
    ^bb0(%in: !F, %in_0: !F, %out: !F):
        %1 = arith.mulf %in, %in_0 : !F
        %2 = arith.addf %out, %1 : !F
        linalg.yield %2 : !F
    }
    return
    }
    """) |> IR.get_operation
    first(IR.OperationIterator(IR.get_first_block(first(IR.RegionIterator(mod_op))))) |> API.mlirOperationClone |> IR.Operation
end

begin
    bodyRegion = ns()
    push!(IR.get_first_block(bodyRegion), op())
    mod = builtin.module_(;
        additional_attributes=[NamedAttribute("transform.with_named_sequence", IR.Attribute(API.mlirUnitAttrGet(IR.context())))],
        bodyRegion
    ) |> MModule
end

begin
    mlir_opt(mod, "transform-interpreter")
    mlir_opt(mod, "test-transform-dialect-erase-schedule")
    mlir_opt(mod, "one-shot-bufferize{bufferize-function-boundaries=true}")
    mlir_opt(mod, "fold-memref-subview-ops")
    # mlir_opt(mod, "convert-linalg-to-affine-loops")
    # mlir_opt(mod, "func.func(affine-loop-coalescing)")
    # mlir_opt(mod, "func.func(affine-loop-invariant-code-motion)")
    # mlir_opt(mod, "func.func(affine-loop-unroll{unroll-full})")
    # mlir_opt(mod, "builtin.module(func.func(affine-loop-normalize))")
    # mlir_opt(mod, "lower-affine")
    # mlir_opt(mod, "mem2reg")
    display(mod)
    println("\n---------------------------\n")
    display(Brutus.simplify(mod))
end


lowerModuleToLLVM(mod)

addr = jit(mod)("_mlir_ciface_f")

a = rand(Float32, 1024, 1024)
b = rand(Float32, 1024, 1024)
c = zeros(Float32, 1024, 1024)
@ccall $addr(MemRef(c)::Ref{MemRef}, MemRef(c)::Ref{MemRef}, MemRef(a)::Ref{MemRef}, MemRef(b)::Ref{MemRef})::Nothing

using Chairmarks

@b ccall(addr, Nothing, (Ref{MemRef}, Ref{MemRef}, Ref{MemRef}, Ref{MemRef}), MemRef(c), MemRef(c), MemRef(a), MemRef(b))

c ≈ a*b

using LoopVectorization, Chairmarks


function f(c, a, b)
    I, J, K = size(a)..., size(b, 2)

    @turbo for i in 1:I
        for j in 1:J
            temp = zero(eltype(c))
            for k in 1:K
                temp += a[i, k] * b[k, j]
            end
            c[i, j] = temp
        end
    end
end

function f2(c, a, b)
    I, J, K = size(a)..., size(b, 2)

    for i in 1:I
        for j in 1:J
            temp = zero(eltype(c))
            for k in 1:K
                @inbounds temp += a[i, k] * b[k, j]
            end
            @inbounds c[i, j] = temp
        end
    end
end

function f3(c, a, b)
    I, J, K = size(a)..., size(b, 2)

    for j in 1:J
        for k in 1:K
            for i in 1:I
                @inbounds c[i, j] += a[i, k] * b[k, j]
            end
        end
    end
end

function f4(c, a, b)
    N = 4

    for j in 1:1024
        for k in 1:1024
            for i in 1:4:1024
                Base.@nexprs 4 l -> (@inbounds c[i+l-1, j] += a[i+l-1, k] * b[k, j])
                # @inbounds c[i, j] += a[i, k] * b[k, j]
            end
        end
    end
end




@b f(c, a, b) seconds=3
@b f2(c, a, b)
@b f3(c, a, b) seconds=3 # 90ms
@b f4(c, a, b) seconds=3
BLAS.set_num_threads(1)
@b mul!(c, a, b) seconds=3
