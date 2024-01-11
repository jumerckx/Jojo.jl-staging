module Types

export i1, i8, i16, i32, i64, f32, f64, index

using MLIR
using MLIR.IR: AbstractValue, Value, MLIRType, get_value
using MLIR.API: MlirValue

struct MLIRInteger{N} <: AbstractValue
    value::MlirValue
    MLIRInteger{N}(i::V) where {N, V<:AbstractValue} = new(get_value(i))
end
MLIR.IR.MLIRType(::Type{MLIRInteger{N}}) where {N} = MLIRType(MLIR.API.mlirIntegerTypeGet(MLIR.IR.context(), N))

const i1 = MLIRInteger{1}
const i8 = MLIRInteger{8}
const i16 = MLIRInteger{16}
const i32 = MLIRInteger{32}
const i64 = MLIRInteger{64}

struct MLIRF64 <: AbstractValue
    value::MlirValue
end
const f64 = MLIRF64

struct MLIRF32 <: AbstractValue
    value::MlirValue
end
const f32 = MLIRF32

struct MLIRIndex <: AbstractValue
    value::MlirValue
end
const index = MLIRIndex

end # Types
