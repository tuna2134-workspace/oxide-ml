use serde::{Deserialize, Serialize};

/// Element type of a tensor.
///
/// Only `F32` is executable by the reference CPU backend in this milestone.
/// The other variants are part of the IR vocabulary so that graph
/// serialization and future backends do not need a breaking change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    Bool,
}

impl DType {
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 => 8,
            DType::I8 | DType::U8 | DType::Bool => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }

    pub const fn is_integer(self) -> bool {
        matches!(
            self,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::U8
        )
    }

    pub const fn is_bool(self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Whether the reference CPU backend can execute this dtype.
    pub const fn is_reference_supported(self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }

    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F64 => "f64",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::Bool => "bool",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
