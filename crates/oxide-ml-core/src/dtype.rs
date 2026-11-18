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
}

impl DType {
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::F64 => 8,
        }
    }

    pub const fn is_float(self) -> bool {
        true
    }

    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F64 => "f64",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
