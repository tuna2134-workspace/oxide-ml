use serde::{Deserialize, Serialize};

/// Logical execution device attached to graph values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(u32),
}

impl Device {
    pub const fn is_cpu(self) -> bool {
        matches!(self, Device::Cpu)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => f.write_str("cpu"),
            Device::Cuda(index) => write!(f, "cuda:{index}"),
        }
    }
}
