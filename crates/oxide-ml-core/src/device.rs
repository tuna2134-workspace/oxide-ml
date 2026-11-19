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

    /// Identifier for a CUDA device.
    ///
    /// This only constructs the logical device; availability is checked by the
    /// backend that opens it (`oxide-ml-cuda`).
    pub fn cuda(index: usize) -> crate::Result<Device> {
        Ok(Device::Cuda(index as u32))
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
