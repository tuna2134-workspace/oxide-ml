//! Reference CPU backend: correctness-first kernels and graph executor.

mod executor;
mod kernels;

pub use executor::{ExecutionInputs, ExecutionResult, ExecutionStats, Executor};
pub use kernels::CpuBackend;
