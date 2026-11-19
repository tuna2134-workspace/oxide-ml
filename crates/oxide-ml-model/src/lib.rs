//! Model definition, parameter discovery and execution for oxide-ml.
//!
//! This crate is the bridge between user-defined modules and the graph stack:
//!
//! - [`Module`] describes structure and owned parameters without a graph.
//! - [`ParameterRegistry`] discovers stable, name-based parameter specs.
//! - [`CompiledModel`] builds, optimizes and executes a graph for a device.
//! - [`Trainer`] owns the training loop, dataset iteration and checkpoints.
//!
//! Device-specific execution lives in backend crates. The CPU workspace uses
//! [`CompiledModel`] directly; `oxide-ml-cuda` builds a CUDA model from the same
//! graph and parameter names.

#![allow(clippy::type_complexity)]

mod checkpoint;
mod context;
mod dataset;
mod loss;
mod module;
pub mod ops;
mod parameter;
mod runtime;
mod tensor;
mod trainer;

pub use checkpoint::{
    CHECKPOINT_FORMAT, CHECKPOINT_VERSION, Checkpoint, CheckpointParameter, LoadReport, load_model,
    save_model,
};
pub use context::{ModelContext, module_forward, module_forward_multi, single_output};
pub use dataset::{Collated, DataLoader, Dataset, Sample, TensorDataset, collate};
pub use loss::{Loss, Reduction};
pub use module::{
    Module, ModuleDescriptor, ModuleMode, ModulePath, Sequential, for_each_parameter,
};
pub use parameter::{ParameterRegistry, ParameterRole, ParameterSpec};
pub use runtime::{
    BuiltModel, CompiledModel, InputSpec, ModelCompileOptions, ModelReport, TrainingOutput,
    WriteBackReport, collect_gradients, parameter_store_named, predict_many, predict_single,
    write_back_named,
};
pub use tensor::{Tensor, stack};
pub use trainer::{
    EpochReport, LrScheduler, StepLr, TrainReport, Trainer, TrainerBuilder, detached_parameters,
    gradients_from_named, merge_gradients,
};

pub use oxide_ml_autograd::TrainingGraph;
pub use oxide_ml_core::{DType, Device, Error, ParameterId, ParameterStore, Result, ValueId};
pub use oxide_ml_optim::{Gradients, Optimizer, OptimizerState};

/// Graph value handle used by generated forward functions.
pub type Value = ValueId;

/// Re-exports used by `#[derive(Module)]`, `#[model]` and `graph!`.
///
/// This module is not part of the stable public API.
#[doc(hidden)]
pub mod __private {
    pub use oxide_ml_core::{DType, Error, Result, ValueId};

    pub use crate::context::{ModelContext, module_forward, module_forward_multi, single_output};
    pub use crate::module::{Module, ModuleDescriptor, ModuleMode, ModulePath};
    pub use crate::ops;
    pub use crate::parameter::{ParameterRegistry, ParameterRole, ParameterSpec};
    pub use crate::runtime::{predict_many, predict_single};
    pub use crate::tensor::Tensor;
    pub use oxide_ml_core::{DenseTensor, Shape};
    pub use oxide_ml_graph::{Graph, GraphBuilder, Op, OpAttrs, Reduction};
}
