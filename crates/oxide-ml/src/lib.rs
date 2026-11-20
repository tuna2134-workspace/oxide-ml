//! oxide-ml: a graph-based machine-learning framework with an integrated model
//! API, proc macros, autograd, a graph compiler and CPU/CUDA backends.
//!
//! The recommended entry point is [`prelude`]:
//!
//! ```ignore
//! use oxide_ml::prelude::*;
//!
//! #[derive(Module)]
//! struct Mlp {
//!     fc1: Linear,
//!     fc2: Linear,
//! }
//!
//! impl Mlp {
//!     #[model]
//!     fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
//!         let x = relu(self.fc1.forward(x));
//!         self.fc2.forward(x)
//!     }
//! }
//!
//! let mut model = Mlp { fc1: Linear::new(2, 8), fc2: Linear::new(8, 1) };
//! let mut trainer = Trainer::builder()
//!     .optimizer(AdamW::new(1e-2)?)
//!     .loss(MSELoss::new())
//!     .build()?;
//! trainer.fit(&mut model, &dataset, 100)?;
//! ```
//!
//! The low-level stack (`GraphBuilder`, `Executor`, `TrainingGraph`, ...) stays
//! available at the crate root. The original graph-builder layers live in
//! [`legacy`].

pub use oxide_ml_autograd::{
    FunctionRule, GradientRegistry, GradientRule, TrainingGraph, build_training_graph,
    build_training_graph_with_registry,
};
pub use oxide_ml_compiler::{
    CommonSubexpressionElimination, CompileOptions, CompileReport, CompiledGraph, Compiler,
    ConstantFolding, DeadCodeElimination, GraphPass, KernelPlan, KernelPlanEntry, MemoryPlanner,
    OperatorFusion, PassManager, PassReport, PassToggles,
};
pub use oxide_ml_core::{
    DType, DenseTensor, Device, Dim, Error, MemoryPlan, NodeId, Parameter, ParameterId,
    ParameterStore, Result, Rng, Scalar, Shape, SymbolId, Tensor as RankedTensor, ValueId,
};
pub use oxide_ml_cpu::{CpuBackend, ExecutionInputs, ExecutionResult, Executor};
pub use oxide_ml_graph::{
    ActivationKind, ConstantData, ConstantTensor, Conv2dConfig, GRAPH_FORMAT_VERSION, Graph,
    GraphBuilder, GraphVerifier, IR_VERSION, Node, Op, OpAttrs, Pool2dConfig, Reduction,
    SERIALIZATION_FORMAT_VERSION, ValueKind, infer_outputs,
};
pub use oxide_ml_kernel::{
    AutogradSupport, BACKEND_CAPABILITY_VERSION, BackendCapabilities, ConstraintKind,
    CustomBackward, CustomCall, CustomKernel, CustomKernelRegistry, Dimension, KERNEL_ABI_VERSION,
    KERNEL_DESCRIPTOR_VERSION, KernelBackend, KernelCall, KernelCapabilities, KernelCatalog,
    KernelDescriptor, KernelOperator, KernelProvider, KernelRequest, KernelSelection, LaunchConfig,
    ProviderKind, SideEffect, SpecializationKey, TensorConstraint, TensorSpec,
};
pub use oxide_ml_optim::{AdamW, Gradients, Optimizer, RmsProp, Sgd, SgdMomentum};

pub use oxide_ml_macros::{Module, graph, model};
pub use oxide_ml_model::{
    BuiltModel, CHECKPOINT_FORMAT, CHECKPOINT_VERSION, COMPILE_CACHE_VERSION, CacheStats,
    Checkpoint, CheckpointParameter, CompileCache, CompileKey, CompiledModel, DataLoader, Dataset,
    EarlyStopping, EarlyStoppingConfig, EpochReport, InputSpec, KvCache, KvCacheConfig,
    KvLayerConfig, LoadReport, Loss, LrScheduler, ModelCompileOptions, ModelContext, ModelReport,
    ModelState, Module, ModuleDescriptor, ModuleList, ModuleMode, ModulePath, ParameterRegistry,
    ParameterRole, ParameterSpec, Runtime, RuntimeBuilder, Sample, Sequential, StepLr, Tensor,
    TensorDataset, TrainReport, Trainer, TrainerBuilder, TrainingOutput, WriteBackReport,
    classification_accuracy, clip_gradients, collate, collect_gradients, compile_cached,
    compile_cached_with, global_compile_cache, load_model, module_forward, module_forward_multi,
    predict_many, predict_many_with_cache, predict_single, save_model, single_output, stack,
};
pub use oxide_ml_nn::{
    Attention, AvgPool2d, BCEWithLogitsLoss, BatchNorm, Conv1d, Conv2d, Conv3d, CrossEntropyLoss,
    Dropout, Embedding, GELU, GatedActivation, GatedMlp, GqaConfig, GroupNorm, Gru, HuberLoss,
    InstanceNorm, LayerNorm, Linear, Lstm, MSELoss, MaxPool1d, MaxPool2d, MultiHeadAttention,
    NLLLoss, ReLU, RmsNorm, RmsNormNoScale, Rnn, RopeConfig, Sigmoid, Softmax, Tanh, TiedLinear,
    TiedWeightSource, apply_rope, apply_rope_at, causal_mask, grouped_query_attention, rms_norm,
    rotate_half, rotate_half_value,
};

use std::collections::HashMap;

/// Collect gradients with the low-level `TrainingGraph` API.
pub fn collect_graph_gradients<T: Scalar>(
    training: &TrainingGraph,
    result: &ExecutionResult<T>,
) -> Result<HashMap<ParameterId, DenseTensor<T>>> {
    let mut gradients = HashMap::new();
    for (parameter, gradient) in training.parameter_gradients() {
        gradients.insert(parameter, result.value(gradient)?.clone());
    }
    Ok(gradients)
}

/// Minimal low-level training-loop owner for the graph API.
///
/// This is the original `Trainer`. New code should use
/// [`oxide_ml_model::Trainer`], which works with modules and datasets.
pub struct GraphTrainer<T: Scalar = f32> {
    pub training: TrainingGraph,
    pub executor: Executor<T>,
    pub parameters: ParameterStore<T>,
    pub optimizer: Box<dyn Optimizer<T>>,
}

impl<T: Scalar> GraphTrainer<T> {
    pub fn new(
        training: TrainingGraph,
        parameters: ParameterStore<T>,
        optimizer: Box<dyn Optimizer<T>>,
    ) -> Result<Self> {
        let executor = Executor::new(training.graph.clone())?;
        Ok(GraphTrainer {
            training,
            executor,
            parameters,
            optimizer,
        })
    }

    pub fn run(&self, inputs: &ExecutionInputs<T>) -> Result<ExecutionResult<T>> {
        self.executor.run(&self.parameters, inputs)
    }

    pub fn run_planned(
        &self,
        inputs: &ExecutionInputs<T>,
        plan: &MemoryPlan,
    ) -> Result<(ExecutionResult<T>, oxide_ml_cpu::ExecutionStats)> {
        self.executor.run_planned(&self.parameters, inputs, plan)
    }

    pub fn loss(&self, result: &ExecutionResult<T>) -> Result<f32> {
        let tensor = result.value(self.training.loss)?;
        if tensor.numel() != 1 {
            return Err(Error::execution("loss tensor must be scalar"));
        }
        Ok(tensor.data()[0].to_f64() as f32)
    }

    /// Forward + backward + optimizer update.
    pub fn step(&mut self, inputs: &ExecutionInputs<T>) -> Result<f32> {
        let result = self.executor.run(&self.parameters, inputs)?;
        let loss = self.loss(&result)?;
        let gradients = collect_graph_gradients(&self.training, &result)?;
        self.optimizer.step(&mut self.parameters, &gradients)?;
        Ok(loss)
    }

    pub fn step_planned(&mut self, inputs: &ExecutionInputs<T>, plan: &MemoryPlan) -> Result<f32> {
        let (result, _) = self.executor.run_planned(&self.parameters, inputs, plan)?;
        let loss = self.loss(&result)?;
        let gradients = collect_graph_gradients(&self.training, &result)?;
        self.optimizer.step(&mut self.parameters, &gradients)?;
        Ok(loss)
    }

    pub fn inference_executor(&self) -> Result<Executor<T>> {
        let mut forward = self.training.graph.clone();
        forward.nodes.truncate(self.training.forward_node_count);
        forward.outputs = self.training.forward_outputs.clone();
        forward.validate()?;
        Executor::new(forward)
    }
}

/// Compatibility module for the original graph-builder layers and trainer.
pub mod legacy {
    pub use crate::{GraphTrainer, collect_graph_gradients};
    pub use oxide_ml_nn::legacy::{LayerNorm, Linear, Mlp, Module, init_parameters, parameter_ids};

    /// Original low-level trainer name.
    pub use crate::GraphTrainer as Trainer;
}

/// Re-exports used by the proc macros. Not part of the stable API.
#[doc(hidden)]
pub mod __private {
    pub use oxide_ml_model::__private::*;
    pub use oxide_ml_model::ParameterRole;
}

/// Re-export the most common imports for the model API.
pub mod prelude {
    pub use crate::ReLU;
    pub use crate::legacy::init_parameters;
    pub use crate::{
        ActivationKind, AdamW, Attention, AutogradSupport, AvgPool2d, BCEWithLogitsLoss, BatchNorm,
        CacheStats, Checkpoint, CommonSubexpressionElimination, CompileCache, CompileOptions,
        CompileReport, CompiledModel, Compiler, ConstantFolding, ConstantTensor, ConstraintKind,
        Conv1d, Conv2d, Conv2dConfig, Conv3d, CpuBackend, CrossEntropyLoss, CustomBackward,
        CustomCall, CustomKernel, CustomKernelRegistry, DType, DataLoader, Dataset,
        DeadCodeElimination, DenseTensor, Device, Dim, Dimension, Dropout, EarlyStopping,
        EarlyStoppingConfig, Embedding, EpochReport, Error, ExecutionInputs, Executor, GELU,
        GatedActivation, GatedMlp, GqaConfig, GradientRegistry, Gradients, Graph, GraphBuilder,
        GraphVerifier, GroupNorm, Gru, HuberLoss, IR_VERSION, InputSpec, InstanceNorm,
        KernelCatalog, KernelDescriptor, KernelOperator, KernelPlan, KernelPlanEntry, KvCache,
        KvCacheConfig, KvLayerConfig, LaunchConfig, LayerNorm, Linear, Loss, Lstm, MSELoss,
        MaxPool1d, MaxPool2d, MemoryPlan, MemoryPlanner, ModelCompileOptions, ModelState, Module,
        ModuleList, ModulePath, MultiHeadAttention, NLLLoss, Node, NodeId, Op, OpAttrs, Optimizer,
        Parameter, ParameterId, ParameterRegistry, ParameterRole, ParameterStore, PassManager,
        PassToggles, Pool2dConfig, Reduction, Result, RmsNorm, RmsNormNoScale, Rng, Rnn,
        RopeConfig, Runtime, RuntimeBuilder, SERIALIZATION_FORMAT_VERSION, Sample, Scalar,
        Sequential, Sgd, Shape, SideEffect, Sigmoid, Softmax, SpecializationKey, StepLr, SymbolId,
        Tanh, Tensor, TensorConstraint, TensorDataset, TiedLinear, TiedWeightSource, TrainReport,
        Trainer, TrainingGraph, ValueId, ValueKind, build_training_graph,
        build_training_graph_with_registry, classification_accuracy, clip_gradients,
        collect_gradients, collect_graph_gradients, compile_cached, compile_cached_with,
        global_compile_cache, graph, load_model, model, predict_many_with_cache, save_model,
    };

    /// Elementwise operations usable directly inside `#[model]` bodies.
    pub use oxide_ml_model::ops;
}
