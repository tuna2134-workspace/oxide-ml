//! oxide-ml public facade.
//!
//! This milestone implements the CPU execution path from the design:
//! typed tensors, symbolic-shape SSA IR, a reference executor, reverse-mode
//! autograd, core layers and optimizers. CUDA lowering is intentionally kept
//! behind the `KernelBackend` ABI for a later phase.

pub use oxide_ml_autograd::{
    FunctionRule, GradientRegistry, GradientRule, TrainingGraph, build_training_graph,
    build_training_graph_with_registry,
};
pub use oxide_ml_compiler::{
    CommonSubexpressionElimination, CompileOptions, CompileReport, CompiledGraph, Compiler,
    ConstantFolding, DeadCodeElimination, GraphPass, MemoryPlanner, OperatorFusion, PassManager,
    PassReport, PassToggles,
};
pub use oxide_ml_core::{
    DType, DenseTensor, Device, Dim, Error, MemoryPlan, NodeId, Parameter, ParameterId,
    ParameterStore, Result, Rng, Scalar, Shape, SymbolId, Tensor, TensorValue, ValueId,
};
pub use oxide_ml_cpu::{CpuBackend, ExecutionInputs, ExecutionResult, Executor};
pub use oxide_ml_graph::{
    ActivationKind, ConstantData, ConstantTensor, Conv2dConfig, GRAPH_FORMAT_VERSION, Graph,
    GraphBuilder, GraphVerifier, IR_VERSION, Op, OpAttrs, Pool2dConfig, Reduction,
    SERIALIZATION_FORMAT_VERSION, ValueKind, infer_outputs,
};
pub use oxide_ml_kernel::{
    BACKEND_CAPABILITY_VERSION, BackendCapabilities, KERNEL_ABI_VERSION, KernelBackend, KernelCall,
    TensorSpec,
};
pub use oxide_ml_nn::{LayerNorm, Linear, Mlp, Module, init_parameters, parameter_ids};
pub use oxide_ml_optim::{AdamW, Gradients, Optimizer, RmsProp, Sgd, SgdMomentum};

use std::collections::HashMap;

/// Collect the gradients produced by one training-graph execution into the
/// optimizer-facing `ParameterId -> Tensor` map.
pub fn collect_gradients<T: Scalar>(
    training: &TrainingGraph,
    result: &ExecutionResult<T>,
) -> Result<Gradients<T>> {
    let mut gradients = HashMap::new();
    for (parameter, gradient) in training.parameter_gradients() {
        gradients.insert(parameter, result.value(gradient)?.clone());
    }
    Ok(gradients)
}

/// Minimal training-loop owner for examples and tests.
///
/// Larger applications can call `Executor` and `Optimizer` directly; this type
/// exists so the ordinary training loop stays short.
pub struct Trainer<T: Scalar = f32> {
    pub training: TrainingGraph,
    pub executor: Executor<T>,
    pub parameters: ParameterStore<T>,
    pub optimizer: Box<dyn Optimizer<T>>,
}

impl<T: Scalar> Trainer<T> {
    pub fn new(
        training: TrainingGraph,
        parameters: ParameterStore<T>,
        optimizer: Box<dyn Optimizer<T>>,
    ) -> Result<Self> {
        let executor = Executor::new(training.graph.clone())?;
        Ok(Trainer {
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
        let gradients = collect_gradients(&self.training, &result)?;
        self.optimizer.step(&mut self.parameters, &gradients)?;
        Ok(loss)
    }

    /// Forward + backward + optimizer update using a compiled memory plan.
    pub fn step_planned(&mut self, inputs: &ExecutionInputs<T>, plan: &MemoryPlan) -> Result<f32> {
        let (result, _) = self.executor.run_planned(&self.parameters, inputs, plan)?;
        let loss = self.loss(&result)?;
        let gradients = collect_gradients(&self.training, &result)?;
        self.optimizer.step(&mut self.parameters, &gradients)?;
        Ok(loss)
    }

    pub fn inference_executor(&self) -> Result<Executor<T>> {
        // The forward prefix is a valid graph on its own; rebuild a graph that
        // excludes the appended backward nodes.
        let mut forward = self.training.graph.clone();
        forward.nodes.truncate(self.training.forward_node_count);
        forward.outputs = self.training.forward_outputs.clone();
        forward.validate()?;
        Executor::new(forward)
    }
}

/// Re-export the most common imports.
pub mod prelude {
    pub use crate::{
        ActivationKind, AdamW, CommonSubexpressionElimination, CompileOptions, CompileReport,
        Compiler, ConstantFolding, ConstantTensor, Conv2dConfig, DType, DeadCodeElimination,
        DenseTensor, Device, Dim, Error, ExecutionInputs, Executor, GradientRegistry, Gradients,
        Graph, GraphBuilder, GraphVerifier, IR_VERSION, LayerNorm, Linear, MemoryPlan,
        MemoryPlanner, Mlp, Module, NodeId, Op, OpAttrs, Optimizer, ParameterId, ParameterStore,
        PassManager, PassToggles, Pool2dConfig, Reduction, Result, Rng,
        SERIALIZATION_FORMAT_VERSION, Scalar, Sgd, Shape, Tensor, Trainer, TrainingGraph, ValueId,
        ValueKind, build_training_graph, build_training_graph_with_registry, collect_gradients,
        init_parameters,
    };
}
