//! Compile oxide-ml modules into device-resident CUDA graphs.
//!
//! The graph, parameter names and shapes come from `oxide_ml_model`, so the
//! same model definition runs on CPU and CUDA. Parameters are uploaded once
//! into a [`CudaParameterStore`]; forward execution never returns intermediate
//! values to the host.

use std::collections::HashMap;

use oxide_ml_autograd::{TrainingGraph, build_training_graph};
use oxide_ml_core::{DenseTensor, ParameterId, ParameterStore, Result as MlResult, ValueId};
use oxide_ml_graph::Graph;
use oxide_ml_model::{
    BuiltModel, InputSpec, Loss, ModelCompileOptions, Module, ParameterSpec, WriteBackReport,
    parameter_store_named, write_back_named,
};

use crate::device::CudaDevice;
use crate::error::{CudaError, Result};
use crate::executor::CudaExecutor;
use crate::optimizer::CudaOptimizer;
use crate::param::CudaParameterStore;
use crate::trainer::CudaTrainer;

fn parameter_ids(graph: &Graph) -> HashMap<String, ParameterId> {
    graph
        .parameters()
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.id))
        .collect()
}

fn aliases(built: &BuiltModel) -> HashMap<String, String> {
    built
        .registry
        .aliases()
        .iter()
        .map(|(alias, canonical)| (alias.clone(), canonical.clone()))
        .collect()
}

fn host_parameter_store<M: Module + ?Sized>(
    model: &M,
    specs: &[ParameterSpec],
    ids: &HashMap<String, ParameterId>,
    aliases: &HashMap<String, String>,
) -> MlResult<ParameterStore<f32>> {
    parameter_store_named(model, specs, ids, aliases)
}

/// A forward-only model whose values stay on the device between nodes.
pub struct CudaModel {
    executor: CudaExecutor,
    graph: Graph,
    parameters: CudaParameterStore,
    parameter_ids: HashMap<String, ParameterId>,
    aliases: HashMap<String, String>,
    parameter_specs: Vec<ParameterSpec>,
    input_names: Vec<String>,
    input_values: Vec<ValueId>,
    output_values: Vec<ValueId>,
}

impl CudaModel {
    /// Compile `model` for `device`.
    pub fn compile<M: Module + ?Sized>(
        device: CudaDevice,
        model: &M,
        inputs: &[InputSpec],
        options: ModelCompileOptions,
    ) -> Result<Self> {
        let built = BuiltModel::build(model, inputs, None, options.mode)?;
        let (built, _, _) = built.optimize(&options)?;
        let parameter_ids = parameter_ids(&built.graph);
        let aliases = aliases(&built);
        let host = host_parameter_store(model, built.registry.specs(), &parameter_ids, &aliases)?;
        let executor = CudaExecutor::new(device)?;
        let parameters = CudaParameterStore::from_host(&built.graph, &host, executor.pool())?;
        Ok(CudaModel {
            executor,
            input_names: built
                .input_specs
                .iter()
                .map(|spec| spec.name.clone())
                .collect(),
            input_values: built.inputs,
            output_values: built.outputs,
            graph: built.graph,
            parameters,
            parameter_ids,
            aliases,
            parameter_specs: built.registry.specs().to_vec(),
        })
    }

    pub fn executor(&self) -> &CudaExecutor {
        &self.executor
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn parameter_specs(&self) -> &[ParameterSpec] {
        &self.parameter_specs
    }

    /// Download the current parameters into a host store.
    pub fn host_parameters(&self) -> Result<ParameterStore<f32>> {
        let stream = self.executor.device().default_stream();
        let mut store = ParameterStore::new();
        for spec in &self.parameter_specs {
            let id = *self.parameter_ids.get(&spec.name).ok_or_else(|| {
                CudaError::Internal(format!("missing parameter id for {:?}", spec.name))
            })?;
            let data = self.parameters.download(id, &stream)?;
            store.add(id, spec.name.clone(), data, spec.role.is_trainable())?;
        }
        Ok(store)
    }

    /// Copy device parameters back into the module definition.
    pub fn sync_to_model<M: Module + ?Sized>(&self, model: &mut M) -> Result<WriteBackReport> {
        let store = self.host_parameters()?;
        Ok(write_back_named(
            model,
            &self.parameter_ids,
            &self.aliases,
            &store,
        )?)
    }

    /// Run forward inference. Inputs and outputs are host tensors; all
    /// intermediates stay on the device.
    pub fn run(&self, inputs: &[(&str, DenseTensor<f32>)]) -> Result<Vec<DenseTensor<f32>>> {
        let mut host = HashMap::new();
        for (name, tensor) in inputs {
            let index = self
                .input_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| CudaError::Shape(format!("unknown model input {name:?}")))?;
            host.insert(self.input_values[index], tensor.clone());
        }
        for (index, value) in self.input_values.iter().enumerate() {
            if !host.contains_key(value) {
                return Err(CudaError::Shape(format!(
                    "missing model input {:?}",
                    self.input_names[index]
                )));
            }
        }
        let run = self.executor.run(&self.graph, &self.parameters, &host)?;
        let stream = self.executor.device().default_stream();
        self.output_values
            .iter()
            .map(|value| run.download(&stream, *value))
            .collect()
    }
}

/// A training model with device-resident parameters and gradients.
pub struct CudaTrainingModel {
    pub trainer: CudaTrainer,
    pub parameter_ids: HashMap<String, ParameterId>,
    pub aliases: HashMap<String, String>,
    pub parameter_specs: Vec<ParameterSpec>,
    pub input_names: Vec<String>,
    pub input_values: Vec<ValueId>,
    pub target_name: String,
    pub target_value: ValueId,
}

impl CudaTrainingModel {
    /// Build a training graph from a module and move its parameters to `device`.
    pub fn new<M: Module + ?Sized>(
        device: CudaDevice,
        model: &M,
        inputs: &[InputSpec],
        target: InputSpec,
        loss: &dyn Loss,
        optimizer: Box<dyn CudaOptimizer>,
    ) -> Result<Self> {
        let options = ModelCompileOptions::training();
        let built = BuiltModel::build(model, inputs, Some((loss, target.clone())), options.mode)?;
        let (built, _, _) = built.optimize(&options)?;
        let parameter_ids = parameter_ids(&built.graph);
        let aliases = aliases(&built);
        let host = host_parameter_store(model, built.registry.specs(), &parameter_ids, &aliases)?;
        let loss_value = built
            .loss
            .ok_or_else(|| CudaError::Internal("training build produced no loss".into()))?;
        let training: TrainingGraph = build_training_graph(&built.graph, loss_value)?;
        let trainer = CudaTrainer::new(device, training, &host, optimizer)?;
        let target_value = built
            .target_value
            .ok_or_else(|| CudaError::Internal("training build produced no target input".into()))?;
        Ok(CudaTrainingModel {
            trainer,
            parameter_ids,
            aliases,
            parameter_specs: built.registry.specs().to_vec(),
            input_names: built
                .input_specs
                .iter()
                .map(|spec| spec.name.clone())
                .collect(),
            input_values: built.inputs,
            target_name: target.name,
            target_value,
        })
    }

    /// One training step. Returns the scalar loss.
    pub fn step(
        &mut self,
        inputs: &[(&str, DenseTensor<f32>)],
        targets: &DenseTensor<f32>,
    ) -> Result<f32> {
        let mut host = HashMap::new();
        for (name, tensor) in inputs {
            let index = self
                .input_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| CudaError::Shape(format!("unknown model input {name:?}")))?;
            host.insert(self.input_values[index], tensor.clone());
        }
        host.insert(self.target_value, targets.clone());
        self.trainer.step(&host)
    }

    /// Download parameters into a host store.
    pub fn host_parameters(&self) -> Result<ParameterStore<f32>> {
        let stream = self.trainer.executor.device().default_stream();
        let mut store = ParameterStore::new();
        for spec in &self.parameter_specs {
            let id = *self.parameter_ids.get(&spec.name).ok_or_else(|| {
                CudaError::Internal(format!("missing parameter id for {:?}", spec.name))
            })?;
            let data = self.trainer.parameters.download(id, &stream)?;
            store.add(id, spec.name.clone(), data, spec.role.is_trainable())?;
        }
        Ok(store)
    }

    /// Copy trained device parameters back into the module definition.
    pub fn sync_to_model<M: Module + ?Sized>(&self, model: &mut M) -> Result<WriteBackReport> {
        let store = self.host_parameters()?;
        Ok(write_back_named(
            model,
            &self.parameter_ids,
            &self.aliases,
            &store,
        )?)
    }
}
