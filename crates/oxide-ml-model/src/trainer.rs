//! Training loop, dataset iteration and evaluation.

use oxide_ml_core::{DenseTensor, Device, Error, ParameterId, ParameterStore, Result};
use oxide_ml_kernel::CustomKernelRegistry;
use oxide_ml_optim::Optimizer;
use std::sync::Arc;

use crate::dataset::{DataLoader, Dataset, collate};
use crate::loss::Loss;
use crate::module::Module;
use crate::runtime::{CompiledModel, InputSpec, ModelCompileOptions, WriteBackReport};
use crate::tensor::Tensor;

/// Learning-rate schedule evaluated once per epoch.
pub trait LrScheduler: Send + Sync {
    fn learning_rate(&self, base: f32, epoch: usize) -> f32;
}

/// Multiply the learning rate by `gamma` every `step_size` epochs.
#[derive(Clone, Copy, Debug)]
pub struct StepLr {
    pub step_size: usize,
    pub gamma: f32,
}

impl StepLr {
    pub fn new(step_size: usize, gamma: f32) -> Result<Self> {
        if step_size == 0 {
            return Err(Error::invalid_shape("StepLr", "step_size must be > 0"));
        }
        if !(gamma.is_finite() && gamma > 0.0) {
            return Err(Error::invalid_shape("StepLr", "gamma must be positive"));
        }
        Ok(StepLr { step_size, gamma })
    }
}

impl LrScheduler for StepLr {
    fn learning_rate(&self, base: f32, epoch: usize) -> f32 {
        let decays = (epoch / self.step_size) as i32;
        base * self.gamma.powi(decays)
    }
}

/// One epoch of training.
#[derive(Clone, Debug, PartialEq)]
pub struct EpochReport {
    pub epoch: usize,
    pub loss: f32,
    pub steps: usize,
    pub samples: usize,
}

/// Result of [`Trainer::fit`].
#[derive(Clone, Debug, PartialEq)]
pub struct TrainReport {
    pub epochs: Vec<EpochReport>,
    pub steps: usize,
    pub samples: usize,
    pub final_loss: f32,
    pub parameters: usize,
    pub trainable_parameters: usize,
    pub write_back: WriteBackReport,
}

impl TrainReport {
    pub fn loss(&self) -> f32 {
        self.final_loss
    }
}

/// Builder-style training configuration.
pub struct TrainerBuilder {
    optimizer: Option<Box<dyn Optimizer<f32>>>,
    loss: Option<Box<dyn Loss>>,
    device: Device,
    batch_size: usize,
    shuffle: Option<u64>,
    drop_last: bool,
    sync_model: bool,
    scheduler: Option<Box<dyn LrScheduler>>,
    custom_kernels: Option<Arc<CustomKernelRegistry<f32>>>,
    accumulation: usize,
}

impl Default for TrainerBuilder {
    fn default() -> Self {
        TrainerBuilder {
            optimizer: None,
            loss: None,
            device: Device::Cpu,
            batch_size: 32,
            shuffle: Some(0),
            drop_last: false,
            sync_model: true,
            scheduler: None,
            custom_kernels: None,
            accumulation: 1,
        }
    }
}

impl std::fmt::Debug for Trainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trainer")
            .field("device", &self.device)
            .field("batch_size", &self.batch_size)
            .field("has_optimizer", &self.optimizer.is_some())
            .field("has_loss", &self.loss.is_some())
            .finish()
    }
}

/// Device-independent training loop.
///
/// The CPU workspace executes on `Device::Cpu`. A CUDA device is rejected with
/// an explicit error; `oxide-ml-cuda` provides `CudaTrainer` over the same
/// `CompiledModel` graph and parameter names.
pub struct Trainer {
    optimizer: Option<Box<dyn Optimizer<f32>>>,
    loss: Option<Box<dyn Loss>>,
    device: Device,
    batch_size: usize,
    shuffle: Option<u64>,
    drop_last: bool,
    sync_model: bool,
    scheduler: Option<Box<dyn LrScheduler>>,
    custom_kernels: Option<Arc<CustomKernelRegistry<f32>>>,
    accumulation: usize,
}

impl Default for Trainer {
    fn default() -> Self {
        Self::new()
    }
}

impl Trainer {
    /// Start an unconfigured trainer. `fit` fails until optimizer and loss are
    /// supplied; use [`Trainer::builder`] for validated construction.
    pub fn new() -> Self {
        TrainerBuilder::default().into_trainer()
    }

    pub fn builder() -> TrainerBuilder {
        TrainerBuilder::default()
    }

    pub fn optimizer<O: Optimizer<f32> + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Some(Box::new(optimizer));
        self
    }

    pub fn loss<L: Loss + 'static>(mut self, loss: L) -> Self {
        self.loss = Some(Box::new(loss));
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    pub fn shuffle(mut self, seed: Option<u64>) -> Self {
        self.shuffle = seed;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Copy updated parameters back into the module after every step. Disable
    /// only if the caller keeps the returned parameter store authoritative.
    pub fn sync_model(mut self, sync: bool) -> Self {
        self.sync_model = sync;
        self
    }

    pub fn scheduler<S: LrScheduler + 'static>(mut self, scheduler: S) -> Self {
        self.scheduler = Some(Box::new(scheduler));
        self
    }

    /// Register user-defined kernels used by the model graph.
    pub fn custom_kernels(mut self, kernels: Arc<CustomKernelRegistry<f32>>) -> Self {
        self.custom_kernels = Some(kernels);
        self
    }

    pub fn optimizer_ref(&self) -> Option<&dyn Optimizer<f32>> {
        self.optimizer.as_deref()
    }

    pub fn optimizer_mut(&mut self) -> Option<&mut Box<dyn Optimizer<f32>>> {
        self.optimizer.as_mut()
    }

    pub fn with_parts(mut self, optimizer: Box<dyn Optimizer<f32>>, loss: Box<dyn Loss>) -> Self {
        self.optimizer = Some(optimizer);
        self.loss = Some(loss);
        self
    }

    fn ensure_cpu(&self) -> Result<()> {
        match self.device {
            Device::Cpu => Ok(()),
            Device::Cuda(index) => Err(Error::unsupported(format!(
                "Trainer runs on the CPU workspace; cuda:{index} requires \
                 oxide-ml-cuda's CudaTrainer with the same model and dataset"
            ))),
        }
    }

    /// Train `model` on `dataset` for `epochs` epochs.
    pub fn fit<M: Module + ?Sized>(
        &mut self,
        model: &mut M,
        dataset: &dyn Dataset,
        epochs: usize,
    ) -> Result<TrainReport> {
        self.ensure_cpu()?;
        if dataset.is_empty() {
            return Err(Error::execution("cannot train on an empty dataset"));
        }
        let optimizer = self
            .optimizer
            .as_mut()
            .ok_or_else(|| Error::invalid_graph("Trainer requires an optimizer"))?;
        let loss = self
            .loss
            .as_deref()
            .ok_or_else(|| Error::invalid_graph("Trainer requires a loss function"))?;

        let input_specs: Vec<InputSpec> = dataset
            .input_specs()
            .into_iter()
            .map(|spec| InputSpec::batched(spec.name, spec.dims))
            .collect();
        let target_specs = dataset.target_specs();
        if target_specs.len() != 1 {
            return Err(Error::unsupported(format!(
                "Trainer supports exactly one target tensor, dataset provides {}",
                target_specs.len()
            )));
        }
        let target = InputSpec::batched(target_specs[0].name.clone(), target_specs[0].dims.clone());

        let mut options = ModelCompileOptions::training();
        options.custom_kernels = self.custom_kernels.clone();
        let compiled =
            CompiledModel::compile_training(model, &input_specs, target.clone(), loss, options)?;
        let mut store = compiled.parameter_store(model)?;
        let base_lr = optimizer.learning_rate();

        model.set_training(true);
        let mut loader = DataLoader::new(self.batch_size)?;
        if let Some(seed) = self.shuffle {
            loader = loader.shuffle(seed);
        }
        loader = loader.drop_last(self.drop_last);

        let accumulation = self.accumulation.max(1);
        let mut pending: oxide_ml_optim::Gradients<f32> = oxide_ml_optim::Gradients::new();
        let mut pending_batches = 0usize;
        let mut report = TrainReport {
            epochs: Vec::new(),
            steps: 0,
            samples: 0,
            final_loss: 0.0,
            parameters: compiled.report().parameters,
            trainable_parameters: compiled.report().trainable_parameters,
            write_back: WriteBackReport::default(),
        };

        for epoch in 0..epochs {
            if let Some(scheduler) = &self.scheduler {
                let lr = scheduler.learning_rate(base_lr, epoch);
                optimizer.set_learning_rate(lr)?;
            }
            let batches = loader.batches(dataset.len(), epoch as u64);
            let mut total_loss = 0.0f64;
            let mut samples = 0usize;
            let mut steps = 0usize;
            for batch in &batches {
                let mut batch_samples = Vec::with_capacity(batch.len());
                for index in batch {
                    batch_samples.push(dataset.get(*index)?);
                }
                let (input_tensors, target_tensors) = collate(&batch_samples)?;
                let mut owned: Vec<(String, Tensor<f32>)> = input_specs
                    .iter()
                    .zip(input_tensors)
                    .map(|(spec, tensor)| (spec.name.clone(), tensor))
                    .collect();
                if let Some(first) = target_tensors.into_iter().next() {
                    owned.push((target.name.clone(), first));
                }
                let borrowed: Vec<(&str, &Tensor<f32>)> = owned
                    .iter()
                    .map(|(name, tensor)| (name.as_str(), tensor))
                    .collect();
                let output = compiled.run_training(&store, &borrowed)?;
                accumulate_gradients(&mut pending, &output.gradients)?;
                pending_batches += 1;
                if pending_batches >= accumulation {
                    scale_gradients(&mut pending, 1.0 / pending_batches as f32);
                    optimizer.step(&mut store, &pending)?;
                    pending.clear();
                    pending_batches = 0;
                    if self.sync_model {
                        report.write_back = compiled.write_back(model, &store)?;
                    }
                }
                total_loss += output.loss as f64 * batch.len() as f64;
                samples += batch.len();
                steps += 1;
            }
            let epoch_loss = if samples == 0 {
                0.0
            } else {
                (total_loss / samples as f64) as f32
            };
            if pending_batches > 0 {
                scale_gradients(&mut pending, 1.0 / pending_batches as f32);
                optimizer.step(&mut store, &pending)?;
                pending.clear();
                pending_batches = 0;
                if self.sync_model {
                    report.write_back = compiled.write_back(model, &store)?;
                }
            }
            report.epochs.push(EpochReport {
                epoch,
                loss: epoch_loss,
                steps,
                samples,
            });
            report.steps += steps;
            report.samples += samples;
            report.final_loss = epoch_loss;
        }

        if !self.sync_model || compiled.report().aliases != 0 {
            report.write_back = compiled.write_back(model, &store)?;
        }
        model.set_training(false);
        Ok(report)
    }

    /// Save model parameters and optimizer state.
    ///
    /// `parameters` must be a store built from the same graph as the model
    /// (for example `CompiledModel::parameter_store`). Optimizer state is keyed
    /// by stable parameter name, so it survives a rebuild.
    pub fn save_checkpoint<M: Module + ?Sized>(
        &self,
        path: impl AsRef<std::path::Path>,
        model: &M,
        parameters: &ParameterStore<f32>,
        step: u64,
    ) -> Result<crate::checkpoint::LoadReport> {
        let mut checkpoint = crate::Checkpoint::from_model(model)?;
        if let Some(optimizer) = &self.optimizer {
            checkpoint.optimizer = optimizer.state_dict(parameters)?;
        }
        checkpoint.step = step;
        checkpoint.save(path)?;
        Ok(crate::checkpoint::LoadReport {
            loaded: checkpoint.parameters.len(),
            ..Default::default()
        })
    }

    /// Restore model parameters and optimizer state from a checkpoint.
    pub fn load_checkpoint<M: Module + ?Sized>(
        &mut self,
        path: impl AsRef<std::path::Path>,
        model: &mut M,
        parameters: &ParameterStore<f32>,
    ) -> Result<crate::checkpoint::LoadReport> {
        let checkpoint = crate::Checkpoint::load(path)?;
        let report = checkpoint.apply(model)?;
        if let (Some(optimizer), Some(state)) = (&mut self.optimizer, &checkpoint.optimizer) {
            optimizer.load_state_dict(parameters, state)?;
        }
        Ok(report)
    }

    /// Compute the mean loss over `dataset` without updating parameters.
    pub fn evaluate<M: Module + ?Sized>(&self, model: &M, dataset: &dyn Dataset) -> Result<f32> {
        self.ensure_cpu()?;
        let loss = self
            .loss
            .as_deref()
            .ok_or_else(|| Error::invalid_graph("Trainer requires a loss function"))?;
        let input_specs: Vec<InputSpec> = dataset
            .input_specs()
            .into_iter()
            .map(|spec| InputSpec::batched(spec.name, spec.dims))
            .collect();
        let target_specs = dataset.target_specs();
        if target_specs.len() != 1 {
            return Err(Error::unsupported(
                "evaluate supports exactly one target tensor",
            ));
        }
        let target = InputSpec::batched(target_specs[0].name.clone(), target_specs[0].dims.clone());
        let options = ModelCompileOptions {
            custom_kernels: self.custom_kernels.clone(),
            ..Default::default()
        };
        let compiled =
            CompiledModel::compile_loss(model, &input_specs, target.clone(), loss, options)?;
        let store = compiled.parameter_store(model)?;
        let mut loader = DataLoader::new(self.batch_size)?;
        if let Some(seed) = self.shuffle {
            loader = loader.shuffle(seed);
        }
        let mut total = 0.0f64;
        let mut samples = 0usize;
        for batch in loader.batches(dataset.len(), 0) {
            let mut batch_samples = Vec::with_capacity(batch.len());
            for index in &batch {
                batch_samples.push(dataset.get(*index)?);
            }
            let (input_tensors, target_tensors) = collate(&batch_samples)?;
            let mut owned: Vec<(String, Tensor<f32>)> = input_specs
                .iter()
                .zip(input_tensors)
                .map(|(spec, tensor)| (spec.name.clone(), tensor))
                .collect();
            if let Some(first) = target_tensors.into_iter().next() {
                owned.push((target.name.clone(), first));
            }
            let borrowed: Vec<(&str, &Tensor<f32>)> = owned
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor))
                .collect();
            total += compiled.evaluate(&store, &borrowed)? as f64 * batch.len() as f64;
            samples += batch.len();
        }
        Ok(if samples == 0 {
            0.0
        } else {
            (total / samples as f64) as f32
        })
    }
}

impl TrainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn optimizer<O: Optimizer<f32> + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Some(Box::new(optimizer));
        self
    }

    pub fn loss<L: Loss + 'static>(mut self, loss: L) -> Self {
        self.loss = Some(Box::new(loss));
        self
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        if batch_size > 0 {
            self.batch_size = batch_size;
        }
        self
    }

    pub fn shuffle(mut self, seed: Option<u64>) -> Self {
        self.shuffle = seed;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn sync_model(mut self, sync: bool) -> Self {
        self.sync_model = sync;
        self
    }

    pub fn scheduler<S: LrScheduler + 'static>(mut self, scheduler: S) -> Self {
        self.scheduler = Some(Box::new(scheduler));
        self
    }

    pub fn custom_kernels(mut self, kernels: Arc<CustomKernelRegistry<f32>>) -> Self {
        self.custom_kernels = Some(kernels);
        self
    }

    /// Accumulate gradients over `steps` micro-batches before updating.
    pub fn accumulation_steps(mut self, steps: usize) -> Self {
        self.accumulation = steps.max(1);
        self
    }

    pub fn build(self) -> Result<Trainer> {
        if self.optimizer.is_none() {
            return Err(Error::invalid_graph("TrainerBuilder requires an optimizer"));
        }
        if self.loss.is_none() {
            return Err(Error::invalid_graph("TrainerBuilder requires a loss"));
        }
        Ok(self.into_trainer())
    }

    fn into_trainer(self) -> Trainer {
        Trainer {
            optimizer: self.optimizer,
            loss: self.loss,
            device: self.device,
            batch_size: self.batch_size,
            shuffle: self.shuffle,
            drop_last: self.drop_last,
            sync_model: self.sync_model,
            scheduler: self.scheduler,
            custom_kernels: self.custom_kernels,
            accumulation: self.accumulation,
        }
    }
}

/// A parameter store detached from a model, useful for checkpoint round trips.
pub fn detached_parameters<M: Module + ?Sized>(
    model: &M,
    compiled: &CompiledModel,
) -> Result<ParameterStore<f32>> {
    compiled.parameter_store(model)
}

fn accumulate_gradients(
    target: &mut oxide_ml_optim::Gradients<f32>,
    source: &oxide_ml_optim::Gradients<f32>,
) -> Result<()> {
    for (parameter, gradient) in source {
        match target.get_mut(parameter) {
            Some(existing) => {
                if existing.data().len() != gradient.data().len() {
                    return Err(Error::shape_mismatch(
                        format!("gradient accumulation for {parameter}"),
                        existing.shape().clone(),
                        gradient.shape().clone(),
                    ));
                }
                for (slot, value) in existing.data_mut().iter_mut().zip(gradient.data()) {
                    *slot += *value;
                }
            }
            None => {
                target.insert(*parameter, gradient.clone());
            }
        }
    }
    Ok(())
}

fn scale_gradients(gradients: &mut oxide_ml_optim::Gradients<f32>, factor: f32) {
    for gradient in gradients.values_mut() {
        for value in gradient.data_mut() {
            *value *= factor;
        }
    }
}

/// Sum gradients into `target` (utility for custom training loops).
pub fn merge_gradients(
    target: &mut oxide_ml_optim::Gradients<f32>,
    source: &oxide_ml_optim::Gradients<f32>,
) -> Result<()> {
    accumulate_gradients(target, source)
}

/// Build a gradient map keyed by parameter id from named tensors.
pub fn gradients_from_named(
    pairs: impl IntoIterator<Item = (ParameterId, DenseTensor<f32>)>,
) -> oxide_ml_optim::Gradients<f32> {
    pairs.into_iter().collect()
}
