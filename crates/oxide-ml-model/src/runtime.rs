//! Compile a module tree into an executable graph and run it.
//!
//! The pipeline is:
//!
//! ```text
//! Module tree -> parameter discovery -> GraphBuilder -> Graph IR
//!             -> Graph verification -> optimization -> memory planning
//!             -> CPU executor (or a CUDA model built from the same graph)
//! ```
//!
//! The graph is built once per input shape signature and reused across batches.
//! Parameters are bound by stable name, so a checkpoint, a parameter store and
//! a compiled graph never depend on construction order.

use std::collections::HashMap;
use std::sync::Arc;

use oxide_ml_autograd::{GradientRegistry, TrainingGraph, build_training_graph_with_registry};
use oxide_ml_compiler::{CompileOptions, CompileReport, Compiler, KernelPlan, PassToggles};
use oxide_ml_core::{
    DType, DenseTensor, Device, Dim, Error, MemoryPlan, ParameterId, ParameterStore, Result, Shape,
    ValueId,
};
use oxide_ml_cpu::{CpuBackend, ExecutionInputs, ExecutionResult, Executor};
use oxide_ml_graph::{Graph, GraphVerifier};
use oxide_ml_kernel::{CustomKernelRegistry, KernelCatalog};
use oxide_ml_optim::Gradients;

use crate::context::ModelContext;
use crate::loss::Loss;
use crate::module::{Module, ModuleMode, ModulePath};
use crate::parameter::{ParameterRegistry, ParameterRole, ParameterSpec};
use crate::tensor::Tensor;

const OUTPUT_PREFIX: &str = "__oxide_output_";
const LOSS_NAME: &str = "__oxide_loss";

/// Description of one graph input.
///
/// `batch: true` replaces the leading dimension with a shared symbolic batch
/// dimension, which lets one compiled graph serve every batch size. `dims` are
/// the per-sample dimensions in that case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputSpec {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: DType,
    pub batch: bool,
}

impl InputSpec {
    pub fn batched(name: impl Into<String>, dims: impl Into<Vec<usize>>) -> Self {
        InputSpec {
            name: name.into(),
            dims: dims.into(),
            dtype: DType::F32,
            batch: true,
        }
    }

    pub fn unbatched(name: impl Into<String>, dims: impl Into<Vec<usize>>) -> Self {
        InputSpec {
            name: name.into(),
            dims: dims.into(),
            dtype: DType::F32,
            batch: false,
        }
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    fn shape(&self, batch_symbol: Option<oxide_ml_core::SymbolId>) -> Result<Shape> {
        if self.batch {
            let symbol = batch_symbol.ok_or_else(|| {
                Error::invalid_shape("input", "batched input requires a batch symbol")
            })?;
            let mut dims = Vec::with_capacity(self.dims.len() + 1);
            dims.push(Dim::Symbol(symbol));
            dims.extend(self.dims.iter().copied().map(Dim::Static));
            Ok(Shape::new(dims))
        } else {
            if self.dims.contains(&0) {
                return Err(Error::invalid_shape(
                    "input",
                    format!("input {:?} has a zero-sized dimension", self.name),
                ));
            }
            Ok(Shape::from_static(&self.dims))
        }
    }
}

/// Options controlling graph compilation and execution.
#[derive(Clone, Debug)]
pub struct ModelCompileOptions {
    pub optimize: bool,
    pub memory_plan: bool,
    pub mode: ModuleMode,
    pub gradient_registry: Option<GradientRegistry>,
    /// Registry of user-defined kernels used by the graph.
    pub custom_kernels: Option<Arc<CustomKernelRegistry<f32>>>,
    /// Candidate kernels for selection. `None` uses the generic CPU catalog.
    pub kernel_catalog: Option<Arc<KernelCatalog>>,
    /// Target architecture used for provider filtering.
    pub architecture: String,
    /// Fail compilation when a node has no compatible provider.
    pub strict_kernel_selection: bool,
}

impl Default for ModelCompileOptions {
    fn default() -> Self {
        ModelCompileOptions {
            optimize: true,
            memory_plan: true,
            mode: ModuleMode::Eval,
            gradient_registry: None,
            custom_kernels: None,
            kernel_catalog: None,
            architecture: "portable".to_string(),
            strict_kernel_selection: false,
        }
    }
}

impl ModelCompileOptions {
    pub fn training() -> Self {
        ModelCompileOptions {
            mode: ModuleMode::Train,
            ..Default::default()
        }
    }

    pub fn without_optimization(mut self) -> Self {
        self.optimize = false;
        self.memory_plan = false;
        self
    }
}

/// Summary of one compiled model.
#[derive(Clone, Debug, Default)]
pub struct ModelReport {
    pub parameters: usize,
    pub trainable_parameters: usize,
    pub buffers: usize,
    pub values: usize,
    pub nodes: usize,
    pub optimized: bool,
    pub passes: Vec<String>,
    pub memory_plan_bytes: usize,
    pub aliases: usize,
    /// Nodes with a selected kernel provider.
    pub kernels: usize,
    /// Fused operations in the optimized graph.
    pub fusions: usize,
    /// Compiler wall-clock time in microseconds.
    pub compile_time_micros: u128,
    pub diagnostics: Vec<String>,
}

/// A graph built from a module tree, before execution is attached.
pub struct BuiltModel {
    pub graph: Graph,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub loss: Option<ValueId>,
    pub registry: ParameterRegistry,
    pub input_specs: Vec<InputSpec>,
    pub target_spec: Option<InputSpec>,
    pub target_value: Option<ValueId>,
}

impl BuiltModel {
    /// Discover parameters and build the forward graph.
    pub fn build<M: Module + ?Sized>(
        model: &M,
        inputs: &[InputSpec],
        loss: Option<(&dyn Loss, InputSpec)>,
        mode: ModuleMode,
    ) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::invalid_graph(
                "a model requires at least one input specification",
            ));
        }
        let mut names = std::collections::HashSet::new();
        for spec in inputs {
            if !names.insert(spec.name.clone()) {
                return Err(Error::invalid_graph(format!(
                    "duplicate input name {:?}",
                    spec.name
                )));
            }
        }
        let mut registry = ParameterRegistry::new();
        model.visit_parameters(&ModulePath::root(), &mut registry)?;

        let mut builder = oxide_ml_graph::GraphBuilder::new();
        let needs_batch = inputs.iter().any(|spec| spec.batch)
            || loss.as_ref().map(|(_, spec)| spec.batch).unwrap_or(false);
        let batch_symbol = needs_batch.then(|| builder.symbol("batch"));

        let mut input_values = Vec::with_capacity(inputs.len());
        for spec in inputs {
            let shape = spec.shape(batch_symbol)?;
            input_values.push(builder.input(&spec.name, shape, spec.dtype));
        }

        let mut parameters = HashMap::new();
        for spec in registry.specs() {
            let value = builder.try_parameter(
                &spec.name,
                Shape::from_static(&spec.shape),
                spec.dtype,
                spec.role.is_trainable(),
            )?;
            parameters.insert(spec.name.clone(), value);
        }
        let aliases: HashMap<String, String> = registry
            .aliases()
            .iter()
            .map(|(alias, canonical)| (alias.clone(), canonical.clone()))
            .collect();

        let mut cx = ModelContext::new(
            builder,
            parameters,
            aliases,
            mode,
            DType::F32,
            input_values.clone(),
        );

        let outputs = model.forward_multi(&mut cx, &input_values)?;
        if outputs.is_empty() {
            return Err(Error::invalid_graph("model forward produced no outputs"));
        }
        for (index, output) in outputs.iter().enumerate() {
            let name = format!("{OUTPUT_PREFIX}{index}");
            cx.builder().set_value_name(*output, name)?;
        }

        let (loss_value, target_spec) = match loss {
            Some((loss, target_spec)) => {
                let target_shape = target_spec.shape(batch_symbol)?;
                let target = cx
                    .builder()
                    .input(&target_spec.name, target_shape, target_spec.dtype);
                let value = loss.forward(&mut cx, outputs[0], target)?;
                cx.builder().set_value_name(value, LOSS_NAME)?;
                (Some(value), Some(target_spec))
            }
            None => (None, None),
        };

        let mut graph_outputs = outputs.clone();
        if let Some(loss) = loss_value {
            graph_outputs.push(loss);
        }
        let graph = cx.into_builder().finish(&graph_outputs)?;
        validate_parameters(&registry, &graph)?;

        // The loss target is a graph input. Execution must supply it too, so it
        // joins the recorded input list after the model inputs.
        let target_value = target_spec.as_ref().map(|spec| {
            graph
                .value_by_name(&spec.name)
                .expect("target input was created above")
                .id
        });
        let mut execution_inputs = input_values.clone();
        if let Some(value) = target_value {
            execution_inputs.push(value);
        }

        Ok(BuiltModel {
            graph,
            inputs: execution_inputs,
            outputs,
            loss: loss_value,
            registry,
            input_specs: inputs.to_vec(),
            target_spec,
            target_value,
        })
    }

    /// Run graph optimization and memory planning.
    pub fn optimize(
        self,
        options: &ModelCompileOptions,
    ) -> Result<(Self, Option<CompileReport>, Option<MemoryPlan>)> {
        if !options.optimize && !options.memory_plan {
            return Ok((self, None, None));
        }
        let compiler = Compiler::new(
            Box::new(CpuBackend::<f32>::new()),
            CompileOptions {
                optimize: options.optimize,
                memory_planning: options.memory_plan,
                dump_graphs: false,
                passes: PassToggles::default(),
                kernel_catalog: options.kernel_catalog.clone(),
                architecture: options.architecture.clone(),
                strict_kernel_selection: options.strict_kernel_selection,
            },
        );
        let report = compiler.compile(self.graph.clone())?;
        let optimized = BuiltModel {
            graph: report.graph,
            inputs: self.inputs,
            outputs: self.outputs,
            loss: self.loss,
            registry: self.registry,
            input_specs: self.input_specs,
            target_spec: self.target_spec,
            target_value: self.target_value,
        };
        Ok((optimized, Some(report.report), report.memory_plan))
    }
}

fn validate_parameters(registry: &ParameterRegistry, graph: &Graph) -> Result<()> {
    if graph.parameters().len() != registry.len() {
        return Err(Error::invalid_graph(format!(
            "module traversal discovered {} parameters but the graph registered {}",
            registry.len(),
            graph.parameters().len()
        )));
    }
    for spec in registry.specs() {
        let parameter = graph
            .parameters()
            .iter()
            .find(|parameter| parameter.name == spec.name)
            .ok_or_else(|| {
                Error::invalid_graph(format!(
                    "parameter {:?} was discovered but never registered in the graph",
                    spec.name
                ))
            })?;
        let expected = Shape::from_static(&spec.shape);
        if parameter.shape != expected {
            return Err(Error::shape_mismatch(
                format!("parameter {:?}", spec.name),
                expected,
                parameter.shape.clone(),
            ));
        }
        if parameter.trainable != spec.role.is_trainable() {
            return Err(Error::invalid_graph(format!(
                "parameter {:?} trainable flag mismatch",
                spec.name
            )));
        }
    }
    Ok(())
}

/// A module compiled into forward and (optionally) training graphs.
pub struct CompiledModel {
    forward_graph: Graph,
    training: Option<TrainingGraph>,
    forward_executor: Executor<f32>,
    training_executor: Option<Executor<f32>>,
    memory_plan: Option<MemoryPlan>,
    parameter_ids: HashMap<String, ParameterId>,
    parameter_specs: Vec<ParameterSpec>,
    aliases: HashMap<String, String>,
    input_names: Vec<String>,
    input_values: Vec<ValueId>,
    output_values: Vec<ValueId>,
    loss_value: Option<ValueId>,
    target_name: Option<String>,
    kernel_plan: KernelPlan,
    report: ModelReport,
}

impl std::fmt::Debug for CompiledModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledModel")
            .field("inputs", &self.input_names)
            .field("outputs", &self.output_values.len())
            .field("parameters", &self.parameter_specs.len())
            .field("training", &self.training.is_some())
            .finish()
    }
}

impl CompiledModel {
    /// Compile a forward-only model (inference).
    pub fn compile<M: Module + ?Sized>(
        model: &M,
        inputs: &[InputSpec],
        options: ModelCompileOptions,
    ) -> Result<Self> {
        Self::compile_inner(model, inputs, None, false, options)
    }

    /// Compile a model with a loss but without building the backward graph.
    ///
    /// Used for evaluation, where only the forward loss is needed.
    pub fn compile_loss<M: Module + ?Sized>(
        model: &M,
        inputs: &[InputSpec],
        target: InputSpec,
        loss: &dyn Loss,
        options: ModelCompileOptions,
    ) -> Result<Self> {
        Self::compile_inner(model, inputs, Some((loss, target)), false, options)
    }

    /// Compile a model with a loss and target input for training.
    pub fn compile_training<M: Module + ?Sized>(
        model: &M,
        inputs: &[InputSpec],
        target: InputSpec,
        loss: &dyn Loss,
        options: ModelCompileOptions,
    ) -> Result<Self> {
        Self::compile_inner(model, inputs, Some((loss, target)), true, options)
    }

    fn compile_inner<M: Module + ?Sized>(
        model: &M,
        inputs: &[InputSpec],
        loss: Option<(&dyn Loss, InputSpec)>,
        backward: bool,
        options: ModelCompileOptions,
    ) -> Result<Self> {
        let built = BuiltModel::build(model, inputs, loss, options.mode)?;
        Self::from_built(built, &options, backward)
    }

    /// Compile an already-built graph. Used by the compile cache so a cached
    /// entry is keyed by the canonical hash of exactly the graph it compiled.
    pub fn from_built(
        built: BuiltModel,
        options: &ModelCompileOptions,
        backward: bool,
    ) -> Result<Self> {
        let (built, compile_report, memory_plan) = built.optimize(options)?;
        GraphVerifier::verify(&built.graph)?;

        let loss_value = built.loss;
        let training = match loss_value.filter(|_| backward) {
            Some(loss_value) => {
                let registry = match (&options.gradient_registry, &options.custom_kernels) {
                    (Some(registry), _) => registry.clone(),
                    (None, Some(kernels)) => {
                        GradientRegistry::with_custom_kernels(kernels.as_ref())
                    }
                    (None, None) => GradientRegistry::builtin(),
                };
                Some(build_training_graph_with_registry(
                    &built.graph,
                    loss_value,
                    &registry,
                )?)
            }
            None => None,
        };

        let forward_executor = match &options.custom_kernels {
            Some(kernels) => Executor::with_custom(built.graph.clone(), Arc::clone(kernels))?,
            None => Executor::new(built.graph.clone())?,
        };
        let training_executor = match (&training, &options.custom_kernels) {
            (Some(training), Some(kernels)) => Some(Executor::with_custom(
                training.graph.clone(),
                Arc::clone(kernels),
            )?),
            (Some(training), None) => Some(Executor::new(training.graph.clone())?),
            (None, _) => None,
        };

        let mut parameter_ids = HashMap::new();
        for parameter in built.graph.parameters() {
            parameter_ids.insert(parameter.name.clone(), parameter.id);
        }

        let mut report = ModelReport {
            parameters: built.registry.len(),
            trainable_parameters: built
                .registry
                .specs()
                .iter()
                .filter(|spec| spec.role.is_trainable())
                .count(),
            buffers: built
                .registry
                .specs()
                .iter()
                .filter(|spec| spec.role == ParameterRole::Buffer)
                .count(),
            values: built.graph.values().len(),
            nodes: built.graph.nodes().len(),
            optimized: options.optimize,
            passes: compile_report
                .as_ref()
                .map(|report| {
                    report
                        .passes
                        .iter()
                        .map(|pass| pass.name.to_string())
                        .collect()
                })
                .unwrap_or_default(),
            memory_plan_bytes: memory_plan
                .as_ref()
                .map(MemoryPlan::peak_bytes)
                .unwrap_or(0),
            aliases: built.registry.aliases().len(),
            kernels: compile_report
                .as_ref()
                .map(|report| report.kernel_count)
                .unwrap_or(0),
            fusions: compile_report
                .as_ref()
                .map(|report| report.fusion_count)
                .unwrap_or(0),
            compile_time_micros: compile_report
                .as_ref()
                .map(|report| report.compilation_time_micros)
                .unwrap_or(0),
            diagnostics: compile_report
                .as_ref()
                .map(|report| report.diagnostics.clone())
                .unwrap_or_default(),
        };
        if let Some(compile_report) = &compile_report {
            report.nodes = compile_report.after.nodes;
            report.values = compile_report.after.values;
        }

        Ok(CompiledModel {
            forward_graph: built.graph,
            training,
            forward_executor,
            training_executor,
            memory_plan,
            parameter_ids,
            parameter_specs: built.registry.specs().to_vec(),
            aliases: built
                .registry
                .aliases()
                .iter()
                .map(|(alias, canonical)| (alias.clone(), canonical.clone()))
                .collect(),
            input_names: built
                .input_specs
                .iter()
                .map(|spec| spec.name.clone())
                .chain(built.target_spec.iter().map(|spec| spec.name.clone()))
                .collect(),
            input_values: built.inputs,
            output_values: built.outputs,
            loss_value,
            target_name: built.target_spec.as_ref().map(|spec| spec.name.clone()),
            kernel_plan: compile_report
                .as_ref()
                .map(|report| report.kernel_plan.clone())
                .unwrap_or_default(),
            report,
        })
    }

    pub fn report(&self) -> &ModelReport {
        &self.report
    }

    /// Kernel selection result for the compiled graph.
    pub fn kernel_plan(&self) -> &KernelPlan {
        &self.kernel_plan
    }

    pub fn forward_graph(&self) -> &Graph {
        &self.forward_graph
    }

    pub fn training_graph(&self) -> Option<&TrainingGraph> {
        self.training.as_ref()
    }

    pub fn training_executor(&self) -> Option<&Executor<f32>> {
        self.training_executor.as_ref()
    }

    pub fn parameter_specs(&self) -> &[ParameterSpec] {
        &self.parameter_specs
    }

    pub fn parameter_ids(&self) -> &HashMap<String, ParameterId> {
        &self.parameter_ids
    }

    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    pub fn aliases(&self) -> &HashMap<String, String> {
        &self.aliases
    }

    pub fn output_values(&self) -> &[ValueId] {
        &self.output_values
    }

    pub fn loss_value(&self) -> Option<ValueId> {
        self.loss_value
    }

    pub fn target_name(&self) -> Option<&str> {
        self.target_name.as_deref()
    }

    /// Materialize a parameter store from the current module weights.
    pub fn parameter_store<M: Module + ?Sized>(&self, model: &M) -> Result<ParameterStore<f32>> {
        parameter_store_named(
            model,
            &self.parameter_specs,
            &self.parameter_ids,
            &self.aliases,
        )
    }

    /// Copy optimizer-updated parameters back into the module tree.
    pub fn write_back<M: Module + ?Sized>(
        &self,
        model: &mut M,
        store: &ParameterStore<f32>,
    ) -> Result<WriteBackReport> {
        write_back_named(model, &self.parameter_ids, &self.aliases, store)
    }

    fn execution_inputs(&self, inputs: &[(&str, &Tensor<f32>)]) -> Result<ExecutionInputs<f32>> {
        let mut execution = ExecutionInputs::new();
        for (name, tensor) in inputs {
            let index = self
                .input_names
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| Error::execution(format!("unknown model input {name:?}")))?;
            if tensor.device() != Device::Cpu {
                return Err(Error::unsupported(format!(
                    "the portable Tensor is host-resident; move the data to the CPU or use the CUDA backend directly (input {name:?} is on {})",
                    tensor.device()
                )));
            }
            execution.insert(self.input_values[index], tensor.dense().clone());
        }
        for (index, value) in self.input_values.iter().enumerate() {
            if execution.get(*value).is_none() {
                return Err(Error::execution(format!(
                    "missing model input {:?}",
                    self.input_names[index]
                )));
            }
        }
        Ok(execution)
    }

    /// Forward-only execution. Returns model outputs in declaration order.
    pub fn run(
        &self,
        parameters: &ParameterStore<f32>,
        inputs: &[(&str, &Tensor<f32>)],
    ) -> Result<Vec<Tensor<f32>>> {
        let execution = self.execution_inputs(inputs)?;
        let result = match &self.memory_plan {
            Some(plan) => {
                let (result, _) = self
                    .forward_executor
                    .run_planned(parameters, &execution, plan)?;
                result
            }
            None => self.forward_executor.run(parameters, &execution)?,
        };
        self.output_values
            .iter()
            .map(|value| Ok(Tensor::from_dense(result.value(*value)?.clone())))
            .collect()
    }

    /// Training execution: forward, backward, loss and parameter gradients.
    pub fn run_training(
        &self,
        parameters: &ParameterStore<f32>,
        inputs: &[(&str, &Tensor<f32>)],
    ) -> Result<TrainingOutput> {
        let training = self
            .training
            .as_ref()
            .ok_or_else(|| Error::execution("model was compiled without a loss"))?;
        let executor = self
            .training_executor
            .as_ref()
            .ok_or_else(|| Error::execution("model has no training executor"))?;
        let execution = self.execution_inputs(inputs)?;
        let result = executor.run(parameters, &execution)?;
        let loss_value = self
            .loss_value
            .ok_or_else(|| Error::execution("compiled model has no loss value"))?;
        let loss = result.value(loss_value)?;
        if loss.numel() != 1 {
            return Err(Error::execution("training loss must be a scalar"));
        }
        let gradients = collect_gradients(training, &result)?;
        let outputs = self
            .output_values
            .iter()
            .map(|value| Ok(Tensor::from_dense(result.value(*value)?.clone())))
            .collect::<Result<Vec<_>>>()?;
        Ok(TrainingOutput {
            loss: loss.data()[0],
            gradients,
            outputs,
        })
    }

    /// Forward loss for evaluation (no gradients are built).
    pub fn evaluate(
        &self,
        parameters: &ParameterStore<f32>,
        inputs: &[(&str, &Tensor<f32>)],
    ) -> Result<f32> {
        let execution = self.execution_inputs(inputs)?;
        let result: ExecutionResult<f32> = match &self.memory_plan {
            Some(plan) => {
                let (result, _) = self
                    .forward_executor
                    .run_planned(parameters, &execution, plan)?;
                result
            }
            None => self.forward_executor.run(parameters, &execution)?,
        };
        let loss = self
            .loss_value
            .ok_or_else(|| Error::execution("model was compiled without a loss"))?;
        let tensor = result.value(loss)?;
        if tensor.numel() != 1 {
            return Err(Error::execution("loss must be scalar"));
        }
        Ok(tensor.data()[0])
    }
}

/// Result of one training execution.
#[derive(Clone, Debug)]
pub struct TrainingOutput {
    pub loss: f32,
    pub gradients: Gradients<f32>,
    pub outputs: Vec<Tensor<f32>>,
}

/// Report from copying parameter values back into a module tree.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WriteBackReport {
    pub updated: usize,
    pub skipped_aliases: usize,
    pub unexpected: Vec<String>,
}

/// Materialize a parameter store by stable name.
///
/// Backend crates (for example the CUDA model) use this to build their device
/// parameter storage from exactly the same names as the CPU path.
pub fn parameter_store_named<M: Module + ?Sized>(
    model: &M,
    specs: &[ParameterSpec],
    parameter_ids: &HashMap<String, ParameterId>,
    aliases: &HashMap<String, String>,
) -> Result<ParameterStore<f32>> {
    let mut tensors: HashMap<String, (DenseTensor<f32>, ParameterRole)> = HashMap::new();
    model.visit_tensors(&ModulePath::root(), &mut |path, tensor, role| {
        let name = path.as_string();
        if tensors
            .insert(name.clone(), (tensor.clone(), role))
            .is_some()
        {
            return Err(Error::invalid_graph(format!(
                "module traversal visited parameter {name:?} twice"
            )));
        }
        Ok(())
    })?;

    let mut store = ParameterStore::new();
    for spec in specs {
        let id = *parameter_ids.get(&spec.name).ok_or_else(|| {
            Error::invalid_graph(format!("compiled graph has no parameter {:?}", spec.name))
        })?;
        let (data, role) = tensors.get(&spec.name).ok_or_else(|| {
            Error::invalid_graph(format!(
                "module traversal did not provide a tensor for {:?}",
                spec.name
            ))
        })?;
        if data.dims() != spec.shape {
            return Err(Error::shape_mismatch(
                format!("parameter {:?}", spec.name),
                Shape::from_static(&spec.shape),
                data.shape().clone(),
            ));
        }
        store.add(id, spec.name.clone(), data.clone(), role.is_trainable())?;
    }
    for alias in aliases.keys() {
        if !tensors.contains_key(alias) {
            return Err(Error::invalid_graph(format!(
                "shared parameter alias {alias:?} was not visited"
            )));
        }
    }
    Ok(store)
}

/// Copy parameter values back into a module tree by stable name.
pub fn write_back_named<M: Module + ?Sized>(
    model: &mut M,
    parameter_ids: &HashMap<String, ParameterId>,
    aliases: &HashMap<String, String>,
    store: &ParameterStore<f32>,
) -> Result<WriteBackReport> {
    let mut report = WriteBackReport::default();
    model.visit_tensors_mut(&ModulePath::root(), &mut |path, tensor, _role| {
        let name = path.as_string();
        if aliases.contains_key(&name) {
            report.skipped_aliases += 1;
            return Ok(());
        }
        let Some(id) = parameter_ids.get(&name) else {
            report.unexpected.push(name);
            return Ok(());
        };
        let stored = store.get(*id)?;
        if tensor.dims() != stored.data.dims() {
            return Err(Error::shape_mismatch(
                format!("parameter {name:?}"),
                tensor.shape().clone(),
                stored.data.shape().clone(),
            ));
        }
        tensor.copy_from(&stored.data);
        report.updated += 1;
        Ok(())
    })?;
    Ok(report)
}

/// Collect parameter gradients from a training execution.
pub fn collect_gradients(
    training: &TrainingGraph,
    result: &ExecutionResult<f32>,
) -> Result<Gradients<f32>> {
    let mut gradients = HashMap::new();
    for (parameter, gradient) in training.parameter_gradients() {
        gradients.insert(parameter, result.value(gradient)?.clone());
    }
    Ok(gradients)
}

/// Build, compile and run a one-shot forward pass on host tensors.
///
/// Compilation goes through the process-wide compile cache.
pub fn predict_many<M: Module + ?Sized>(
    model: &M,
    inputs: Vec<Tensor<f32>>,
) -> Result<Vec<Tensor<f32>>> {
    predict_many_with_cache(model, inputs, crate::cache::global())
}

/// Like [`predict_many`] but uses an explicit cache.
pub fn predict_many_with_cache<M: Module + ?Sized>(
    model: &M,
    inputs: Vec<Tensor<f32>>,
    cache: &crate::cache::CompileCache,
) -> Result<Vec<Tensor<f32>>> {
    if inputs.is_empty() {
        return Err(Error::invalid_graph("predict requires at least one input"));
    }
    let specs: Vec<InputSpec> = inputs
        .iter()
        .enumerate()
        .map(|(index, tensor)| InputSpec::unbatched(format!("input{index}"), tensor.dims()))
        .collect();
    let options = ModelCompileOptions::default();
    let compiled = crate::cache::compile_cached_with(cache, model, &specs, &options)?;
    let store = compiled.parameter_store(model)?;
    let refs: Vec<(&str, &Tensor<f32>)> = specs
        .iter()
        .zip(inputs.iter())
        .map(|(spec, tensor)| (spec.name.as_str(), tensor))
        .collect();
    compiled.run(&store, &refs)
}

/// Convenience wrapper for single-input / single-output models.
pub fn predict_single<M: Module + ?Sized>(model: &M, input: Tensor<f32>) -> Result<Tensor<f32>> {
    let mut outputs = predict_many(model, vec![input])?;
    if outputs.len() != 1 {
        return Err(Error::invalid_graph(format!(
            "predict_single expects one output, model produced {}",
            outputs.len()
        )));
    }
    Ok(outputs.remove(0))
}
