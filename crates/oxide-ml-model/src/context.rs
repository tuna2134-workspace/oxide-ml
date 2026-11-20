//! Graph-building context handed to [`crate::Module::forward`].
//!
//! The context owns the `GraphBuilder`, the current module path, and the
//! name -> graph-value map for parameters. Modules ask for parameters by their
//! *relative* name (`weight`), and the context resolves it against the path
//! stack maintained by `#[derive(Module)]`.

use std::collections::HashMap;

use oxide_ml_core::{DType, Error, Result, ValueId};
use oxide_ml_graph::GraphBuilder;

use crate::module::{Module, ModuleMode};

pub struct ModelContext {
    builder: GraphBuilder,
    parameters: HashMap<String, ValueId>,
    aliases: HashMap<String, String>,
    path: Vec<String>,
    mode: ModuleMode,
    dtype: DType,
    inputs: Vec<ValueId>,
    operations: usize,
}

impl ModelContext {
    pub(crate) fn new(
        builder: GraphBuilder,
        parameters: HashMap<String, ValueId>,
        aliases: HashMap<String, String>,
        mode: ModuleMode,
        dtype: DType,
        inputs: Vec<ValueId>,
    ) -> Self {
        ModelContext {
            builder,
            parameters,
            aliases,
            path: Vec::new(),
            mode,
            dtype,
            inputs,
            operations: 0,
        }
    }

    pub fn mode(&self) -> ModuleMode {
        self.mode
    }

    pub fn is_training(&self) -> bool {
        self.mode.is_training()
    }

    pub fn set_mode(&mut self, mode: ModuleMode) {
        self.mode = mode;
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    pub fn input(&self, index: usize) -> Result<ValueId> {
        self.inputs.get(index).copied().ok_or_else(|| {
            Error::invalid_graph(format!("model input index {index} is out of range"))
        })
    }

    pub fn depth(&self) -> usize {
        self.path.len()
    }

    pub fn current_path(&self) -> String {
        self.path.join(".")
    }

    pub fn path(&self) -> &[String] {
        &self.path
    }

    pub fn enter(&mut self, segment: &str) {
        self.path.push(segment.to_string());
    }

    pub fn leave(&mut self) {
        self.path.pop();
    }

    pub fn truncate(&mut self, depth: usize) {
        self.path.truncate(depth);
    }

    /// Number of graph operations appended through this context.
    pub fn operations(&self) -> usize {
        self.operations
    }

    pub(crate) fn count_operation(&mut self) {
        self.operations += 1;
    }

    /// Resolve a parameter relative to the current module path.
    pub fn parameter(&mut self, name: &str) -> Result<ValueId> {
        let mut full = self.current_path();
        if !full.is_empty() {
            full.push('.');
        }
        full.push_str(name);
        if let Some(value) = self.parameters.get(&full) {
            return Ok(*value);
        }
        if let Some(canonical) = self.aliases.get(&full)
            && let Some(value) = self.parameters.get(canonical)
        {
            return Ok(*value);
        }
        Err(Error::invalid_graph(format!(
            "parameter {full:?} was used by a module but not discovered during traversal"
        )))
    }

    pub fn has_parameter(&self, name: &str) -> bool {
        let mut full = self.current_path();
        if !full.is_empty() {
            full.push('.');
        }
        full.push_str(name);
        self.parameters.contains_key(&full) || self.aliases.contains_key(&full)
    }

    pub fn builder(&mut self) -> &mut GraphBuilder {
        &mut self.builder
    }

    pub fn graph(&self) -> &oxide_ml_graph::Graph {
        self.builder.graph()
    }

    pub fn constant_scalar(&mut self, value: f32) -> ValueId {
        self.count_operation();
        self.builder.constant_scalar(value)
    }

    pub fn constant_scalar_as(&mut self, value: f32, dtype: DType) -> Result<ValueId> {
        self.count_operation();
        self.builder.constant_scalar_as(value as f64, dtype)
    }

    pub fn value_shape(&self, value: ValueId) -> Result<oxide_ml_core::Shape> {
        self.builder.value_shape(value)
    }

    pub fn value_dtype(&self, value: ValueId) -> Result<DType> {
        self.builder.value_dtype(value)
    }

    pub(crate) fn into_builder(self) -> GraphBuilder {
        self.builder
    }
}

/// Call a submodule's single-output forward pass with a path scope.
///
/// `segments` is a dot-separated relative path (e.g. `"fc1"` or `"blocks.0"`).
pub fn module_forward<M: Module + ?Sized>(
    cx: &mut ModelContext,
    segments: &str,
    module: &M,
    input: ValueId,
) -> Result<ValueId> {
    let outputs = module_forward_multi(cx, segments, module, &[input])?;
    single_output(outputs)
}

/// Call a submodule's multi-input forward pass with a path scope.
pub fn module_forward_multi<M: Module + ?Sized>(
    cx: &mut ModelContext,
    segments: &str,
    module: &M,
    inputs: &[ValueId],
) -> Result<Vec<ValueId>> {
    let depth = cx.depth();
    for segment in segments.split('.') {
        if !segment.is_empty() {
            cx.enter(segment);
        }
    }
    let result = module.forward_multi(cx, inputs);
    cx.truncate(depth);
    result
}

/// Require exactly one output value.
pub fn single_output(outputs: Vec<ValueId>) -> Result<ValueId> {
    match outputs.as_slice() {
        [value] => Ok(*value),
        other => Err(Error::invalid_graph(format!(
            "module produced {} outputs where exactly one was expected",
            other.len()
        ))),
    }
}

/// Require exactly two outputs from a multi-output module call.
pub fn output_pair(outputs: Vec<ValueId>) -> Result<(ValueId, ValueId)> {
    match outputs.as_slice() {
        [first, second] => Ok((*first, *second)),
        other => Err(Error::invalid_graph(format!(
            "module produced {} outputs where exactly two were expected",
            other.len()
        ))),
    }
}

/// Require exactly three outputs from a multi-output module call.
pub fn output_triple(outputs: Vec<ValueId>) -> Result<(ValueId, ValueId, ValueId)> {
    match outputs.as_slice() {
        [first, second, third] => Ok((*first, *second, *third)),
        other => Err(Error::invalid_graph(format!(
            "module produced {} outputs where exactly three were expected",
            other.len()
        ))),
    }
}

/// Require exactly four outputs from a multi-output module call.
pub fn output_quad(outputs: Vec<ValueId>) -> Result<(ValueId, ValueId, ValueId, ValueId)> {
    match outputs.as_slice() {
        [first, second, third, fourth] => Ok((*first, *second, *third, *fourth)),
        other => Err(Error::invalid_graph(format!(
            "module produced {} outputs where exactly four were expected",
            other.len()
        ))),
    }
}
