//! Module definition and traversal contracts.
//!
//! A [`Module`] is a backend-independent description of a computation plus the
//! parameters it owns. Modules are *not* bound to a `GraphBuilder`: the same
//! value can be traversed to discover parameters (for a parameter store or a
//! checkpoint) and replayed into a graph through a [`crate::ModelContext`].
//!
//! Parameter names are derived from the module hierarchy, not from graph ids.
//! The root path is empty, so a top-level `Linear` owns `weight`/`bias`, while
//! the same `Linear` stored in a field `fc1` owns `fc1.weight`/`fc1.bias`.

use std::fmt;
use std::sync::Arc;

use oxide_ml_core::{DenseTensor, Error, Result, ValueId};

use crate::context::ModelContext;
use crate::parameter::{ParameterRegistry, ParameterRole};

/// Train / eval state propagated through the module tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ModuleMode {
    #[default]
    Train,
    Eval,
}

impl ModuleMode {
    pub const fn is_training(self) -> bool {
        matches!(self, ModuleMode::Train)
    }
}

/// Stable, dot-separated path inside a module tree.
///
/// `ModulePath::root().child("fc1").child("weight")` renders as `fc1.weight`.
/// Paths never contain graph ids, so checkpoints remain compatible across graph
/// rewrites.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct ModulePath {
    segments: Vec<String>,
}

impl ModulePath {
    pub fn root() -> Self {
        ModulePath::default()
    }

    pub fn from_segments(segments: impl IntoIterator<Item = impl Into<String>>) -> Self {
        ModulePath {
            segments: segments.into_iter().map(Into::into).collect(),
        }
    }

    pub fn child(&self, segment: impl Into<String>) -> Self {
        let segment = segment.into();
        let mut segments = self.segments.clone();
        segments.push(segment);
        ModulePath { segments }
    }

    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    pub fn is_root(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn parent(&self) -> Option<ModulePath> {
        if self.segments.is_empty() {
            return None;
        }
        let mut segments = self.segments.clone();
        segments.pop();
        Some(ModulePath { segments })
    }

    /// Dot-joined rendering. The root path is the empty string.
    pub fn as_string(&self) -> String {
        self.segments.join(".")
    }

    /// Join a parameter name onto this path.
    pub fn parameter(&self, name: &str) -> String {
        if self.segments.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.as_string(), name)
        }
    }
}

impl fmt::Display for ModulePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_string())
    }
}

/// One node of the serialized module hierarchy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleDescriptor {
    pub module_type: String,
    pub path: ModulePath,
    pub children: Vec<ModuleDescriptor>,
    pub parameters: Vec<String>,
}

impl ModuleDescriptor {
    pub fn new(module_type: impl Into<String>, path: ModulePath) -> Self {
        ModuleDescriptor {
            module_type: module_type.into(),
            path,
            children: Vec::new(),
            parameters: Vec::new(),
        }
    }

    pub fn find(&self, path: &ModulePath) -> Option<&ModuleDescriptor> {
        if &self.path == path {
            return Some(self);
        }
        self.children.iter().find_map(|child| child.find(path))
    }

    pub fn find_mut(&mut self, path: &ModulePath) -> Option<&mut ModuleDescriptor> {
        if &self.path == path {
            return Some(self);
        }
        self.children
            .iter_mut()
            .find_map(|child| child.find_mut(path))
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
            + self
                .children
                .iter()
                .map(ModuleDescriptor::parameter_count)
                .sum::<usize>()
    }

    pub fn module_count(&self) -> usize {
        1 + self
            .children
            .iter()
            .map(ModuleDescriptor::module_count)
            .sum::<usize>()
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        self.render_into(0, &mut out);
        out
    }

    fn render_into(&self, depth: usize, out: &mut String) {
        for _ in 0..depth {
            out.push_str("  ");
        }
        if self.path.is_root() {
            out.push_str(&format!("{} (root)\n", self.module_type));
        } else {
            out.push_str(&format!("{}: {}\n", self.path, self.module_type));
        }
        for parameter in &self.parameters {
            for _ in 0..depth + 1 {
                out.push_str("  ");
            }
            out.push_str(parameter);
            out.push('\n');
        }
        for child in &self.children {
            child.render_into(depth + 1, out);
        }
    }
}

/// A neural-network building block.
///
/// Implementations normally come from `#[derive(Module)]`; hand-written
/// implementations are supported for containers and adapters.
pub trait Module {
    /// Type name recorded in module metadata (not used for dispatch).
    fn module_type(&self) -> &'static str {
        "module"
    }

    /// Discover parameters under `path` without building a graph.
    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()>;

    /// Append this module's forward computation to `cx`.
    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>>;

    /// Append this module's forward computation for a multi-input module.
    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        if inputs.len() != 1 {
            return Err(Error::invalid_graph(format!(
                "{} expects 1 input, got {}",
                self.module_type(),
                inputs.len()
            )));
        }
        self.forward(cx, inputs[0])
    }

    /// Forward with optional graph inputs (layers that thread cached K/V, PLE
    /// residuals or other per-layer state).
    ///
    /// The default implementation requires every slot and calls
    /// [`Module::forward_multi`], so ordinary modules need no changes. Modules
    /// generated from a `#[model]` method with `Option<Tensor<f32>>`
    /// parameters override this to skip absent optional inputs.
    fn forward_slots(
        &self,
        cx: &mut ModelContext,
        inputs: &[Option<ValueId>],
    ) -> Result<Vec<ValueId>> {
        let mut values = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            values.push(input.ok_or_else(|| {
                Error::invalid_graph(format!(
                    "{} requires input {index}, but it is absent",
                    self.module_type()
                ))
            })?);
        }
        self.forward_multi(cx, &values)
    }

    /// Switch train / eval state. Stateless layers ignore this.
    fn set_training(&mut self, _training: bool) {}

    fn is_training(&self) -> bool {
        true
    }

    /// Visit every concrete parameter tensor (read-only), in stable name order.
    fn visit_tensors(
        &self,
        _path: &ModulePath,
        _visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        Ok(())
    }

    /// Visit every concrete parameter tensor for in-place updates.
    ///
    /// Containers that cannot guarantee unique access (a shared `Arc`) skip the
    /// tensor instead of failing, and report it through the write-back report.
    fn visit_tensors_mut(
        &mut self,
        _path: &ModulePath,
        _visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        Ok(())
    }

    /// Record the module hierarchy.
    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children
            .push(ModuleDescriptor::new(self.module_type(), path.clone()));
    }

    /// Declare the execution state this module needs, if any.
    ///
    /// Runtimes allocate one state value per session. Stateless modules (the
    /// default) return `Ok(None)`. Decoder models return their KV-cache layout
    /// and the runtime reserves capacity as the session grows.
    fn state_config(&self) -> Result<Option<crate::state::KvCacheConfig>> {
        Ok(None)
    }

    /// Build the full descriptor tree.
    fn descriptor(&self) -> Result<ModuleDescriptor> {
        let mut registry = ParameterRegistry::new();
        self.visit_parameters(&ModulePath::root(), &mut registry)?;
        let mut holder = ModuleDescriptor::new("root", ModulePath::root());
        self.describe(&ModulePath::root(), &mut holder);
        let mut descriptor = holder
            .children
            .into_iter()
            .next()
            .unwrap_or_else(|| ModuleDescriptor::new(self.module_type(), ModulePath::root()));
        attach_parameters(&mut descriptor, registry.specs());
        Ok(descriptor)
    }
}

fn attach_parameters(descriptor: &mut ModuleDescriptor, specs: &[crate::parameter::ParameterSpec]) {
    for spec in specs {
        if let Some(node) = descriptor.find_mut(&spec.module_path) {
            node.parameters.push(spec.name.clone());
        }
    }
}

/// Apply a function to every parameter tensor in a module tree.
pub fn for_each_parameter<M: Module + ?Sized>(
    module: &M,
    f: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
) -> Result<()> {
    module.visit_tensors(&ModulePath::root(), f)
}

// ---------------------------------------------------------------------------
// Blanket implementations for common containers
// ---------------------------------------------------------------------------

impl Module for () {
    fn module_type(&self) -> &'static str {
        "identity"
    }

    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, _cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        Ok(vec![input])
    }
}

impl<M: Module + ?Sized> Module for Box<M> {
    fn module_type(&self) -> &'static str {
        (**self).module_type()
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        (**self).visit_parameters(path, registry)
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        (**self).forward(cx, input)
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        (**self).forward_multi(cx, inputs)
    }

    fn set_training(&mut self, training: bool) {
        (**self).set_training(training);
    }

    fn is_training(&self) -> bool {
        (**self).is_training()
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        (**self).visit_tensors(path, visitor)
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        (**self).visit_tensors_mut(path, visitor)
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        (**self).describe(path, out)
    }
}

impl<M: Module> Module for Option<M> {
    fn module_type(&self) -> &'static str {
        "option"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        match self {
            Some(module) => module.visit_parameters(path, registry),
            None => Ok(()),
        }
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        match self {
            Some(module) => module.forward(cx, input),
            None => Err(Error::invalid_graph(format!(
                "optional module at {} is None",
                cx.current_path()
            ))),
        }
    }

    fn set_training(&mut self, training: bool) {
        if let Some(module) = self {
            module.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.as_ref().map(Module::is_training).unwrap_or(true)
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        match self {
            Some(module) => module.visit_tensors(path, visitor),
            None => Ok(()),
        }
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        match self {
            Some(module) => module.visit_tensors_mut(path, visitor),
            None => Ok(()),
        }
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        match self {
            Some(module) => module.describe(path, out),
            None => out
                .children
                .push(ModuleDescriptor::new("None", path.clone())),
        }
    }
}

impl<M: Module> Module for Vec<M> {
    fn module_type(&self) -> &'static str {
        "list"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        for (index, module) in self.iter().enumerate() {
            module.visit_parameters(&path.child(index.to_string()), registry)?;
        }
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let mut value = input;
        for (index, module) in self.iter().enumerate() {
            value = crate::context::module_forward(cx, &index.to_string(), module, value)?;
        }
        Ok(vec![value])
    }

    fn set_training(&mut self, training: bool) {
        for module in self.iter_mut() {
            module.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.first().map(Module::is_training).unwrap_or(true)
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        for (index, module) in self.iter().enumerate() {
            module.visit_tensors(&path.child(index.to_string()), visitor)?;
        }
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        for (index, module) in self.iter_mut().enumerate() {
            module.visit_tensors_mut(&path.child(index.to_string()), visitor)?;
        }
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        let mut node = ModuleDescriptor::new(self.module_type(), path.clone());
        for (index, module) in self.iter().enumerate() {
            module.describe(&path.child(index.to_string()), &mut node);
        }
        out.children.push(node);
    }
}

impl<M: Module + ?Sized> Module for Arc<M> {
    fn module_type(&self) -> &'static str {
        (**self).module_type()
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        (**self).visit_parameters(path, registry)
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        (**self).forward(cx, input)
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        (**self).forward_multi(cx, inputs)
    }

    fn is_training(&self) -> bool {
        (**self).is_training()
    }

    fn set_training(&mut self, _training: bool) {
        // A shared module keeps the mode chosen by its owning container.
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        (**self).visit_tensors(path, visitor)
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        if let Some(module) = Arc::get_mut(self) {
            module.visit_tensors_mut(path, visitor)
        } else {
            // Shared storage cannot be mutated through one alias. The parameter
            // store stays authoritative and write-back reports this tensor.
            Ok(())
        }
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        (**self).describe(path, out)
    }
}

/// A typed, indexable list of homogeneous modules.
///
/// `ModuleList<M>` is the model-API counterpart of PyTorch's `ModuleList`: it
/// owns a sequence of modules, assigns stable integer path segments (`0`, `1`,
/// ...) for parameter discovery, and can be iterated from a `#[model]` forward
/// pass. Unlike [`Vec<M>`], it is a distinct type, so a model field cannot be
/// confused with a plain data vector, and it carries collection conveniences
/// (`get`, `first`, `last`, `insert`, `remove`) that keep module identity
/// stable.
///
/// ```
/// use oxide_ml_model::{Module, ModuleList, ModulePath, ParameterRegistry};
/// # fn main() -> oxide_ml_core::Result<()> {
/// let layers: ModuleList<()> = ModuleList::from_vec(vec![(), ()]);
/// let mut registry = ParameterRegistry::new();
/// layers.visit_parameters(&ModulePath::root().child("layers"), &mut registry)?;
/// assert!(registry.specs().is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ModuleList<M> {
    modules: Vec<M>,
}

impl<M> ModuleList<M> {
    pub const fn new() -> Self {
        ModuleList {
            modules: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        ModuleList {
            modules: Vec::with_capacity(capacity),
        }
    }

    pub fn from_vec(modules: Vec<M>) -> Self {
        ModuleList { modules }
    }

    pub fn push(&mut self, module: M) {
        self.modules.push(module);
    }

    pub fn insert(&mut self, index: usize, module: M) {
        self.modules.insert(index, module);
    }

    /// Remove a module. Removing a module shifts the parameter paths of every
    /// later element, so do this before compiling or saving a checkpoint.
    pub fn remove(&mut self, index: usize) -> M {
        self.modules.remove(index)
    }

    pub fn pop(&mut self) -> Option<M> {
        self.modules.pop()
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&M> {
        self.modules.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut M> {
        self.modules.get_mut(index)
    }

    pub fn first(&self) -> Option<&M> {
        self.modules.first()
    }

    pub fn last(&self) -> Option<&M> {
        self.modules.last()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, M> {
        self.modules.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, M> {
        self.modules.iter_mut()
    }

    pub fn as_slice(&self) -> &[M] {
        &self.modules
    }

    pub fn as_mut_slice(&mut self) -> &mut [M] {
        &mut self.modules
    }

    pub fn into_vec(self) -> Vec<M> {
        self.modules
    }

    pub fn extend<I: IntoIterator<Item = M>>(&mut self, modules: I) {
        self.modules.extend(modules);
    }
}

impl<M> Default for ModuleList<M> {
    fn default() -> Self {
        ModuleList::new()
    }
}

impl<M> From<Vec<M>> for ModuleList<M> {
    fn from(modules: Vec<M>) -> Self {
        ModuleList::from_vec(modules)
    }
}

impl<M> FromIterator<M> for ModuleList<M> {
    fn from_iter<I: IntoIterator<Item = M>>(iter: I) -> Self {
        ModuleList::from_vec(iter.into_iter().collect())
    }
}

impl<M> IntoIterator for ModuleList<M> {
    type Item = M;
    type IntoIter = std::vec::IntoIter<M>;

    fn into_iter(self) -> Self::IntoIter {
        self.modules.into_iter()
    }
}

impl<'a, M> IntoIterator for &'a ModuleList<M> {
    type Item = &'a M;
    type IntoIter = std::slice::Iter<'a, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.modules.iter()
    }
}

impl<'a, M> IntoIterator for &'a mut ModuleList<M> {
    type Item = &'a mut M;
    type IntoIter = std::slice::IterMut<'a, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.modules.iter_mut()
    }
}

impl<M> std::ops::Index<usize> for ModuleList<M> {
    type Output = M;

    fn index(&self, index: usize) -> &M {
        &self.modules[index]
    }
}

impl<M> std::ops::IndexMut<usize> for ModuleList<M> {
    fn index_mut(&mut self, index: usize) -> &mut M {
        &mut self.modules[index]
    }
}

impl<M> std::fmt::Debug for ModuleList<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModuleList")
            .field("len", &self.modules.len())
            .finish()
    }
}

impl<M: Module> Module for ModuleList<M> {
    fn module_type(&self) -> &'static str {
        "ModuleList"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        for (index, module) in self.modules.iter().enumerate() {
            module.visit_parameters(&path.child(index.to_string()), registry)?;
        }
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let mut value = input;
        for (index, module) in self.modules.iter().enumerate() {
            value = crate::context::module_forward(cx, &index.to_string(), module, value)?;
        }
        Ok(vec![value])
    }

    fn set_training(&mut self, training: bool) {
        for module in self.modules.iter_mut() {
            module.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.modules
            .first()
            .map(Module::is_training)
            .unwrap_or(true)
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        for (index, module) in self.modules.iter().enumerate() {
            module.visit_tensors(&path.child(index.to_string()), visitor)?;
        }
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        for (index, module) in self.modules.iter_mut().enumerate() {
            module.visit_tensors_mut(&path.child(index.to_string()), visitor)?;
        }
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        let mut node = ModuleDescriptor::new(self.module_type(), path.clone());
        for (index, module) in self.modules.iter().enumerate() {
            module.describe(&path.child(index.to_string()), &mut node);
        }
        out.children.push(node);
    }
}

/// A linear stack of modules (`Sequential`).
pub struct Sequential {
    modules: Vec<Box<dyn Module>>,
    training: bool,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Sequential {
            modules: Vec::new(),
            training: true,
        }
    }

    /// Append a module and return `self` for builder-style construction.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    /// Append a module in place.
    pub fn push<M: Module + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn modules(&self) -> &[Box<dyn Module>] {
        &self.modules
    }

    /// Run a one-shot forward pass on host tensors.
    pub fn predict(&self, input: crate::tensor::Tensor<f32>) -> Result<crate::tensor::Tensor<f32>> {
        crate::runtime::predict_single(self, input)
    }
}

impl Module for Sequential {
    fn module_type(&self) -> &'static str {
        "Sequential"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        for (index, module) in self.modules.iter().enumerate() {
            module.visit_parameters(&path.child(index.to_string()), registry)?;
        }
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let mut value = input;
        for (index, module) in self.modules.iter().enumerate() {
            value = crate::context::module_forward(cx, &index.to_string(), module.as_ref(), value)?;
        }
        Ok(vec![value])
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for module in self.modules.iter_mut() {
            module.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        for (index, module) in self.modules.iter().enumerate() {
            module.visit_tensors(&path.child(index.to_string()), visitor)?;
        }
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        for (index, module) in self.modules.iter_mut().enumerate() {
            module.visit_tensors_mut(&path.child(index.to_string()), visitor)?;
        }
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        let mut node = ModuleDescriptor::new(self.module_type(), path.clone());
        for (index, module) in self.modules.iter().enumerate() {
            module.describe(&path.child(index.to_string()), &mut node);
        }
        out.children.push(node);
    }
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("modules", &self.modules.len())
            .field("training", &self.training)
            .finish()
    }
}
