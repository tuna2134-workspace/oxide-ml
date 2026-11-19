//! Graph-independent neural-network modules.
//!
//! Modules own their parameters as concrete `DenseTensor<f32>` values and are
//! replayed into a graph through [`oxide_ml_model::ModelContext`]. Parameter
//! names are stable module paths (`fc1.weight`, `norm.gamma`, ...).
//!
//! Everything here is F32. The low-level graph API remains generic over
//! `Scalar`; the model API is deliberately F32-only in this milestone.

use std::sync::atomic::{AtomicU64, Ordering};

use oxide_ml_core::{DenseTensor, Error, Result, Rng, Shape, ValueId};
use oxide_ml_graph::{ActivationKind, Conv2dConfig, Pool2dConfig};
use oxide_ml_model::{
    ModelContext, Module, ModuleDescriptor, ModulePath, ParameterRegistry, ParameterRole,
};

static INIT_COUNTER: AtomicU64 = AtomicU64::new(0x0BAD_5EED_1234_5678);

fn next_seed() -> u64 {
    INIT_COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
}

fn zeros(dims: impl AsRef<[usize]>) -> DenseTensor<f32> {
    DenseTensor::zeros(dims)
}

fn ones(dims: impl AsRef<[usize]>) -> DenseTensor<f32> {
    DenseTensor::ones(dims)
}

fn xavier(rows: usize, cols: usize, seed: u64) -> DenseTensor<f32> {
    let mut rng = Rng::new(seed);
    let bound = (6.0 / (rows + cols) as f32).sqrt();
    let values = (0..rows * cols)
        .map(|_| rng.uniform(-bound, bound))
        .collect();
    DenseTensor::from_vec([rows, cols], values).expect("shape and data agree")
}

fn linear_parameters(
    in_features: usize,
    out_features: usize,
    bias: bool,
    seed: u64,
) -> (DenseTensor<f32>, Option<DenseTensor<f32>>) {
    let weight = xavier(out_features, in_features, seed);
    let bias = bias.then(|| zeros([out_features]));
    (weight, bias)
}

fn visit<'a>(
    path: &ModulePath,
    name: &str,
    tensor: &'a DenseTensor<f32>,
    role: ParameterRole,
    visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
) -> Result<()> {
    visitor(&path.child(name), tensor, role)
}

fn visit_mut<'a>(
    path: &ModulePath,
    name: &str,
    tensor: &'a mut DenseTensor<f32>,
    role: ParameterRole,
    visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
) -> Result<()> {
    visitor(&path.child(name), tensor, role)
}

fn register(
    path: &ModulePath,
    name: &str,
    tensor: &DenseTensor<f32>,
    role: ParameterRole,
    registry: &mut ParameterRegistry,
) -> Result<()> {
    registry.insert_tensor(&path.child(name), tensor, role)
}

fn check_last_dim(cx: &ModelContext, value: ValueId, expected: usize, context: &str) -> Result<()> {
    let shape = cx.value_shape(value)?;
    if shape.rank() < 2 {
        return Err(Error::RankMismatch {
            context: context.to_string(),
            expected: 2,
            got: shape.rank(),
        });
    }
    let actual = shape
        .dim(shape.rank() - 1)
        .and_then(oxide_ml_core::Dim::static_usize);
    if actual != Some(expected) {
        return Err(Error::shape_mismatch(
            context,
            Shape::from_static([expected]),
            shape,
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// Fully connected layer: `y = x W^T + b`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: DenseTensor<f32>,
    pub bias: Option<DenseTensor<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_config(in_features, out_features, true)
    }

    pub fn with_config(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self::with_seed(in_features, out_features, bias, next_seed())
    }

    pub fn with_seed(in_features: usize, out_features: usize, bias: bool, seed: u64) -> Self {
        let (weight, bias) = linear_parameters(in_features, out_features, bias, seed);
        Linear {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn from_tensors(weight: DenseTensor<f32>, bias: Option<DenseTensor<f32>>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(Error::RankMismatch {
                context: "Linear::from_tensors".into(),
                expected: 2,
                got: dims.len(),
            });
        }
        if let Some(bias) = &bias
            && bias.dims() != [dims[0]]
        {
            return Err(Error::shape_mismatch(
                "Linear bias",
                Shape::from_static([dims[0]]),
                bias.shape().clone(),
            ));
        }
        Ok(Linear {
            in_features: dims[1],
            out_features: dims[0],
            weight,
            bias,
        })
    }
}

impl Module for Linear {
    fn module_type(&self) -> &'static str {
        "Linear"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        register(
            path,
            "weight",
            &self.weight,
            ParameterRole::Trainable,
            registry,
        )?;
        if let Some(bias) = &self.bias {
            register(path, "bias", bias, ParameterRole::Trainable, registry)?;
        }
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        check_last_dim(cx, input, self.in_features, "Linear input")?;
        let weight = cx.parameter("weight")?;
        let output = cx.builder().matmul_with(input, weight, false, true)?;
        let output = match self.bias {
            Some(_) => {
                let bias = cx.parameter("bias")?;
                cx.builder().add(output, bias)?
            }
            None => output,
        };
        Ok(vec![output])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit(
            path,
            "weight",
            &self.weight,
            ParameterRole::Trainable,
            visitor,
        )?;
        if let Some(bias) = &self.bias {
            visit(path, "bias", bias, ParameterRole::Trainable, visitor)?;
        }
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit_mut(
            path,
            "weight",
            &mut self.weight,
            ParameterRole::Trainable,
            visitor,
        )?;
        if let Some(bias) = &mut self.bias {
            visit_mut(path, "bias", bias, ParameterRole::Trainable, visitor)?;
        }
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!(
                "Linear(in_features={}, out_features={}, bias={})",
                self.in_features,
                self.out_features,
                self.bias.is_some()
            ),
            path.clone(),
        ));
    }
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

macro_rules! activation_module {
    ($name:ident, $kind:expr, $label:literal) => {
        /// Elementwise activation module.
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl $name {
            pub fn new() -> Self {
                $name
            }
        }

        impl Module for $name {
            fn module_type(&self) -> &'static str {
                $label
            }

            fn visit_parameters(
                &self,
                _path: &ModulePath,
                _registry: &mut ParameterRegistry,
            ) -> Result<()> {
                Ok(())
            }

            fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
                Ok(vec![cx.builder().activation($kind, input)?])
            }

            fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
                out.children
                    .push(ModuleDescriptor::new($label, path.clone()));
            }
        }
    };
}

activation_module!(ReLU, ActivationKind::ReLU, "ReLU");
activation_module!(GELU, ActivationKind::GELU, "GELU");
activation_module!(Sigmoid, ActivationKind::Sigmoid, "Sigmoid");
activation_module!(Tanh, ActivationKind::Tanh, "Tanh");

/// Softmax over one axis.
#[derive(Clone, Debug)]
pub struct Softmax {
    pub axis: i32,
}

impl Softmax {
    pub fn new(axis: i32) -> Self {
        Softmax { axis }
    }

    /// Softmax over the last axis.
    pub fn last() -> Self {
        Softmax { axis: -1 }
    }
}

impl Module for Softmax {
    fn module_type(&self) -> &'static str {
        "Softmax"
    }

    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        Ok(vec![cx.builder().softmax(input, self.axis)?])
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!("Softmax(axis={})", self.axis),
            path.clone(),
        ));
    }
}

// ---------------------------------------------------------------------------
// Dropout
// ---------------------------------------------------------------------------

/// Inverted dropout.
///
/// The graph IR has no runtime random source in this milestone, so a stochastic
/// mask cannot be produced without rebuilding the graph per step. `Dropout` in
/// training mode therefore returns an explicit error instead of silently
/// applying a fixed or missing mask. In eval mode it is the identity.
#[derive(Clone, Debug)]
pub struct Dropout {
    pub p: f32,
    training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Result<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(Error::invalid_shape("Dropout", "p must be in [0, 1)"));
        }
        Ok(Dropout { p, training: true })
    }

    pub fn p(&self) -> f32 {
        self.p
    }
}

impl Module for Dropout {
    fn module_type(&self) -> &'static str {
        "Dropout"
    }

    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        if cx.is_training() && self.training && self.p > 0.0 {
            return Err(Error::unsupported(format!(
                "Dropout(p={}) requires a per-step random mask, which the graph IR does not \
                 expose yet; use p=0, eval mode, or rebuild the graph per step",
                self.p
            )));
        }
        Ok(vec![input])
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!("Dropout(p={})", self.p),
            path.clone(),
        ));
    }
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Layer normalization over trailing dimensions.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub gamma: DenseTensor<f32>,
    pub beta: DenseTensor<f32>,
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize) -> Self {
        Self::with_shape(vec![normalized_shape])
    }

    pub fn with_shape(normalized_shape: Vec<usize>) -> Self {
        Self::with_eps(normalized_shape, 1e-5)
    }

    pub fn with_eps(normalized_shape: Vec<usize>, eps: f32) -> Self {
        LayerNorm {
            gamma: ones(&normalized_shape),
            beta: zeros(&normalized_shape),
            normalized_shape,
            eps,
        }
    }
}

impl Module for LayerNorm {
    fn module_type(&self) -> &'static str {
        "LayerNorm"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        register(
            path,
            "gamma",
            &self.gamma,
            ParameterRole::Trainable,
            registry,
        )?;
        register(path, "beta", &self.beta, ParameterRole::Trainable, registry)?;
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let gamma = cx.parameter("gamma")?;
        let beta = cx.parameter("beta")?;
        Ok(vec![cx.builder().layer_norm(
            input,
            gamma,
            beta,
            self.normalized_shape.clone(),
            self.eps,
        )?])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit(
            path,
            "gamma",
            &self.gamma,
            ParameterRole::Trainable,
            visitor,
        )?;
        visit(path, "beta", &self.beta, ParameterRole::Trainable, visitor)?;
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit_mut(
            path,
            "gamma",
            &mut self.gamma,
            ParameterRole::Trainable,
            visitor,
        )?;
        visit_mut(
            path,
            "beta",
            &mut self.beta,
            ParameterRole::Trainable,
            visitor,
        )?;
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!(
                "LayerNorm(normalized_shape={:?}, eps={})",
                self.normalized_shape, self.eps
            ),
            path.clone(),
        ));
    }
}

/// Batch normalization over `[N, C, ...]`.
///
/// Training normalizes with per-batch statistics. Eval mode uses the running
/// statistics when `track_running_stats` is enabled. Running statistics are not
/// updated by this milestone (there is no in-graph state write); they are saved
/// and restored by checkpoints, so they can be supplied externally.
#[derive(Clone, Debug)]
pub struct BatchNorm {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub gamma: Option<DenseTensor<f32>>,
    pub beta: Option<DenseTensor<f32>>,
    pub running_mean: Option<DenseTensor<f32>>,
    pub running_var: Option<DenseTensor<f32>>,
    training: bool,
}

impl BatchNorm {
    pub fn new(num_features: usize) -> Self {
        BatchNorm {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            gamma: Some(ones([num_features])),
            beta: Some(zeros([num_features])),
            running_mean: Some(zeros([num_features])),
            running_var: Some(ones([num_features])),
            training: true,
        }
    }

    pub fn without_running_stats(mut self) -> Self {
        self.running_mean = None;
        self.running_var = None;
        self
    }
}

impl Module for BatchNorm {
    fn module_type(&self) -> &'static str {
        "BatchNorm"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        if let Some(gamma) = &self.gamma {
            register(path, "gamma", gamma, ParameterRole::Trainable, registry)?;
        }
        if let Some(beta) = &self.beta {
            register(path, "beta", beta, ParameterRole::Trainable, registry)?;
        }
        if let Some(mean) = &self.running_mean {
            register(path, "running_mean", mean, ParameterRole::Buffer, registry)?;
        }
        if let Some(var) = &self.running_var {
            register(path, "running_var", var, ParameterRole::Buffer, registry)?;
        }
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let shape = cx.value_shape(input)?;
        let rank = shape.rank();
        if rank < 2 {
            return Err(Error::RankMismatch {
                context: "BatchNorm input".into(),
                expected: 2,
                got: rank,
            });
        }
        if shape.dim(1).and_then(oxide_ml_core::Dim::static_usize) != Some(self.num_features) {
            return Err(Error::shape_mismatch(
                "BatchNorm channels",
                Shape::from_static([self.num_features]),
                shape.clone(),
            ));
        }
        let reduce_axes: Vec<usize> = std::iter::once(0).chain(2..rank).collect();
        let element_count = reduce_axes
            .iter()
            .filter_map(|&axis| shape.dim(axis).and_then(oxide_ml_core::Dim::static_usize))
            .product::<usize>()
            .max(1);

        let use_running =
            !cx.is_training() && self.running_mean.is_some() && self.running_var.is_some();
        let mut broadcast = vec![1usize; rank];
        broadcast[1] = self.num_features;

        let (mean, variance) = if use_running {
            let mean = cx.parameter("running_mean")?;
            let var = cx.parameter("running_var")?;
            let mean = cx.builder().reshape(mean, Shape::from_static(&broadcast))?;
            let var = cx.builder().reshape(var, Shape::from_static(&broadcast))?;
            (mean, var)
        } else {
            let mean = cx.builder().reduce_mean(input, reduce_axes.clone(), true)?;
            // E[x^2] - E[x]^2, with a correction to the unbiased estimator.
            let square = cx.builder().mul(input, input)?;
            let mean_square = cx.builder().reduce_mean(square, reduce_axes, true)?;
            let mean_squared = cx.builder().mul(mean, mean)?;
            let variance = cx.builder().sub(mean_square, mean_squared)?;
            let correction = if element_count > 1 {
                (element_count as f32) / ((element_count - 1) as f32)
            } else {
                1.0
            };
            let correction = cx.constant_scalar(correction);
            let variance = cx.builder().mul(variance, correction)?;
            (mean, variance)
        };

        let eps = cx.constant_scalar(self.eps);
        let variance = cx.builder().add(variance, eps)?;
        let std = cx.builder().sqrt(variance)?;
        let centered = cx.builder().sub(input, mean)?;
        let normalized = cx.builder().div(centered, std)?;
        let mut output = normalized;
        if self.gamma.is_some() {
            let gamma = cx.parameter("gamma")?;
            let gamma = cx
                .builder()
                .reshape(gamma, Shape::from_static(&broadcast))?;
            output = cx.builder().mul(output, gamma)?;
        }
        if self.beta.is_some() {
            let beta = cx.parameter("beta")?;
            let beta = cx.builder().reshape(beta, Shape::from_static(&broadcast))?;
            output = cx.builder().add(output, beta)?;
        }
        Ok(vec![output])
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        if let Some(gamma) = &self.gamma {
            visit(path, "gamma", gamma, ParameterRole::Trainable, visitor)?;
        }
        if let Some(beta) = &self.beta {
            visit(path, "beta", beta, ParameterRole::Trainable, visitor)?;
        }
        if let Some(mean) = &self.running_mean {
            visit(path, "running_mean", mean, ParameterRole::Buffer, visitor)?;
        }
        if let Some(var) = &self.running_var {
            visit(path, "running_var", var, ParameterRole::Buffer, visitor)?;
        }
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        if let Some(gamma) = &mut self.gamma {
            visit_mut(path, "gamma", gamma, ParameterRole::Trainable, visitor)?;
        }
        if let Some(beta) = &mut self.beta {
            visit_mut(path, "beta", beta, ParameterRole::Trainable, visitor)?;
        }
        if let Some(mean) = &mut self.running_mean {
            visit_mut(path, "running_mean", mean, ParameterRole::Buffer, visitor)?;
        }
        if let Some(var) = &mut self.running_var {
            visit_mut(path, "running_var", var, ParameterRole::Buffer, visitor)?;
        }
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!("BatchNorm(num_features={})", self.num_features),
            path.clone(),
        ));
    }
}

/// Group normalization over `[N, C, ...]`.
#[derive(Clone, Debug)]
pub struct GroupNorm {
    pub num_groups: usize,
    pub num_channels: usize,
    pub eps: f32,
    pub gamma: Option<DenseTensor<f32>>,
    pub beta: Option<DenseTensor<f32>>,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize) -> Result<Self> {
        if num_groups == 0 || num_channels % num_groups != 0 {
            return Err(Error::invalid_shape(
                "GroupNorm",
                format!("num_channels {num_channels} must be divisible by num_groups {num_groups}"),
            ));
        }
        Ok(GroupNorm {
            num_groups,
            num_channels,
            eps: 1e-5,
            gamma: Some(ones([num_channels])),
            beta: Some(zeros([num_channels])),
        })
    }

    pub fn without_affine(mut self) -> Self {
        self.gamma = None;
        self.beta = None;
        self
    }

    fn forward_impl(&self, cx: &mut ModelContext, input: ValueId) -> Result<ValueId> {
        let shape = cx.value_shape(input)?;
        let rank = shape.rank();
        if rank < 2 {
            return Err(Error::RankMismatch {
                context: "GroupNorm input".into(),
                expected: 2,
                got: rank,
            });
        }
        if shape.dim(1).and_then(oxide_ml_core::Dim::static_usize) != Some(self.num_channels) {
            return Err(Error::shape_mismatch(
                "GroupNorm channels",
                Shape::from_static([self.num_channels]),
                shape.clone(),
            ));
        }
        let per_group = self.num_channels / self.num_groups;
        let mut grouped_dims = Vec::with_capacity(rank + 1);
        grouped_dims.push(shape.dim(0).unwrap_or(oxide_ml_core::Dim::Dynamic));
        grouped_dims.extend([
            oxide_ml_core::Dim::Static(self.num_groups),
            oxide_ml_core::Dim::Static(per_group),
        ]);
        for axis in 2..rank {
            grouped_dims.push(shape.dim(axis).unwrap_or(oxide_ml_core::Dim::Dynamic));
        }
        let grouped = cx
            .builder()
            .reshape(input, Shape::new(grouped_dims.clone()))?;
        let reduce_axes: Vec<usize> = (2..grouped_dims.len()).collect();
        let mean = cx
            .builder()
            .reduce_mean(grouped, reduce_axes.clone(), true)?;
        let square = cx.builder().mul(grouped, grouped)?;
        let mean_square = cx.builder().reduce_mean(square, reduce_axes, true)?;
        let mean_squared = cx.builder().mul(mean, mean)?;
        let variance = cx.builder().sub(mean_square, mean_squared)?;
        let eps = cx.constant_scalar(self.eps);
        let variance = cx.builder().add(variance, eps)?;
        let std = cx.builder().sqrt(variance)?;
        let centered = cx.builder().sub(grouped, mean)?;
        let normalized = cx.builder().div(centered, std)?;
        let mut output = cx.builder().reshape(normalized, shape.clone())?;

        let mut broadcast = vec![1usize; rank];
        broadcast[1] = self.num_channels;
        if self.gamma.is_some() {
            let gamma = cx.parameter("gamma")?;
            let gamma = cx
                .builder()
                .reshape(gamma, Shape::from_static(&broadcast))?;
            output = cx.builder().mul(output, gamma)?;
        }
        if self.beta.is_some() {
            let beta = cx.parameter("beta")?;
            let beta = cx.builder().reshape(beta, Shape::from_static(&broadcast))?;
            output = cx.builder().add(output, beta)?;
        }
        Ok(output)
    }
}

impl Module for GroupNorm {
    fn module_type(&self) -> &'static str {
        "GroupNorm"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        if let Some(gamma) = &self.gamma {
            register(path, "gamma", gamma, ParameterRole::Trainable, registry)?;
        }
        if let Some(beta) = &self.beta {
            register(path, "beta", beta, ParameterRole::Trainable, registry)?;
        }
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        Ok(vec![self.forward_impl(cx, input)?])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        if let Some(gamma) = &self.gamma {
            visit(path, "gamma", gamma, ParameterRole::Trainable, visitor)?;
        }
        if let Some(beta) = &self.beta {
            visit(path, "beta", beta, ParameterRole::Trainable, visitor)?;
        }
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        if let Some(gamma) = &mut self.gamma {
            visit_mut(path, "gamma", gamma, ParameterRole::Trainable, visitor)?;
        }
        if let Some(beta) = &mut self.beta {
            visit_mut(path, "beta", beta, ParameterRole::Trainable, visitor)?;
        }
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!(
                "GroupNorm(num_groups={}, num_channels={})",
                self.num_groups, self.num_channels
            ),
            path.clone(),
        ));
    }
}

/// Instance normalization (`GroupNorm` with one group per channel).
pub type InstanceNorm = GroupNormAdapter;

/// Adapter type so `InstanceNorm::new(channels)` has the expected signature.
#[derive(Clone, Debug)]
pub struct GroupNormAdapter {
    inner: GroupNorm,
}

impl GroupNormAdapter {
    pub fn new(num_channels: usize) -> Result<Self> {
        Ok(GroupNormAdapter {
            inner: GroupNorm::new(num_channels, num_channels)?,
        })
    }

    pub fn without_affine(mut self) -> Self {
        self.inner = self.inner.without_affine();
        self
    }
}

impl Module for GroupNormAdapter {
    fn module_type(&self) -> &'static str {
        "InstanceNorm"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.inner.visit_parameters(path, registry)
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        self.inner.forward(cx, input)
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.inner.visit_tensors(path, visitor)
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.inner.visit_tensors_mut(path, visitor)
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        self.inner.describe(path, out);
    }
}

// ---------------------------------------------------------------------------
// Convolution and pooling
// ---------------------------------------------------------------------------

/// 2-D convolution over NCHW with OIHW weights.
#[derive(Clone, Debug)]
pub struct Conv2d {
    pub weight: DenseTensor<f32>,
    pub bias: Option<DenseTensor<f32>>,
    pub config: Conv2dConfig,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
    ) -> Self {
        Self::with_config(
            in_channels,
            out_channels,
            kernel,
            Conv2dConfig {
                stride,
                padding,
                dilation: [1, 1],
                groups: 1,
            },
        )
    }

    pub fn with_config(
        in_channels: usize,
        out_channels: usize,
        kernel: [usize; 2],
        config: Conv2dConfig,
    ) -> Self {
        let fan_in = in_channels * kernel[0] * kernel[1];
        let fan_out = out_channels * kernel[0] * kernel[1];
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let mut rng = Rng::new(next_seed());
        let values = (0..out_channels * in_channels * kernel[0] * kernel[1])
            .map(|_| rng.uniform(-bound, bound))
            .collect();
        let weight =
            DenseTensor::from_vec([out_channels, in_channels, kernel[0], kernel[1]], values)
                .expect("shape and data agree");
        Conv2d {
            weight,
            bias: Some(zeros([out_channels])),
            config,
            in_channels,
            out_channels,
        }
    }

    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }
}

impl Module for Conv2d {
    fn module_type(&self) -> &'static str {
        "Conv2d"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        register(
            path,
            "weight",
            &self.weight,
            ParameterRole::Trainable,
            registry,
        )?;
        if let Some(bias) = &self.bias {
            register(path, "bias", bias, ParameterRole::Trainable, registry)?;
        }
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let weight = cx.parameter("weight")?;
        let bias = if self.bias.is_some() {
            Some(cx.parameter("bias")?)
        } else {
            None
        };
        Ok(vec![cx.builder().conv2d(
            input,
            weight,
            bias,
            self.config,
        )?])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit(
            path,
            "weight",
            &self.weight,
            ParameterRole::Trainable,
            visitor,
        )?;
        if let Some(bias) = &self.bias {
            visit(path, "bias", bias, ParameterRole::Trainable, visitor)?;
        }
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit_mut(
            path,
            "weight",
            &mut self.weight,
            ParameterRole::Trainable,
            visitor,
        )?;
        if let Some(bias) = &mut self.bias {
            visit_mut(path, "bias", bias, ParameterRole::Trainable, visitor)?;
        }
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!(
                "Conv2d(in_channels={}, out_channels={}, kernel={:?})",
                self.in_channels,
                self.out_channels,
                [self.weight.dims()[2], self.weight.dims()[3]]
            ),
            path.clone(),
        ));
    }
}

/// 1-D convolution implemented as a `[N, C, 1, L]` 2-D convolution.
#[derive(Clone, Debug)]
pub struct Conv1d {
    pub weight: DenseTensor<f32>,
    pub bias: Option<DenseTensor<f32>>,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let fan_in = in_channels * kernel;
        let fan_out = out_channels * kernel;
        let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
        let mut rng = Rng::new(next_seed());
        let values = (0..out_channels * in_channels * kernel)
            .map(|_| rng.uniform(-bound, bound))
            .collect();
        Conv1d {
            weight: DenseTensor::from_vec([out_channels, in_channels, kernel], values)
                .expect("shape and data agree"),
            bias: Some(zeros([out_channels])),
            stride,
            padding,
            dilation: 1,
            in_channels,
            out_channels,
        }
    }

    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }
}

impl Module for Conv1d {
    fn module_type(&self) -> &'static str {
        "Conv1d"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        register(
            path,
            "weight",
            &self.weight,
            ParameterRole::Trainable,
            registry,
        )?;
        if let Some(bias) = &self.bias {
            register(path, "bias", bias, ParameterRole::Trainable, registry)?;
        }
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let shape = cx.value_shape(input)?;
        if shape.rank() != 3 {
            return Err(Error::RankMismatch {
                context: "Conv1d input".into(),
                expected: 3,
                got: shape.rank(),
            });
        }
        let batch = shape.dim(0).unwrap_or(oxide_ml_core::Dim::Dynamic);
        let channels = shape.dim(1).unwrap_or(oxide_ml_core::Dim::Dynamic);
        let length = shape
            .dim(2)
            .and_then(oxide_ml_core::Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("Conv1d", "sequence length must be static"))?;
        let kernel = self.weight.dims()[2];
        let out_length = (length + 2 * self.padding)
            .saturating_sub(self.dilation * (kernel - 1) + 1)
            / self.stride
            + 1;
        let reshaped = cx.builder().reshape(
            input,
            Shape::new(vec![
                batch,
                channels,
                oxide_ml_core::Dim::Static(1),
                oxide_ml_core::Dim::Static(length),
            ]),
        )?;
        let weight = cx.parameter("weight")?;
        let weight = cx.builder().reshape(
            weight,
            Shape::from_static([self.out_channels, self.in_channels, 1, kernel]),
        )?;
        let bias = if self.bias.is_some() {
            Some(cx.parameter("bias")?)
        } else {
            None
        };
        let config = Conv2dConfig {
            stride: [1, self.stride],
            padding: [0, self.padding],
            dilation: [1, self.dilation],
            groups: 1,
        };
        let output = cx.builder().conv2d(reshaped, weight, bias, config)?;
        let output = cx.builder().reshape(
            output,
            Shape::new(vec![
                batch,
                oxide_ml_core::Dim::Static(self.out_channels),
                oxide_ml_core::Dim::Static(out_length),
            ]),
        )?;
        Ok(vec![output])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit(
            path,
            "weight",
            &self.weight,
            ParameterRole::Trainable,
            visitor,
        )?;
        if let Some(bias) = &self.bias {
            visit(path, "bias", bias, ParameterRole::Trainable, visitor)?;
        }
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit_mut(
            path,
            "weight",
            &mut self.weight,
            ParameterRole::Trainable,
            visitor,
        )?;
        if let Some(bias) = &mut self.bias {
            visit_mut(path, "bias", bias, ParameterRole::Trainable, visitor)?;
        }
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!(
                "Conv1d(in_channels={}, out_channels={}, kernel={})",
                self.in_channels,
                self.out_channels,
                self.weight.dims()[2]
            ),
            path.clone(),
        ));
    }
}

/// 3-D convolution is not implemented in this milestone.
#[derive(Clone, Debug)]
pub struct Conv3d;

impl Conv3d {
    pub fn new(_in_channels: usize, _out_channels: usize, _kernel: [usize; 3]) -> Self {
        Conv3d
    }
}

impl Module for Conv3d {
    fn module_type(&self) -> &'static str {
        "Conv3d"
    }

    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, _: &mut ModelContext, _: ValueId) -> Result<Vec<ValueId>> {
        Err(Error::unsupported(
            "Conv3d is not implemented (the IR has no 3-D convolution op); \
             use Conv2d over frames or add the op to the graph stack",
        ))
    }
}

macro_rules! pool2d_module {
    ($name:ident, $method:ident, $label:literal) => {
        /// 2-D pooling over NCHW.
        #[derive(Clone, Debug)]
        pub struct $name {
            pub config: Pool2dConfig,
        }

        impl $name {
            pub fn new(kernel: [usize; 2], stride: [usize; 2]) -> Self {
                $name {
                    config: Pool2dConfig::new(kernel, stride),
                }
            }

            pub fn with_padding(mut self, padding: [usize; 2]) -> Self {
                self.config.padding = padding;
                self
            }
        }

        impl Module for $name {
            fn module_type(&self) -> &'static str {
                $label
            }

            fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
                Ok(())
            }

            fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
                Ok(vec![cx.builder().$method(input, self.config)?])
            }

            fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
                out.children.push(ModuleDescriptor::new(
                    format!(
                        "{}(kernel={:?}, stride={:?})",
                        $label, self.config.kernel, self.config.stride
                    ),
                    path.clone(),
                ));
            }
        }
    };
}

pool2d_module!(MaxPool2d, max_pool2d, "MaxPool2d");
pool2d_module!(AvgPool2d, avg_pool2d, "AvgPool2d");

/// 1-D max pooling implemented as `[N, C, 1, L]` 2-D pooling.
#[derive(Clone, Debug)]
pub struct MaxPool1d {
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
}

impl MaxPool1d {
    pub fn new(kernel: usize, stride: usize) -> Self {
        MaxPool1d {
            kernel,
            stride,
            padding: 0,
        }
    }
}

impl Module for MaxPool1d {
    fn module_type(&self) -> &'static str {
        "MaxPool1d"
    }

    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let shape = cx.value_shape(input)?;
        if shape.rank() != 3 {
            return Err(Error::RankMismatch {
                context: "MaxPool1d input".into(),
                expected: 3,
                got: shape.rank(),
            });
        }
        let batch = shape.dim(0).unwrap_or(oxide_ml_core::Dim::Dynamic);
        let channels = shape.dim(1).unwrap_or(oxide_ml_core::Dim::Dynamic);
        let length = shape
            .dim(2)
            .and_then(oxide_ml_core::Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("MaxPool1d", "length must be static"))?;
        let reshaped = cx.builder().reshape(
            input,
            Shape::new(vec![
                batch,
                channels,
                oxide_ml_core::Dim::Static(1),
                oxide_ml_core::Dim::Static(length),
            ]),
        )?;
        let config = Pool2dConfig {
            kernel: [1, self.kernel],
            stride: [1, self.stride],
            padding: [0, self.padding],
            dilation: [1, 1],
        };
        let pooled = cx.builder().max_pool2d(reshaped, config)?;
        let out_length = (length + 2 * self.padding).saturating_sub(self.kernel) / self.stride + 1;
        let output = cx.builder().reshape(
            pooled,
            Shape::new(vec![
                batch,
                channels,
                oxide_ml_core::Dim::Static(out_length),
            ]),
        )?;
        Ok(vec![output])
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!(
                "MaxPool1d(kernel={}, stride={}, padding={})",
                self.kernel, self.stride, self.padding
            ),
            path.clone(),
        ));
    }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Lookup table over integral indices stored in an F32 tensor.
#[derive(Clone, Debug)]
pub struct Embedding {
    pub weight: DenseTensor<f32>,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let mut rng = Rng::new(next_seed());
        let values = (0..num_embeddings * embedding_dim)
            .map(|_| rng.normal() * 0.02)
            .collect();
        Embedding {
            weight: DenseTensor::from_vec([num_embeddings, embedding_dim], values)
                .expect("shape and data agree"),
            num_embeddings,
            embedding_dim,
        }
    }

    pub fn with_seed(num_embeddings: usize, embedding_dim: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let values = (0..num_embeddings * embedding_dim)
            .map(|_| rng.normal() * 0.02)
            .collect();
        Embedding {
            weight: DenseTensor::from_vec([num_embeddings, embedding_dim], values)
                .expect("shape and data agree"),
            num_embeddings,
            embedding_dim,
        }
    }
}

impl Module for Embedding {
    fn module_type(&self) -> &'static str {
        "Embedding"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        register(
            path,
            "weight",
            &self.weight,
            ParameterRole::Trainable,
            registry,
        )
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let weight = cx.parameter("weight")?;
        Ok(vec![cx.builder().embedding(weight, input)?])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit(
            path,
            "weight",
            &self.weight,
            ParameterRole::Trainable,
            visitor,
        )
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visit_mut(
            path,
            "weight",
            &mut self.weight,
            ParameterRole::Trainable,
            visitor,
        )
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!(
                "Embedding(num_embeddings={}, embedding_dim={})",
                self.num_embeddings, self.embedding_dim
            ),
            path.clone(),
        ));
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Scaled dot-product attention over three rank-3 inputs `[B, T, D]`.
#[derive(Clone, Debug)]
pub struct Attention {
    pub scale: Option<f32>,
    pub causal: bool,
}

impl Attention {
    pub fn new() -> Self {
        Attention {
            scale: None,
            causal: false,
        }
    }

    pub fn causal() -> Self {
        Attention {
            scale: None,
            causal: true,
        }
    }

    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = Some(scale);
        self
    }
}

impl Default for Attention {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Attention {
    fn module_type(&self) -> &'static str {
        "Attention"
    }

    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, _: &mut ModelContext, _: ValueId) -> Result<Vec<ValueId>> {
        Err(Error::invalid_graph(
            "Attention requires 3 inputs (query, key, value); call forward_multi",
        ))
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        if inputs.len() != 3 {
            return Err(Error::invalid_graph(format!(
                "Attention expects 3 inputs, got {}",
                inputs.len()
            )));
        }
        let (query, key, value) = (inputs[0], inputs[1], inputs[2]);
        let q_shape = cx.value_shape(query)?;
        if q_shape.rank() < 2 {
            return Err(Error::RankMismatch {
                context: "Attention query".into(),
                expected: 3,
                got: q_shape.rank(),
            });
        }
        let head_dim = q_shape
            .dim(q_shape.rank() - 1)
            .and_then(oxide_ml_core::Dim::static_usize)
            .ok_or_else(|| Error::invalid_shape("Attention", "head dimension must be static"))?;
        let scale = self.scale.unwrap_or_else(|| 1.0 / (head_dim as f32).sqrt());
        let scores = cx.builder().matmul_with(query, key, false, true)?;
        let scale = cx.constant_scalar(scale);
        let mut scores = cx.builder().mul(scores, scale)?;
        if self.causal {
            let length = q_shape
                .dim(q_shape.rank() - 2)
                .and_then(oxide_ml_core::Dim::static_usize)
                .ok_or_else(|| {
                    Error::invalid_shape("Attention", "causal attention needs a static sequence")
                })?;
            let mut mask = vec![0.0f32; length * length];
            for row in 0..length {
                for column in (row + 1)..length {
                    mask[row * length + column] = -1.0e9;
                }
            }
            let mask =
                oxide_ml_graph::ConstantTensor::f32(Shape::from_static([length, length]), mask)?;
            let mask = cx.builder().constant_tensor(mask)?;
            scores = cx.builder().add(scores, mask)?;
        }
        let weights = cx.builder().softmax(scores, -1)?;
        let context = cx.builder().matmul(weights, value)?;
        Ok(vec![context])
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!("Attention(causal={})", self.causal),
            path.clone(),
        ));
    }
}

/// Multi-head self-attention.
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: Option<f32>,
    pub causal: bool,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Result<Self> {
        Self::with_seed(embed_dim, num_heads, next_seed())
    }

    pub fn with_seed(embed_dim: usize, num_heads: usize, seed: u64) -> Result<Self> {
        if num_heads == 0 || embed_dim % num_heads != 0 {
            return Err(Error::invalid_shape(
                "MultiHeadAttention",
                format!("embed_dim {embed_dim} must be divisible by num_heads {num_heads}"),
            ));
        }
        let head_dim = embed_dim / num_heads;
        Ok(MultiHeadAttention {
            query: Linear::with_seed(embed_dim, embed_dim, true, seed),
            key: Linear::with_seed(embed_dim, embed_dim, true, seed ^ 0x11),
            value: Linear::with_seed(embed_dim, embed_dim, true, seed ^ 0x22),
            output: Linear::with_seed(embed_dim, embed_dim, true, seed ^ 0x33),
            num_heads,
            head_dim,
            scale: None,
            causal: false,
        })
    }

    pub fn causal(mut self) -> Self {
        self.causal = true;
        self
    }
}

impl Module for MultiHeadAttention {
    fn module_type(&self) -> &'static str {
        "MultiHeadAttention"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.query
            .visit_parameters(&path.child("query"), registry)?;
        self.key.visit_parameters(&path.child("key"), registry)?;
        self.value
            .visit_parameters(&path.child("value"), registry)?;
        self.output
            .visit_parameters(&path.child("output"), registry)?;
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let shape = cx.value_shape(input)?;
        if shape.rank() != 3 {
            return Err(Error::RankMismatch {
                context: "MultiHeadAttention input".into(),
                expected: 3,
                got: shape.rank(),
            });
        }
        let batch = shape.dim(0).unwrap_or(oxide_ml_core::Dim::Dynamic);
        let sequence = shape
            .dim(1)
            .and_then(oxide_ml_core::Dim::static_usize)
            .ok_or_else(|| {
                Error::invalid_shape("MultiHeadAttention", "sequence length must be static")
            })?;
        let embed_dim = shape
            .dim(2)
            .and_then(oxide_ml_core::Dim::static_usize)
            .ok_or_else(|| {
                Error::invalid_shape("MultiHeadAttention", "embedding dimension must be static")
            })?;
        if embed_dim != self.num_heads * self.head_dim {
            return Err(Error::shape_mismatch(
                "MultiHeadAttention embedding",
                Shape::from_static([self.num_heads * self.head_dim]),
                shape.clone(),
            ));
        }

        let query = oxide_ml_model::module_forward(cx, "query", &self.query, input)?;
        let key = oxide_ml_model::module_forward(cx, "key", &self.key, input)?;
        let value = oxide_ml_model::module_forward(cx, "value", &self.value, input)?;

        let split = Shape::new(vec![
            batch,
            oxide_ml_core::Dim::Static(sequence),
            oxide_ml_core::Dim::Static(self.num_heads),
            oxide_ml_core::Dim::Static(self.head_dim),
        ]);
        let to_heads = |cx: &mut ModelContext, value: ValueId| -> Result<ValueId> {
            let reshaped = cx.builder().reshape(value, split.clone())?;
            cx.builder().transpose(reshaped, vec![0, 2, 1, 3])
        };
        let query = to_heads(cx, query)?;
        let key = to_heads(cx, key)?;
        let value = to_heads(cx, value)?;

        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (self.head_dim as f32).sqrt());
        let scores = cx.builder().matmul_with(query, key, false, true)?;
        let scale = cx.constant_scalar(scale);
        let mut scores = cx.builder().mul(scores, scale)?;
        if self.causal {
            let mut mask = vec![0.0f32; sequence * sequence];
            for row in 0..sequence {
                for column in (row + 1)..sequence {
                    mask[row * sequence + column] = -1.0e9;
                }
            }
            let mask = oxide_ml_graph::ConstantTensor::f32(
                Shape::from_static([sequence, sequence]),
                mask,
            )?;
            let mask = cx.builder().constant_tensor(mask)?;
            scores = cx.builder().add(scores, mask)?;
        }
        let weights = cx.builder().softmax(scores, -1)?;
        let context = cx.builder().matmul(weights, value)?;
        let context = cx.builder().transpose(context, vec![0, 2, 1, 3])?;
        let context = cx.builder().reshape(
            context,
            Shape::new(vec![
                batch,
                oxide_ml_core::Dim::Static(sequence),
                oxide_ml_core::Dim::Static(embed_dim),
            ]),
        )?;
        let output = oxide_ml_model::module_forward(cx, "output", &self.output, context)?;
        Ok(vec![output])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.query.visit_tensors(&path.child("query"), visitor)?;
        self.key.visit_tensors(&path.child("key"), visitor)?;
        self.value.visit_tensors(&path.child("value"), visitor)?;
        self.output.visit_tensors(&path.child("output"), visitor)?;
        Ok(())
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.query
            .visit_tensors_mut(&path.child("query"), visitor)?;
        self.key.visit_tensors_mut(&path.child("key"), visitor)?;
        self.value
            .visit_tensors_mut(&path.child("value"), visitor)?;
        self.output
            .visit_tensors_mut(&path.child("output"), visitor)?;
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        let mut node = ModuleDescriptor::new(
            format!(
                "MultiHeadAttention(num_heads={}, head_dim={})",
                self.num_heads, self.head_dim
            ),
            path.clone(),
        );
        self.query.describe(&path.child("query"), &mut node);
        self.key.describe(&path.child("key"), &mut node);
        self.value.describe(&path.child("value"), &mut node);
        self.output.describe(&path.child("output"), &mut node);
        out.children.push(node);
    }
}

// ---------------------------------------------------------------------------
// Recurrent layers
// ---------------------------------------------------------------------------

/// Gate layout shared by the recurrent implementations.
#[derive(Clone, Debug)]
pub struct RecurrentCell {
    pub input: Linear,
    pub hidden: Linear,
    pub hidden_size: usize,
}

impl RecurrentCell {
    fn new(input_size: usize, hidden_size: usize, gates: usize, seed: u64) -> Result<Self> {
        Ok(RecurrentCell {
            input: Linear::with_seed(input_size, gates * hidden_size, true, seed),
            hidden: Linear::with_seed(hidden_size, gates * hidden_size, false, seed ^ 0x55),
            hidden_size,
        })
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.input
            .visit_parameters(&path.child("input"), registry)?;
        self.hidden
            .visit_parameters(&path.child("hidden"), registry)
    }
}

fn sequence_length(cx: &ModelContext, input: ValueId, context: &str) -> Result<usize> {
    let shape = cx.value_shape(input)?;
    if shape.rank() != 3 {
        return Err(Error::RankMismatch {
            context: context.to_string(),
            expected: 3,
            got: shape.rank(),
        });
    }
    shape
        .dim(1)
        .and_then(oxide_ml_core::Dim::static_usize)
        .ok_or_else(|| Error::invalid_shape(context, "sequence length must be static"))
}

fn slice_gate(
    cx: &mut ModelContext,
    value: ValueId,
    start: usize,
    width: usize,
) -> Result<ValueId> {
    cx.builder().slice(value, 2, start, width)
}

fn cell_activation(cx: &mut ModelContext, kind: ActivationKind, value: ValueId) -> Result<ValueId> {
    cx.builder().activation(kind, value)
}

fn recurrent_forward(
    cx: &mut ModelContext,
    input: ValueId,
    cell: &RecurrentCell,
    kind: RecurrentKind,
) -> Result<ValueId> {
    let sequence = sequence_length(cx, input, "recurrent input")?;
    let width = cell.hidden_size;
    let mut hidden: Option<ValueId> = None;
    let mut memory: Option<ValueId> = None;

    for step in 0..sequence {
        let x = cx.builder().slice(input, 1, step, 1)?;
        let input_projection = oxide_ml_model::module_forward(cx, "cell.input", &cell.input, x)?;
        let hidden_projection = match hidden {
            Some(hidden) => Some(oxide_ml_model::module_forward(
                cx,
                "cell.hidden",
                &cell.hidden,
                hidden,
            )?),
            None => None,
        };

        match kind {
            RecurrentKind::Rnn => {
                let value = match hidden_projection {
                    Some(projection) => cx.builder().add(input_projection, projection)?,
                    None => input_projection,
                };
                hidden = Some(cell_activation(cx, ActivationKind::Tanh, value)?);
            }
            RecurrentKind::Gru => {
                let (xr, xz, xn) = (
                    slice_gate(cx, input_projection, 0, width)?,
                    slice_gate(cx, input_projection, width, width)?,
                    slice_gate(cx, input_projection, 2 * width, width)?,
                );
                let (hr, hz, hn) = match hidden_projection {
                    Some(projection) => (
                        slice_gate(cx, projection, 0, width)?,
                        slice_gate(cx, projection, width, width)?,
                        slice_gate(cx, projection, 2 * width, width)?,
                    ),
                    None => (xr, xz, xn),
                };
                let reset = if hidden.is_some() {
                    let sum = cx.builder().add(xr, hr)?;
                    cell_activation(cx, ActivationKind::Sigmoid, sum)?
                } else {
                    cell_activation(cx, ActivationKind::Sigmoid, xr)?
                };
                let update = if hidden.is_some() {
                    let sum = cx.builder().add(xz, hz)?;
                    cell_activation(cx, ActivationKind::Sigmoid, sum)?
                } else {
                    cell_activation(cx, ActivationKind::Sigmoid, xz)?
                };
                let candidate = if hidden.is_some() {
                    let gated = cx.builder().mul(reset, hn)?;
                    let sum = cx.builder().add(xn, gated)?;
                    cell_activation(cx, ActivationKind::Tanh, sum)?
                } else {
                    cell_activation(cx, ActivationKind::Tanh, xn)?
                };
                hidden = Some(match hidden {
                    Some(previous) => {
                        let one = cx.constant_scalar(1.0);
                        let keep = cx.builder().sub(one, update)?;
                        let kept = cx.builder().mul(keep, previous)?;
                        let added = cx.builder().mul(update, candidate)?;
                        cx.builder().add(kept, added)?
                    }
                    None => cx.builder().mul(update, candidate)?,
                });
            }
            RecurrentKind::Lstm => {
                let (xi, xf, xg, xo) = (
                    slice_gate(cx, input_projection, 0, width)?,
                    slice_gate(cx, input_projection, width, width)?,
                    slice_gate(cx, input_projection, 2 * width, width)?,
                    slice_gate(cx, input_projection, 3 * width, width)?,
                );
                let (hi, hf, hg, ho) = match hidden_projection {
                    Some(projection) => (
                        slice_gate(cx, projection, 0, width)?,
                        slice_gate(cx, projection, width, width)?,
                        slice_gate(cx, projection, 2 * width, width)?,
                        slice_gate(cx, projection, 3 * width, width)?,
                    ),
                    None => (xi, xf, xg, xo),
                };
                let combine = |cx: &mut ModelContext,
                               a: ValueId,
                               b: ValueId,
                               present: bool|
                 -> Result<ValueId> {
                    if present {
                        cx.builder().add(a, b)
                    } else {
                        Ok(a)
                    }
                };
                let input_sum = combine(cx, xi, hi, hidden.is_some())?;
                let input_gate = cell_activation(cx, ActivationKind::Sigmoid, input_sum)?;
                let forget_sum = combine(cx, xf, hf, hidden.is_some())?;
                let forget_gate = cell_activation(cx, ActivationKind::Sigmoid, forget_sum)?;
                let candidate_sum = combine(cx, xg, hg, hidden.is_some())?;
                let cell_gate = cell_activation(cx, ActivationKind::Tanh, candidate_sum)?;
                let output_sum = combine(cx, xo, ho, hidden.is_some())?;
                let output_gate = cell_activation(cx, ActivationKind::Sigmoid, output_sum)?;
                let retained = match memory {
                    Some(previous) => cx.builder().mul(forget_gate, previous)?,
                    None => forget_gate,
                };
                let added = cx.builder().mul(input_gate, cell_gate)?;
                let new_memory = cx.builder().add(retained, added)?;
                let activated = cell_activation(cx, ActivationKind::Tanh, new_memory)?;
                hidden = Some(cx.builder().mul(output_gate, activated)?);
                memory = Some(new_memory);
            }
        }
    }

    let last = hidden
        .ok_or_else(|| Error::invalid_shape("recurrent", "sequence length must be at least 1"))?;
    let shape = cx.value_shape(input)?;
    let batch = shape.dim(0).unwrap_or(oxide_ml_core::Dim::Dynamic);
    let target = Shape::new(vec![batch, oxide_ml_core::Dim::Static(cell.hidden_size)]);
    cx.builder().reshape(last, target)
}

#[derive(Clone, Copy)]
enum RecurrentKind {
    Rnn,
    Gru,
    Lstm,
}
/// Simple recurrent layer (tanh cell), returning the last hidden state `[B, H]`.
#[derive(Clone, Debug)]
pub struct Rnn {
    pub cell: RecurrentCell,
    pub input_size: usize,
}

impl Rnn {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        Ok(Rnn {
            cell: RecurrentCell::new(input_size, hidden_size, 1, next_seed())?,
            input_size,
        })
    }

    pub fn with_seed(input_size: usize, hidden_size: usize, seed: u64) -> Result<Self> {
        Ok(Rnn {
            cell: RecurrentCell::new(input_size, hidden_size, 1, seed)?,
            input_size,
        })
    }
}

impl Module for Rnn {
    fn module_type(&self) -> &'static str {
        "RNN"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.cell.visit_parameters(&path.child("cell"), registry)
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        Ok(vec![recurrent_forward(
            cx,
            input,
            &self.cell,
            RecurrentKind::Rnn,
        )?])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.cell
            .input
            .visit_tensors(&path.child("cell").child("input"), visitor)?;
        self.cell
            .hidden
            .visit_tensors(&path.child("cell").child("hidden"), visitor)
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.cell
            .input
            .visit_tensors_mut(&path.child("cell").child("input"), visitor)?;
        self.cell
            .hidden
            .visit_tensors_mut(&path.child("cell").child("hidden"), visitor)
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!("RNN(hidden_size={})", self.cell.hidden_size),
            path.clone(),
        ));
    }
}

/// GRU layer, returning the last hidden state.
#[derive(Clone, Debug)]
pub struct Gru {
    pub cell: RecurrentCell,
    pub input_size: usize,
}

impl Gru {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        Ok(Gru {
            cell: RecurrentCell::new(input_size, hidden_size, 3, next_seed())?,
            input_size,
        })
    }

    pub fn with_seed(input_size: usize, hidden_size: usize, seed: u64) -> Result<Self> {
        Ok(Gru {
            cell: RecurrentCell::new(input_size, hidden_size, 3, seed)?,
            input_size,
        })
    }
}

impl Module for Gru {
    fn module_type(&self) -> &'static str {
        "GRU"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.cell.visit_parameters(&path.child("cell"), registry)
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        Ok(vec![recurrent_forward(
            cx,
            input,
            &self.cell,
            RecurrentKind::Gru,
        )?])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.cell
            .input
            .visit_tensors(&path.child("cell").child("input"), visitor)?;
        self.cell
            .hidden
            .visit_tensors(&path.child("cell").child("hidden"), visitor)
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.cell
            .input
            .visit_tensors_mut(&path.child("cell").child("input"), visitor)?;
        self.cell
            .hidden
            .visit_tensors_mut(&path.child("cell").child("hidden"), visitor)
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!("GRU(hidden_size={})", self.cell.hidden_size),
            path.clone(),
        ));
    }
}

/// LSTM layer, returning the last hidden state.
#[derive(Clone, Debug)]
pub struct Lstm {
    pub cell: RecurrentCell,
    pub input_size: usize,
}

impl Lstm {
    pub fn new(input_size: usize, hidden_size: usize) -> Result<Self> {
        Ok(Lstm {
            cell: RecurrentCell::new(input_size, hidden_size, 4, next_seed())?,
            input_size,
        })
    }

    pub fn with_seed(input_size: usize, hidden_size: usize, seed: u64) -> Result<Self> {
        Ok(Lstm {
            cell: RecurrentCell::new(input_size, hidden_size, 4, seed)?,
            input_size,
        })
    }
}

impl Module for Lstm {
    fn module_type(&self) -> &'static str {
        "LSTM"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.cell.visit_parameters(&path.child("cell"), registry)
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        Ok(vec![recurrent_forward(
            cx,
            input,
            &self.cell,
            RecurrentKind::Lstm,
        )?])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.cell
            .input
            .visit_tensors(&path.child("cell").child("input"), visitor)?;
        self.cell
            .hidden
            .visit_tensors(&path.child("cell").child("hidden"), visitor)
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.cell
            .input
            .visit_tensors_mut(&path.child("cell").child("input"), visitor)?;
        self.cell
            .hidden
            .visit_tensors_mut(&path.child("cell").child("hidden"), visitor)
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children.push(ModuleDescriptor::new(
            format!("LSTM(hidden_size={})", self.cell.hidden_size),
            path.clone(),
        ));
    }
}
