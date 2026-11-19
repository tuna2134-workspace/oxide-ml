//! User-provided kernels as first-class graph operators.
//!
//! A custom kernel is described by a [`KernelDescriptor`] (dtypes, arity,
//! launch shape, architecture, side effects) and executed through a
//! [`CustomKernelRegistry`]. The graph IR stores only the kernel name, declared
//! output shapes and arity, so a graph remains backend-independent.
//!
//! Kernel bodies are supplied by the caller. On CUDA they wrap a
//! `#[cuda_module]` / `#[kernel]` launch; on CPU they provide a reference
//! implementation (used for tests and as a numerical oracle).

use std::collections::HashMap;
use std::sync::Arc;

use oxide_ml_core::{DType, DenseTensor, Device, Error, Result, Scalar, ValueId};
use oxide_ml_graph::{GraphBuilder, Node};

/// Version of the descriptor metadata. Bump when fields change meaning.
pub const KERNEL_DESCRIPTOR_VERSION: u32 = 1;

/// How strictly a tensor constraint is checked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstraintKind {
    Exact,
    AtLeast,
    Broadcast,
}

/// One dimension constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dimension {
    Any,
    Fixed(usize),
    /// Equal to an input dimension at `(tensor, axis)`.
    SameAs {
        tensor: usize,
        axis: usize,
    },
}

/// Dtype/rank/shape expectation for one operand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorConstraint {
    pub dtype: Option<DType>,
    pub rank: Option<usize>,
    pub dims: Vec<Dimension>,
    pub kind: ConstraintKind,
}

impl TensorConstraint {
    pub fn any() -> Self {
        TensorConstraint {
            dtype: None,
            rank: None,
            dims: Vec::new(),
            kind: ConstraintKind::Exact,
        }
    }

    pub fn contiguous(dims: impl IntoIterator<Item = usize>, dtype: DType) -> Self {
        TensorConstraint {
            dtype: Some(dtype),
            rank: None,
            dims: dims.into_iter().map(Dimension::Fixed).collect(),
            kind: ConstraintKind::Exact,
        }
    }

    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = Some(rank);
        self
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    fn validate(&self, input: &[usize], dtype: DType, position: usize) -> Result<()> {
        if let Some(expected) = self.dtype
            && expected != dtype
        {
            return Err(Error::DTypeMismatch {
                context: format!("custom kernel input {position}"),
                expected,
                got: dtype,
            });
        }
        if let Some(rank) = self.rank
            && input.len() != rank
        {
            return Err(Error::RankMismatch {
                context: format!("custom kernel input {position}"),
                expected: rank,
                got: input.len(),
            });
        }
        for (axis, dimension) in self.dims.iter().enumerate() {
            if axis >= input.len() {
                return Err(Error::RankMismatch {
                    context: format!("custom kernel input {position} constraint"),
                    expected: axis + 1,
                    got: input.len(),
                });
            }
            match dimension {
                Dimension::Any => {}
                Dimension::Fixed(expected) if *expected != input[axis] => {
                    return Err(Error::shape_mismatch(
                        format!("custom kernel input {position} axis {axis}"),
                        oxide_ml_core::Shape::from_static([*expected]),
                        oxide_ml_core::Shape::from_static([input[axis]]),
                    ));
                }
                Dimension::Fixed(_) => {}
                Dimension::SameAs {
                    tensor,
                    axis: other,
                } => {
                    let compared = input.get(*other).copied().filter(|_| *tensor == position);
                    if let Some(compared) = compared
                        && compared != input[axis]
                    {
                        return Err(Error::shape_mismatch(
                            format!("custom kernel input {position} axis {axis}"),
                            oxide_ml_core::Shape::from_static([compared]),
                            oxide_ml_core::Shape::from_static([input[axis]]),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Whether a gradient is available for this kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AutogradSupport {
    /// Training with this kernel is rejected with an explicit error.
    None,
    /// A backward implementation was registered.
    Explicit,
}

/// Documented mutation performed by a kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SideEffect {
    None,
    ReadsInput,
    WritesWorkspace,
    AtomicAccumulation,
}

/// Static launch configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaunchConfig {
    pub block: [u32; 3],
    pub shared_memory_bytes: usize,
    pub elements_per_thread: u32,
}

impl LaunchConfig {
    pub const fn one_dimensional(block: u32) -> Self {
        LaunchConfig {
            block: [block, 1, 1],
            shared_memory_bytes: 0,
            elements_per_thread: 1,
        }
    }

    pub fn grid_for(&self, elements: usize) -> [u32; 3] {
        let per_block = (self.block[0] as usize)
            .saturating_mul(self.block[1] as usize)
            .saturating_mul(self.block[2] as usize)
            .max(1)
            * (self.elements_per_thread as usize).max(1);
        let blocks = elements.div_ceil(per_block).max(1);
        [blocks as u32, 1, 1]
    }
}

impl Default for LaunchConfig {
    fn default() -> Self {
        LaunchConfig::one_dimensional(256)
    }
}

/// Metadata describing one custom kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelDescriptor {
    pub name: String,
    pub abi_version: u32,
    pub dtype: DType,
    pub inputs: Vec<TensorConstraint>,
    pub outputs: Vec<TensorConstraint>,
    pub launch: LaunchConfig,
    pub required_architecture: Option<String>,
    pub autograd: AutogradSupport,
    pub side_effects: Vec<SideEffect>,
}

impl KernelDescriptor {
    pub fn new(
        name: impl Into<String>,
        dtype: DType,
        inputs: Vec<TensorConstraint>,
        outputs: Vec<TensorConstraint>,
    ) -> Self {
        KernelDescriptor {
            name: name.into(),
            abi_version: KERNEL_DESCRIPTOR_VERSION,
            dtype,
            inputs,
            outputs,
            launch: LaunchConfig::default(),
            required_architecture: None,
            autograd: AutogradSupport::None,
            side_effects: vec![SideEffect::None],
        }
    }

    pub fn with_launch(mut self, launch: LaunchConfig) -> Self {
        self.launch = launch;
        self
    }

    pub fn with_architecture(mut self, architecture: impl Into<String>) -> Self {
        self.required_architecture = Some(architecture.into());
        self
    }

    pub fn with_autograd(mut self) -> Self {
        self.autograd = AutogradSupport::Explicit;
        self
    }

    pub fn with_side_effects(mut self, effects: Vec<SideEffect>) -> Self {
        self.side_effects = effects;
        self
    }

    pub fn output_shapes(&self) -> Option<Vec<Vec<usize>>> {
        self.outputs
            .iter()
            .map(|constraint| {
                constraint
                    .dims
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Fixed(value) => Some(*value),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(Error::invalid_graph("custom kernel name is empty"));
        }
        if self.abi_version != KERNEL_DESCRIPTOR_VERSION {
            return Err(Error::invalid_graph(format!(
                "custom kernel {:?} has descriptor version {} (expected {})",
                self.name, self.abi_version, KERNEL_DESCRIPTOR_VERSION
            )));
        }
        if self.outputs.is_empty() {
            return Err(Error::invalid_graph(format!(
                "custom kernel {:?} must declare at least one output",
                self.name
            )));
        }
        if self.launch.block.contains(&0) {
            return Err(Error::invalid_graph(format!(
                "custom kernel {:?} has a zero launch dimension",
                self.name
            )));
        }
        Ok(())
    }
}

/// Fully specialized kernel identity used for cache lookups.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SpecializationKey {
    pub kernel: String,
    pub abi_version: u32,
    pub architecture: String,
    pub dtype: DType,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub layout: String,
    pub constants: Vec<u64>,
}

impl SpecializationKey {
    pub fn new(
        descriptor: &KernelDescriptor,
        input_shapes: Vec<Vec<usize>>,
        output_shapes: Vec<Vec<usize>>,
    ) -> Self {
        SpecializationKey {
            kernel: descriptor.name.clone(),
            abi_version: descriptor.abi_version,
            architecture: descriptor
                .required_architecture
                .clone()
                .unwrap_or_else(|| "portable".to_string()),
            dtype: descriptor.dtype,
            input_shapes,
            output_shapes,
            layout: "contiguous".to_string(),
            constants: Vec::new(),
        }
    }

    pub fn with_architecture(mut self, architecture: impl Into<String>) -> Self {
        self.architecture = architecture.into();
        self
    }
}

/// Arguments passed to a custom kernel implementation.
pub struct CustomCall<'a, T: Scalar = f32> {
    pub name: &'a str,
    pub inputs: &'a [DenseTensor<T>],
    pub output_shapes: &'a [Vec<usize>],
    pub device: Device,
    pub specialization: &'a SpecializationKey,
}

pub type CustomForward<T> =
    Arc<dyn Fn(&CustomCall<'_, T>) -> Result<Vec<DenseTensor<T>>> + Send + Sync>;

/// Graph-level backward rule for a custom kernel.
pub type CustomBackward = Arc<
    dyn Fn(&mut GraphBuilder, &Node, &[Option<ValueId>]) -> Result<Vec<Option<ValueId>>>
        + Send
        + Sync,
>;

/// A registered kernel body plus its descriptor.
pub struct CustomKernel<T: Scalar = f32> {
    descriptor: KernelDescriptor,
    forward: CustomForward<T>,
    backward: Option<CustomBackward>,
}

impl<T: Scalar> CustomKernel<T> {
    pub fn new(descriptor: KernelDescriptor, forward: CustomForward<T>) -> Result<Self> {
        descriptor.validate()?;
        if descriptor.dtype != T::DTYPE {
            return Err(Error::DTypeMismatch {
                context: format!("custom kernel {:?}", descriptor.name),
                expected: T::DTYPE,
                got: descriptor.dtype,
            });
        }
        Ok(CustomKernel {
            descriptor,
            forward,
            backward: None,
        })
    }

    pub fn with_backward(mut self, backward: CustomBackward) -> Self {
        self.backward = Some(backward);
        self
    }

    pub fn descriptor(&self) -> &KernelDescriptor {
        &self.descriptor
    }

    pub fn backward(&self) -> Option<&CustomBackward> {
        self.backward.as_ref()
    }
}

impl<T: Scalar> std::fmt::Debug for CustomKernel<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomKernel")
            .field("descriptor", &self.descriptor.name)
            .field("has_backward", &self.backward.is_some())
            .finish()
    }
}

/// Registry of custom kernels for one scalar type.
pub struct CustomKernelRegistry<T: Scalar = f32> {
    kernels: HashMap<String, Arc<CustomKernel<T>>>,
}

impl<T: Scalar> Default for CustomKernelRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> Clone for CustomKernelRegistry<T> {
    fn clone(&self) -> Self {
        CustomKernelRegistry {
            kernels: self.kernels.clone(),
        }
    }
}

impl<T: Scalar> std::fmt::Debug for CustomKernelRegistry<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomKernelRegistry")
            .field("kernels", &self.kernels.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl<T: Scalar> CustomKernelRegistry<T> {
    pub fn new() -> Self {
        CustomKernelRegistry {
            kernels: HashMap::new(),
        }
    }

    /// Validate and register a kernel. Re-registering the same name is an error.
    pub fn register(&mut self, kernel: CustomKernel<T>) -> Result<()> {
        kernel.descriptor.validate()?;
        if self.kernels.contains_key(&kernel.descriptor.name) {
            return Err(Error::invalid_graph(format!(
                "custom kernel {:?} is already registered",
                kernel.descriptor.name
            )));
        }
        self.kernels
            .insert(kernel.descriptor.name.clone(), Arc::new(kernel));
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&Arc<CustomKernel<T>>> {
        self.kernels.get(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.kernels.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.kernels.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    pub fn has_backward(&self, name: &str) -> bool {
        self.kernels
            .get(name)
            .map(|kernel| kernel.backward.is_some())
            .unwrap_or(false)
    }

    /// Validate a call and produce its specialization key.
    pub fn specialize(
        &self,
        name: &str,
        inputs: &[DenseTensor<T>],
        output_shapes: &[Vec<usize>],
    ) -> Result<SpecializationKey> {
        let kernel = self.kernels.get(name).ok_or_else(|| {
            Error::invalid_graph(format!("custom kernel {name:?} is not registered"))
        })?;
        let descriptor = &kernel.descriptor;
        if inputs.len() != descriptor.inputs.len() {
            return Err(Error::invalid_graph(format!(
                "custom kernel {:?} expects {} inputs, got {}",
                name,
                descriptor.inputs.len(),
                inputs.len()
            )));
        }
        if output_shapes.len() != descriptor.outputs.len() {
            return Err(Error::invalid_graph(format!(
                "custom kernel {:?} expects {} outputs, got {}",
                name,
                descriptor.outputs.len(),
                output_shapes.len()
            )));
        }
        for (position, (constraint, tensor)) in descriptor.inputs.iter().zip(inputs).enumerate() {
            constraint.validate(&tensor.dims(), tensor.dtype(), position)?;
        }
        Ok(SpecializationKey::new(
            descriptor,
            inputs.iter().map(DenseTensor::dims).collect(),
            output_shapes.to_vec(),
        ))
    }

    /// Execute a registered kernel.
    pub fn execute(
        &self,
        name: &str,
        inputs: &[DenseTensor<T>],
        output_shapes: &[Vec<usize>],
        device: Device,
    ) -> Result<Vec<DenseTensor<T>>> {
        let specialization = self.specialize(name, inputs, output_shapes)?;
        let kernel = self
            .kernels
            .get(name)
            .expect("specialize validated registration");
        let call = CustomCall {
            name,
            inputs,
            output_shapes,
            device,
            specialization: &specialization,
        };
        let outputs = (kernel.forward)(&call)?;
        if outputs.len() != output_shapes.len() {
            return Err(Error::execution(format!(
                "custom kernel {:?} produced {} outputs, expected {}",
                name,
                outputs.len(),
                output_shapes.len()
            )));
        }
        for (index, (tensor, expected)) in outputs.iter().zip(output_shapes).enumerate() {
            if tensor.dims() != *expected {
                return Err(Error::shape_mismatch(
                    format!("custom kernel {name:?} output {index}"),
                    oxide_ml_core::Shape::from_static(expected),
                    tensor.shape().clone(),
                ));
            }
        }
        Ok(outputs)
    }

    /// Execute a graph-level backward rule.
    pub fn backward(
        &self,
        name: &str,
        builder: &mut GraphBuilder,
        node: &Node,
        gradients: &[Option<ValueId>],
    ) -> Result<Vec<Option<ValueId>>> {
        let kernel = self
            .kernels
            .get(name)
            .ok_or_else(|| Error::Autograd(format!("custom kernel {name:?} is not registered")))?;
        let backward = kernel.backward.as_ref().ok_or_else(|| {
            Error::Autograd(format!(
                "custom kernel {name:?} has no backward rule; \
                 register one with CustomKernel::with_backward"
            ))
        })?;
        backward(builder, node, gradients)
    }

    /// Snapshot of name -> backward rule, used by the autograd registry.
    pub fn backward_rules(&self) -> HashMap<String, CustomBackward> {
        self.kernels
            .iter()
            .filter_map(|(name, kernel)| {
                kernel
                    .backward
                    .as_ref()
                    .map(|backward| (name.clone(), Arc::clone(backward)))
            })
            .collect()
    }
}

/// A custom kernel and its descriptor, ready to be registered as an operator.
#[derive(Clone, Debug)]
pub struct KernelOperator {
    pub descriptor: KernelDescriptor,
}

impl KernelOperator {
    pub fn new(descriptor: KernelDescriptor) -> Self {
        KernelOperator { descriptor }
    }

    pub fn name(&self) -> &str {
        &self.descriptor.name
    }

    pub fn output_shapes(&self) -> Option<Vec<Vec<usize>>> {
        self.descriptor.output_shapes()
    }
}
