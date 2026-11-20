//! Tied (shared) parameters.
//!
//! A tied parameter is a *single* tensor reachable through two module paths.
//! [`ParameterRegistry`](oxide_ml_model::ParameterRegistry) detects the shared
//! storage and records the second path as an alias of the first canonical
//! parameter, so the graph binds one parameter, the optimizer owns one slot and
//! a checkpoint contains one entry. The alias table still resolves both names,
//! which keeps state dictionaries structurally stable.
//!
//! [`TiedLinear`] is the common case: a language-model head that reuses the
//! token-embedding matrix instead of owning a copy.

use std::sync::Arc;

use oxide_ml_core::{DenseTensor, Error, Result, ValueId};
use oxide_ml_model::{
    ModelContext, Module, ModuleDescriptor, ModulePath, ParameterRegistry, ParameterRole,
};

use crate::modules::{Embedding, Linear};

/// A module whose weight can be reused by a [`TiedLinear`].
pub trait TiedWeightSource {
    /// The `[out_features, in_features]` matrix shared with the head.
    fn tied_weight(&self) -> &DenseTensor<f32>;
}

impl TiedWeightSource for Linear {
    fn tied_weight(&self) -> &DenseTensor<f32> {
        &self.weight
    }
}

impl TiedWeightSource for Embedding {
    fn tied_weight(&self) -> &DenseTensor<f32> {
        &self.weight
    }
}

/// A linear head that shares another module's weight matrix.
///
/// `TiedLinear<Embedding>` computes `logits = hidden @ embedding.weight^T`
/// without owning a weight. Construct the source as an `Arc` and clone that
/// `Arc` into both fields; cloning the *module* would clone the tensor and
/// silently untie the parameters.
///
/// ```ignore
/// #[derive(Module)]
/// struct Lm {
///     embed: Arc<Embedding>,
///     head: TiedLinear<Embedding>,
/// }
///
/// let embed = Arc::new(Embedding::new(32_000, 2048));
/// let model = Lm { head: TiedLinear::new(embed.clone()), embed };
/// assert_eq!(model.descriptor().unwrap().parameter_count(), 1);
/// ```
pub struct TiedLinear<S: ?Sized> {
    source: Arc<S>,
    flatten: bool,
}

impl<S: ?Sized> TiedLinear<S> {
    /// A head that applies the shared matrix to the last dimension, for rank-2
    /// inputs.
    pub fn new(source: Arc<S>) -> Self {
        TiedLinear {
            source,
            flatten: false,
        }
    }

    /// A head that flattens every leading dimension before the matrix product,
    /// for `[batch, sequence, hidden]` decoder states.
    pub fn flattening(source: Arc<S>) -> Self {
        TiedLinear {
            source,
            flatten: true,
        }
    }

    /// The shared source module.
    pub fn source(&self) -> &Arc<S> {
        &self.source
    }
}

impl<S: ?Sized> Clone for TiedLinear<S> {
    fn clone(&self) -> Self {
        TiedLinear {
            source: Arc::clone(&self.source),
            flatten: self.flatten,
        }
    }
}

impl<S: ?Sized> std::fmt::Debug for TiedLinear<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TiedLinear")
            .field("source", &std::any::type_name::<S>())
            .field("flatten", &self.flatten)
            .finish()
    }
}

impl<S: TiedWeightSource + Module + ?Sized> Module for TiedLinear<S> {
    fn module_type(&self) -> &'static str {
        "TiedLinear"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        // Registering the shared tensor makes the registry record this path as
        // an alias of the source's canonical parameter.
        registry.insert_tensor(
            &path.child("weight"),
            self.source.tied_weight(),
            ParameterRole::Trainable,
        )
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        let weight = cx.parameter("weight")?;
        let dims = self.source.tied_weight().dims();
        if dims.len() != 2 {
            return Err(Error::invalid_graph(
                "TiedLinear source weight must be rank 2",
            ));
        }
        let scale = dims[1];
        let shape = cx.value_shape(input)?;
        match shape.dim(shape.rank().saturating_sub(1)) {
            Some(dim) => {
                if dim.static_usize() != Some(scale) {
                    return Err(Error::invalid_shape(
                        "TiedLinear",
                        format!("input last dimension must be {scale}, got {dim:?}"),
                    ));
                }
            }
            None => return Err(Error::invalid_shape("TiedLinear", "input must be ranked")),
        }
        let input = if self.flatten {
            let shape = cx.value_shape(input)?;
            if shape.rank() > 2 {
                let mut rows = 1usize;
                for dim in shape.dims() {
                    match dim.static_usize() {
                        Some(value) => rows *= value,
                        None => {
                            return Err(Error::invalid_shape(
                                "TiedLinear",
                                "flattening a tied head requires static leading dimensions",
                            ));
                        }
                    }
                }
                let rows = rows / scale;
                cx.builder()
                    .reshape(input, oxide_ml_core::Shape::from_static([rows, scale]))?
            } else {
                input
            }
        } else {
            input
        };
        let logits = cx.builder().matmul_with(input, weight, false, true)?;
        Ok(vec![logits])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        visitor(
            &path.child("weight"),
            self.source.tied_weight(),
            ParameterRole::Trainable,
        )
    }

    fn visit_tensors_mut(
        &mut self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &mut DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        // The source owns the tensor; a tied head cannot mutate the shared
        // weight through its own path. The parameter store is authoritative.
        let _ = (path, visitor);
        Ok(())
    }

    fn describe(&self, path: &ModulePath, out: &mut ModuleDescriptor) {
        out.children
            .push(ModuleDescriptor::new(self.module_type(), path.clone()));
    }
}
