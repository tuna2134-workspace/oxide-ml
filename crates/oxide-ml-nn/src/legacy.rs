//! Reference neural-network layers built on the graph builder.
//!
//! Layers own graph value ids, not tensors. This is what lets a compiled graph
//! be reused across batches while `ParameterStore` owns the mutable weights.

use oxide_ml_core::{
    DenseTensor, Dim, Error, ParameterId, ParameterStore, Result, Rng, Scalar, Shape, ValueId,
};
use oxide_ml_graph::{ActivationKind, Graph, GraphBuilder};

/// Trainable module contract.
pub trait Module {
    fn forward(&self, builder: &mut GraphBuilder, input: ValueId) -> Result<ValueId>;
    fn parameters(&self) -> Vec<ValueId>;
}

/// Fully connected layer: `y = x W^T + b`.
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: ValueId,
    pub bias: Option<ValueId>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(
        builder: &mut GraphBuilder,
        prefix: impl AsRef<str>,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self> {
        Self::with_dtype(
            builder,
            prefix,
            in_features,
            out_features,
            bias,
            oxide_ml_core::DType::F32,
        )
    }

    pub fn with_dtype(
        builder: &mut GraphBuilder,
        prefix: impl AsRef<str>,
        in_features: usize,
        out_features: usize,
        bias: bool,
        dtype: oxide_ml_core::DType,
    ) -> Result<Self> {
        let prefix = prefix.as_ref();
        let weight = builder.try_parameter(
            format!("{prefix}.weight"),
            Shape::from_static([out_features, in_features]),
            dtype,
            true,
        )?;
        let bias = if bias {
            Some(builder.try_parameter(
                format!("{prefix}.bias"),
                Shape::from_static([out_features]),
                dtype,
                true,
            )?)
        } else {
            None
        };
        Ok(Linear {
            weight,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, builder: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        let input_shape = builder.value_shape(input)?;
        if input_shape.rank() < 2
            || input_shape
                .dim(input_shape.rank() - 1)
                .and_then(Dim::static_usize)
                != Some(self.in_features)
        {
            return Err(Error::shape_mismatch(
                format!("Linear input (rank >= 2, last dim {})", self.in_features),
                Shape::from_static([self.in_features]),
                input_shape,
            ));
        }
        let output = builder.matmul_with(input, self.weight, false, true)?;
        match self.bias {
            Some(bias) => builder.add(output, bias),
            None => Ok(output),
        }
    }
}

impl Module for Linear {
    fn forward(&self, builder: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        Linear::forward(self, builder, input)
    }

    fn parameters(&self) -> Vec<ValueId> {
        let mut params = vec![self.weight];
        if let Some(bias) = self.bias {
            params.push(bias);
        }
        params
    }
}

/// Layer normalization over the trailing dimensions.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub gamma: ValueId,
    pub beta: ValueId,
    pub normalized_shape: Vec<usize>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(
        builder: &mut GraphBuilder,
        prefix: impl AsRef<str>,
        normalized_shape: Vec<usize>,
        eps: f32,
    ) -> Result<Self> {
        Self::with_dtype(
            builder,
            prefix,
            normalized_shape,
            eps,
            oxide_ml_core::DType::F32,
        )
    }

    pub fn with_dtype(
        builder: &mut GraphBuilder,
        prefix: impl AsRef<str>,
        normalized_shape: Vec<usize>,
        eps: f32,
        dtype: oxide_ml_core::DType,
    ) -> Result<Self> {
        if normalized_shape.is_empty() || normalized_shape.contains(&0) {
            return Err(Error::invalid_shape(
                "LayerNorm",
                "normalized_shape must be non-empty and non-zero",
            ));
        }
        let prefix = prefix.as_ref();
        let gamma = builder.try_parameter(
            format!("{prefix}.gamma"),
            Shape::from_static(&normalized_shape),
            dtype,
            true,
        )?;
        let beta = builder.try_parameter(
            format!("{prefix}.beta"),
            Shape::from_static(&normalized_shape),
            dtype,
            true,
        )?;
        Ok(LayerNorm {
            gamma,
            beta,
            normalized_shape,
            eps,
        })
    }

    pub fn forward(&self, builder: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        builder.layer_norm(
            input,
            self.gamma,
            self.beta,
            self.normalized_shape.clone(),
            self.eps,
        )
    }
}

impl Module for LayerNorm {
    fn forward(&self, builder: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        LayerNorm::forward(self, builder, input)
    }

    fn parameters(&self) -> Vec<ValueId> {
        vec![self.gamma, self.beta]
    }
}

/// Feed-forward MLP with configurable hidden activation.
#[derive(Clone, Debug)]
pub struct Mlp {
    pub layers: Vec<Linear>,
    pub activation: ActivationKind,
}

impl Mlp {
    /// Build an MLP from layer widths, e.g. `[784, 256, 128, 10]`.
    pub fn new(
        builder: &mut GraphBuilder,
        prefix: impl AsRef<str>,
        widths: &[usize],
        activation: ActivationKind,
        bias: bool,
    ) -> Result<Self> {
        Self::with_dtype(
            builder,
            prefix,
            widths,
            activation,
            bias,
            oxide_ml_core::DType::F32,
        )
    }

    pub fn with_dtype(
        builder: &mut GraphBuilder,
        prefix: impl AsRef<str>,
        widths: &[usize],
        activation: ActivationKind,
        bias: bool,
        dtype: oxide_ml_core::DType,
    ) -> Result<Self> {
        if widths.len() < 2 {
            return Err(Error::invalid_shape(
                "Mlp",
                "at least input and output widths are required",
            ));
        }
        let prefix = prefix.as_ref();
        let mut layers = Vec::with_capacity(widths.len() - 1);
        for index in 0..widths.len() - 1 {
            layers.push(Linear::with_dtype(
                builder,
                format!("{prefix}.fc{}", index + 1),
                widths[index],
                widths[index + 1],
                bias,
                dtype,
            )?);
        }
        Ok(Mlp { layers, activation })
    }

    pub fn forward(&self, builder: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        let mut value = input;
        for (index, layer) in self.layers.iter().enumerate() {
            value = layer.forward(builder, value)?;
            if index + 1 < self.layers.len() {
                value = builder.activation(self.activation, value)?;
            }
        }
        Ok(value)
    }
}

impl Module for Mlp {
    fn forward(&self, builder: &mut GraphBuilder, input: ValueId) -> Result<ValueId> {
        Mlp::forward(self, builder, input)
    }

    fn parameters(&self) -> Vec<ValueId> {
        self.layers.iter().flat_map(Module::parameters).collect()
    }
}

/// Construct a `ParameterStore` using deterministic, name-sensitive defaults.
///
/// `gamma` starts at one, `bias`/`beta` at zero, and matrices use Xavier
/// uniform initialization. A checkpoint loader can replace this store wholesale.
pub fn init_parameters<T: Scalar>(graph: &Graph, seed: u64) -> Result<ParameterStore<T>> {
    let mut store = ParameterStore::new();
    let mut rng = Rng::new(seed);
    for parameter in graph.parameters() {
        let dims = parameter
            .shape
            .static_dims()
            .ok_or_else(|| Error::invalid_shape("parameter", "graph parameter must be static"))?;
        let data = if parameter.name.ends_with("gamma") {
            DenseTensor::<T>::ones(&dims)
        } else if parameter.name.ends_with("bias") || parameter.name.ends_with("beta") {
            DenseTensor::<T>::zeros(&dims)
        } else if dims.len() >= 2 {
            let fan_in = if dims.len() == 4 {
                dims[1] * dims[2] * dims[3]
            } else {
                dims[1]
            };
            let fan_out = if dims.len() == 4 {
                dims[0] * dims[2] * dims[3]
            } else {
                dims[0]
            };
            let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
            let values = (0..dims.iter().product::<usize>())
                .map(|_| T::from_f64(rng.uniform(-bound, bound) as f64))
                .collect();
            DenseTensor::from_vec(&dims, values)?
        } else {
            DenseTensor::<T>::zeros(&dims)
        };
        store.add(
            parameter.id,
            parameter.name.clone(),
            data,
            parameter.trainable,
        )?;
    }
    Ok(store)
}

/// Initialize only the parameters referenced by `graph`, falling back to
/// `ParameterId` ordering for unnamed graphs.
pub fn parameter_ids(graph: &Graph) -> Vec<ParameterId> {
    graph
        .parameters()
        .iter()
        .map(|parameter| parameter.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use oxide_ml_core::DType;
    use oxide_ml_graph::GraphBuilder;

    use super::*;

    #[test]
    fn linear_builds_expected_shapes() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::from_static([4, 8]), DType::F32);
        let linear = Linear::new(&mut builder, "fc", 8, 3, true).unwrap();
        let y = linear.forward(&mut builder, x).unwrap();
        assert_eq!(builder.value_shape(y).unwrap(), Shape::from_static([4, 3]));
    }

    #[test]
    fn mlp_builds_expected_shapes() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::from_static([4, 8]), DType::F32);
        let mlp = Mlp::new(&mut builder, "mlp", &[8, 16, 3], ActivationKind::ReLU, true).unwrap();
        let y = mlp.forward(&mut builder, x).unwrap();
        assert_eq!(builder.value_shape(y).unwrap(), Shape::from_static([4, 3]));
        assert_eq!(mlp.parameters().len(), 4);
    }
}
