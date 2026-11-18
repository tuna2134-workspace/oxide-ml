//! Reverse-mode automatic differentiation over the SSA graph.
//!
//! The backward pass is itself a graph: `build_training_graph` clones the
//! forward graph, appends fused gradient nodes in reverse topological order,
//! and returns one executable graph. There is no runtime tape and no second
//! kernel set for matrix gradients.

mod registry;
mod rules;

use std::collections::HashMap;

use oxide_ml_core::{Error, ParameterId, Result, ValueId};
use oxide_ml_graph::{Graph, GraphBuilder, GraphVerifier};

pub use registry::{FunctionRule, GradientRegistry, GradientRule, GradientRuleFn};

/// A forward graph plus its materialized backward graph.
#[derive(Clone, Debug)]
pub struct TrainingGraph {
    pub graph: Graph,
    pub loss: ValueId,
    pub forward_outputs: Vec<ValueId>,
    pub forward_node_count: usize,
    /// Parameter value -> gradient value in `graph`.
    pub parameter_grads: HashMap<ValueId, ValueId>,
    /// Every differentiated value -> accumulated gradient value.
    pub value_grads: HashMap<ValueId, ValueId>,
}

impl TrainingGraph {
    pub fn gradient_for_parameter(&self, id: ParameterId) -> Result<ValueId> {
        let parameter = self.graph.parameter(id)?;
        self.parameter_grads
            .get(&parameter.value)
            .copied()
            .ok_or_else(|| {
                Error::Autograd(format!(
                    "parameter {:?} ({}) did not receive a gradient",
                    parameter.name, id
                ))
            })
    }

    pub fn parameter_gradients(&self) -> impl Iterator<Item = (ParameterId, ValueId)> + '_ {
        self.graph.parameters().iter().filter_map(|parameter| {
            self.parameter_grads
                .get(&parameter.value)
                .map(|&gradient| (parameter.id, gradient))
        })
    }

    pub fn gradient_for_value(&self, value: ValueId) -> Option<ValueId> {
        self.value_grads.get(&value).copied()
    }
}

/// Build a training graph whose backward nodes are appended after the forward
/// nodes. `loss` must be a scalar value in `forward`.
pub fn build_training_graph(forward: &Graph, loss: ValueId) -> Result<TrainingGraph> {
    let registry = GradientRegistry::builtin();
    build_training_graph_with_registry(forward, loss, &registry)
}

/// Build a training graph using an explicit gradient registry. Custom rules
/// must be registered before this call; resolution is a single compile-time
/// lookup table query per forward node.
pub fn build_training_graph_with_registry(
    forward: &Graph,
    loss: ValueId,
    registry: &GradientRegistry,
) -> Result<TrainingGraph> {
    GraphVerifier::verify(forward)?;
    registry.ensure_supported(forward)?;
    let loss_value = forward.value(loss)?;
    if !loss_value.shape.is_scalar() {
        return Err(Error::Autograd(format!(
            "loss value {loss} must be scalar, got shape {}",
            loss_value.shape
        )));
    }

    let forward_node_count = forward.nodes().len();
    let mut builder = GraphBuilder::from_graph(forward.clone());
    let seed = builder.constant_scalar_as(1.0, loss_value.dtype)?;
    let mut grads: HashMap<ValueId, ValueId> = HashMap::new();
    grads.insert(loss, seed);

    for node in forward.nodes().iter().rev() {
        let upstream: Vec<Option<ValueId>> = node
            .outputs
            .iter()
            .map(|output| grads.get(output).copied())
            .collect();
        if upstream.iter().all(Option::is_none) {
            continue;
        }
        let rule = registry.resolve(&node.op)?;
        let input_grads = rule.backward(&mut builder, node, &upstream)?;
        for (input, gradient) in node.inputs.iter().zip(input_grads) {
            if let Some(gradient) = gradient {
                accumulate(&mut builder, &mut grads, *input, gradient)?;
            }
        }
    }

    let mut parameter_grads = HashMap::new();
    for parameter in forward.parameters() {
        if let Some(&gradient) = grads.get(&parameter.value) {
            parameter_grads.insert(parameter.value, gradient);
        }
    }
    // Memory-planned execution only retains explicit roots. Loss and parameter
    // gradients must therefore be graph outputs in a training graph.
    let mut outputs = forward.outputs().to_vec();
    for value in std::iter::once(loss).chain(parameter_grads.values().copied()) {
        if !outputs.contains(&value) {
            outputs.push(value);
        }
    }
    let graph = builder.finish(&outputs)?;

    Ok(TrainingGraph {
        graph,
        loss,
        forward_outputs: forward.outputs().to_vec(),
        forward_node_count,
        parameter_grads,
        value_grads: grads,
    })
}

fn accumulate(
    builder: &mut GraphBuilder,
    grads: &mut HashMap<ValueId, ValueId>,
    value: ValueId,
    gradient: ValueId,
) -> Result<()> {
    match grads.get(&value).copied() {
        None => {
            grads.insert(value, gradient);
            Ok(())
        }
        Some(existing) => {
            let existing_shape = builder.value_shape(existing)?;
            let gradient_shape = builder.value_shape(gradient)?;
            let combined = if existing_shape == gradient_shape {
                builder.add(existing, gradient)?
            } else {
                let target = builder.value_shape(value)?;
                let left = builder.unbroadcast(existing, target.clone())?;
                let right = builder.unbroadcast(gradient, target)?;
                builder.add(left, right)?
            };
            grads.insert(value, combined);
            Ok(())
        }
    }
}
