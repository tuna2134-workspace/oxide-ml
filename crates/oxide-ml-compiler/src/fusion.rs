//! Conservative operator fusion.
//!
//! Every pattern requires the producer's result to have exactly one use and to
//! not be a graph output, so fusion never duplicates work or changes an
//! externally visible value. Fused kernels compute the same operations in the
//! same order; no floating-point reassociation is performed.
//!
//! Patterns:
//!
//! - `Add + ReLU` -> `AddRelu`
//! - `Mul + Add` -> `MulAdd`
//! - `MatMul + Add` -> `MatMulBias`

use std::collections::{HashMap, HashSet};

use oxide_ml_core::{NodeId, Scalar, ValueId};
use oxide_ml_graph::{Graph, Node, Op, OpAttrs};

use crate::pass::{GraphPass, PassContext, PassResult};
use crate::rewrite::remove_nodes;

/// Conservative operator fusion.
pub struct OperatorFusion;

struct Index {
    /// Number of uses of each value.
    uses: HashMap<ValueId, usize>,
    /// Producing node for single-output values.
    producer: HashMap<ValueId, NodeId>,
    outputs: HashSet<ValueId>,
}

impl Index {
    fn build(graph: &Graph) -> Self {
        let mut uses: HashMap<ValueId, usize> = HashMap::new();
        let mut producer = HashMap::new();
        for node in graph.nodes() {
            for &input in &node.inputs {
                *uses.entry(input).or_insert(0) += 1;
            }
            if node.outputs.len() == 1 {
                producer.insert(node.outputs[0], node.id);
            }
        }
        Index {
            uses,
            producer,
            outputs: graph.outputs.iter().copied().collect(),
        }
    }

    fn single_use_producer(&self, value: ValueId, op: &Op, graph: &Graph) -> Option<NodeId> {
        if self.outputs.contains(&value) || self.uses.get(&value).copied().unwrap_or(0) != 1 {
            return None;
        }
        let id = *self.producer.get(&value)?;
        let node = graph.node(id).ok()?;
        (node.op == *op).then_some(id)
    }
}

impl<T: Scalar> GraphPass<T> for OperatorFusion {
    fn name(&self) -> &'static str {
        "fusion"
    }

    fn run(
        &self,
        graph: &mut Graph,
        _context: &PassContext<'_, T>,
    ) -> oxide_ml_core::Result<PassResult> {
        let index = Index::build(graph);
        let mut remove: HashSet<NodeId> = HashSet::new();
        let mut add_relu = 0usize;
        let mut mul_add = 0usize;
        let mut matmul_bias = 0usize;

        // ReLU fusion keeps the ReLU output ValueId on the Add node.
        let mut relu_pairs: Vec<(NodeId, NodeId, ValueId)> = Vec::new();
        for node in graph.nodes() {
            if !matches!(node.op, Op::Add) || node.outputs.len() != 1 {
                continue;
            }
            let sum = node.outputs[0];
            if index.outputs.contains(&sum) || index.uses.get(&sum).copied().unwrap_or(0) != 1 {
                continue;
            }
            // Find the consumer through the node list (a value has one
            // consumer when `uses` == 1).
            let relu = graph
                .nodes()
                .iter()
                .find(|candidate| candidate.inputs == [sum] && matches!(candidate.op, Op::ReLU));
            if let Some(relu) = relu {
                relu_pairs.push((node.id, relu.id, relu.outputs[0]));
            }
        }
        for (add_id, relu_id, relu_out) in &relu_pairs {
            let node = graph
                .nodes
                .iter_mut()
                .find(|node| node.id == *add_id)
                .expect("recorded Add node");
            node.op = Op::AddRelu;
            node.outputs[0] = *relu_out;
            remove.insert(*relu_id);
            add_relu += 1;
        }
        // Reload after mutation: AddRelu nodes must not be considered again.
        let index = Index::build(graph);
        let mut rewrites: Vec<(NodeId, NodeId, Op, OpAttrs, Vec<ValueId>)> = Vec::new();
        for node in graph.nodes() {
            if !matches!(node.op, Op::Add) || node.outputs.len() != 1 {
                continue;
            }
            let output = node.outputs[0];
            // The fused node keeps the Add output ValueId, so an Add that is a
            // graph output can still be fused. Only the producer must not be a
            // graph output (checked in `single_use_producer`).
            if node.inputs.len() != 2 {
                continue;
            }
            let (left, right) = (node.inputs[0], node.inputs[1]);
            let output_shape = graph.value(output)?.shape.clone();

            if let Some(matmul_id) = index.single_use_producer(left, &Op::MatMul, graph) {
                let matmul = graph.node(matmul_id)?;
                if graph.value(left)?.shape == output_shape {
                    let mut attrs = matmul.attrs.clone();
                    if !matches!(attrs, OpAttrs::MatMul { .. }) {
                        attrs = OpAttrs::MatMul {
                            transpose_a: false,
                            transpose_b: false,
                        };
                    }
                    rewrites.push((
                        node.id,
                        matmul_id,
                        Op::MatMulBias,
                        attrs,
                        vec![matmul.inputs[0], matmul.inputs[1], right],
                    ));
                    continue;
                }
            }
            if let Some(matmul_id) = index.single_use_producer(right, &Op::MatMul, graph) {
                let matmul = graph.node(matmul_id)?;
                if graph.value(right)?.shape == output_shape {
                    let mut attrs = matmul.attrs.clone();
                    if !matches!(attrs, OpAttrs::MatMul { .. }) {
                        attrs = OpAttrs::MatMul {
                            transpose_a: false,
                            transpose_b: false,
                        };
                    }
                    rewrites.push((
                        node.id,
                        matmul_id,
                        Op::MatMulBias,
                        attrs,
                        vec![matmul.inputs[0], matmul.inputs[1], left],
                    ));
                    continue;
                }
            }
            if let Some(mul_id) = index.single_use_producer(left, &Op::Mul, graph) {
                let mul = graph.node(mul_id)?;
                rewrites.push((
                    node.id,
                    mul_id,
                    Op::MulAdd,
                    OpAttrs::None,
                    vec![mul.inputs[0], mul.inputs[1], right],
                ));
                continue;
            }
            if let Some(mul_id) = index.single_use_producer(right, &Op::Mul, graph) {
                let mul = graph.node(mul_id)?;
                rewrites.push((
                    node.id,
                    mul_id,
                    Op::MulAdd,
                    OpAttrs::None,
                    vec![mul.inputs[0], mul.inputs[1], left],
                ));
            }
        }
        for (add_id, producer_id, op, attrs, inputs) in rewrites {
            match op {
                Op::MatMulBias => matmul_bias += 1,
                Op::MulAdd => mul_add += 1,
                _ => {}
            }
            let node: &mut Node = graph
                .nodes
                .iter_mut()
                .find(|node| node.id == add_id)
                .expect("recorded Add node");
            node.op = op;
            node.attrs = attrs;
            node.inputs = inputs;
            remove.insert(producer_id);
        }

        if add_relu + mul_add + matmul_bias == 0 {
            return Ok(PassResult::unchanged());
        }
        remove_nodes(graph, &remove);
        Ok(PassResult {
            changed: true,
            notes: vec![format!(
                "fused {add_relu} Add+ReLU, {mul_add} Mul+Add, {matmul_bias} MatMul+Bias"
            )],
        })
    }
}
