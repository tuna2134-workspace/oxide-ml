use std::collections::{HashMap, HashSet};

use oxide_ml_core::{NodeId, Scalar, ValueId};
use oxide_ml_graph::{Graph, Op};

use crate::pass::{GraphPass, PassContext, PassResult};
use crate::rewrite::remove_nodes;

/// Conservative operator fusion.
///
/// `Add + ReLU` becomes `AddRelu` only when the Add result has exactly one use,
/// that use is a ReLU, and the Add result is not a graph output. The fused node
/// takes over the ReLU output ValueId, so downstream handles and graph outputs
/// remain stable.
pub struct OperatorFusion;

impl<T: Scalar> GraphPass<T> for OperatorFusion {
    fn name(&self) -> &'static str {
        "fusion"
    }

    fn run(
        &self,
        graph: &mut Graph,
        _context: &PassContext<'_, T>,
    ) -> oxide_ml_core::Result<PassResult> {
        let mut uses: HashMap<ValueId, Vec<NodeId>> = HashMap::new();
        for node in graph.nodes() {
            for &input in &node.inputs {
                uses.entry(input).or_default().push(node.id);
            }
        }
        let outputs: HashSet<ValueId> = graph.outputs.iter().copied().collect();
        let mut fused_adds: Vec<(NodeId, NodeId, ValueId)> = Vec::new();
        let mut remove: HashSet<NodeId> = HashSet::new();

        for node in graph.nodes() {
            if !matches!(node.op, Op::Add) || node.outputs.len() != 1 {
                continue;
            }
            let sum = node.outputs[0];
            if outputs.contains(&sum) {
                continue;
            }
            let Some(consumers) = uses.get(&sum) else {
                continue;
            };
            if consumers.len() != 1 {
                continue;
            }
            let relu = graph.node(consumers[0])?;
            if !matches!(relu.op, Op::ReLU) || relu.inputs != [sum] || relu.outputs.len() != 1 {
                continue;
            }
            fused_adds.push((node.id, relu.id, relu.outputs[0]));
            remove.insert(relu.id);
        }

        if fused_adds.is_empty() {
            return Ok(PassResult::unchanged());
        }
        let fused = fused_adds.len();
        for (add_id, _relu_id, relu_out) in &fused_adds {
            let node = graph
                .nodes
                .iter_mut()
                .find(|node| node.id == *add_id)
                .expect("recorded Add node");
            node.op = Op::AddRelu;
            node.outputs[0] = *relu_out;
        }
        remove_nodes(graph, &remove);
        Ok(PassResult {
            changed: true,
            notes: vec![format!("fused {fused} Add+ReLU pair(s)")],
        })
    }
}
