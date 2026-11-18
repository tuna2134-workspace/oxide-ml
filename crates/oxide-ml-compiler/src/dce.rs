use std::collections::{HashMap, HashSet};

use oxide_ml_core::{NodeId, Scalar, ValueId};
use oxide_ml_graph::Graph;

use crate::pass::{GraphPass, PassContext, PassResult};
use crate::rewrite::remove_nodes;

/// Dead code elimination. Current IR operations are pure; there are no side
/// effects to preserve. Graph outputs are the roots.
pub struct DeadCodeElimination;

impl<T: Scalar> GraphPass<T> for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dce"
    }

    fn run(
        &self,
        graph: &mut Graph,
        _context: &PassContext<'_, T>,
    ) -> oxide_ml_core::Result<PassResult> {
        let mut producer: HashMap<ValueId, NodeId> = HashMap::new();
        for node in graph.nodes() {
            for &output in &node.outputs {
                producer.insert(output, node.id);
            }
        }
        let mut live_values: HashSet<ValueId> = graph.outputs.iter().copied().collect();
        let mut live_nodes: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<ValueId> = graph.outputs.clone();
        while let Some(value) = stack.pop() {
            if let Some(&node_id) = producer.get(&value) {
                if live_nodes.insert(node_id) {
                    let node = graph.node(node_id)?;
                    for &input in &node.inputs {
                        if live_values.insert(input) {
                            stack.push(input);
                        }
                    }
                }
            }
        }
        // A constant node may be a graph output but still must be retained.
        let remove: HashSet<NodeId> = graph
            .nodes()
            .iter()
            .filter(|node| !live_nodes.contains(&node.id))
            .map(|node| node.id)
            .collect();
        let removed = remove.len();
        remove_nodes(graph, &remove);

        // Current IR operations are pure. If a future op has side effects,
        // this pass must treat it as a root explicitly.

        Ok(PassResult {
            changed: removed > 0,
            notes: vec![format!("removed {removed} node(s)")],
        })
    }
}
