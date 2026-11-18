use std::collections::{HashMap, HashSet};

use oxide_ml_core::{NodeId, ValueId};
use oxide_ml_graph::Graph;

pub(crate) fn apply_replacements(graph: &mut Graph, replacements: &HashMap<ValueId, ValueId>) {
    if replacements.is_empty() {
        return;
    }
    let resolve = |mut value: ValueId| {
        let mut guard = 0;
        while let Some(&next) = replacements.get(&value) {
            if next == value || guard > replacements.len() {
                break;
            }
            value = next;
            guard += 1;
        }
        value
    };
    for node in &mut graph.nodes {
        for input in &mut node.inputs {
            *input = resolve(*input);
        }
    }
    for output in &mut graph.outputs {
        *output = resolve(*output);
    }
}

/// Remove nodes and restore dense `NodeId`s. ValueIds stay stable, which keeps
/// parameters, public API handles, and serialized value metadata valid.
pub(crate) fn remove_nodes(graph: &mut Graph, remove: &HashSet<NodeId>) {
    if remove.is_empty() {
        return;
    }
    graph.nodes.retain(|node| !remove.contains(&node.id));
    for (index, node) in graph.nodes.iter_mut().enumerate() {
        node.id = NodeId(index as u32);
    }
}
