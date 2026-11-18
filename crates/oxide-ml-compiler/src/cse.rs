use std::collections::{HashMap, HashSet};

use oxide_ml_core::{NodeId, Scalar, ValueId};
use oxide_ml_graph::Graph;

use crate::pass::{GraphPass, PassContext, PassResult};
use crate::rewrite::{apply_replacements, remove_nodes};

fn resolve(mut value: ValueId, replacements: &HashMap<ValueId, ValueId>) -> ValueId {
    let mut guard = 0;
    while let Some(&next) = replacements.get(&value) {
        if next == value || guard > replacements.len() {
            break;
        }
        value = next;
        guard += 1;
    }
    value
}

/// Common subexpression elimination.
///
/// Only nodes with a single output are considered, and the key includes the
/// complete op, attrs, and ordered inputs. No floating-point reassociation or
/// value numbering is performed.
pub struct CommonSubexpressionElimination;

impl<T: Scalar> GraphPass<T> for CommonSubexpressionElimination {
    fn name(&self) -> &'static str {
        "cse"
    }

    fn run(
        &self,
        graph: &mut Graph,
        _context: &PassContext<'_, T>,
    ) -> oxide_ml_core::Result<PassResult> {
        let mut seen: HashMap<String, ValueId> = HashMap::new();
        let mut replacements: HashMap<ValueId, ValueId> = HashMap::new();
        let mut remove: HashSet<NodeId> = HashSet::new();
        for node in graph.nodes() {
            if node.outputs.len() != 1 {
                continue;
            }
            let resolved_inputs = node
                .inputs
                .iter()
                .map(|input| resolve(*input, &replacements))
                .collect::<Vec<_>>();
            let key = serde_json::to_string(&(&node.op, &node.attrs, &resolved_inputs))?;
            if let Some(&existing) = seen.get(&key) {
                replacements.insert(node.outputs[0], existing);
                remove.insert(node.id);
            } else {
                seen.insert(key, node.outputs[0]);
            }
        }
        let removed = remove.len();
        apply_replacements(graph, &replacements);
        remove_nodes(graph, &remove);
        Ok(PassResult {
            changed: removed > 0,
            notes: vec![format!("merged {removed} duplicate node(s)")],
        })
    }
}
