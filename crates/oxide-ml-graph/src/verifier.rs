//! Explicit IR verifier entry point.
//!
//! `Graph::validate` contains the structural checks; this type adds an explicit
//! dependency-cycle traversal and returns a small report for compilers and
//! executors. Every execution/compilation path goes through it.

use std::collections::HashMap;

use oxide_ml_core::{Error, NodeId, Result, ValueId};

use crate::ir::{Graph, ValueKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerificationReport {
    pub values: usize,
    pub nodes: usize,
    pub parameters: usize,
    pub symbols: usize,
    pub inputs: usize,
    pub outputs: usize,
}

pub struct GraphVerifier;

impl GraphVerifier {
    pub fn verify(graph: &Graph) -> Result<VerificationReport> {
        graph.validate()?;
        detect_cycles(graph)?;
        Ok(VerificationReport {
            values: graph.values().len(),
            nodes: graph.nodes().len(),
            parameters: graph.parameters().len(),
            symbols: graph.symbols.len(),
            inputs: graph.inputs().len(),
            outputs: graph.outputs().len(),
        })
    }
}

fn detect_cycles(graph: &Graph) -> Result<()> {
    let mut definition: HashMap<ValueId, usize> = HashMap::new();
    for node in graph.nodes() {
        for &output in &node.outputs {
            definition.insert(output, node.id.0 as usize);
        }
    }

    // 0 = unvisited, 1 = active, 2 = done.
    let mut state = vec![0u8; graph.nodes().len()];
    for start in 0..graph.nodes().len() {
        visit(graph, &definition, &mut state, start)?;
    }
    Ok(())
}

fn visit(
    graph: &Graph,
    definition: &HashMap<ValueId, usize>,
    state: &mut [u8],
    node_index: usize,
) -> Result<()> {
    match state[node_index] {
        1 => {
            return Err(Error::invalid_graph(format!(
                "cycle detected at {}",
                NodeId(node_index as u32)
            )));
        }
        2 => return Ok(()),
        _ => {}
    }
    state[node_index] = 1;
    let node = &graph.nodes()[node_index];
    for &input in &node.inputs {
        if let Some(&producer) = definition.get(&input) {
            if producer == node_index {
                return Err(Error::invalid_graph(format!(
                    "{} consumes its own output {input}",
                    node.id
                )));
            }
            visit(graph, definition, state, producer)?;
        } else {
            // The value must be an input/parameter/constant defined outside a
            // node; structural validation has already checked that.
            let value = graph.value(input)?;
            if !matches!(
                value.kind,
                ValueKind::Input | ValueKind::Parameter(_) | ValueKind::Constant
            ) {
                return Err(Error::invalid_graph(format!(
                    "{} consumes {input}, which has no defining node",
                    node.id
                )));
            }
        }
    }
    state[node_index] = 2;
    Ok(())
}
