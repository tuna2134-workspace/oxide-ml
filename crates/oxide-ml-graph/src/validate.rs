use std::collections::HashSet;

use oxide_ml_core::{Error, NodeId, ParameterId, Result, ValueId};

use crate::infer::infer_outputs;
use crate::ir::{Graph, IR_VERSION, SERIALIZATION_FORMAT_VERSION, ValueKind};

impl Graph {
    /// Structural and type validation.
    ///
    /// This is intentionally stricter than "the graph can be serialized": every
    /// node is re-inferred from its operands, so a corrupted or hand-edited
    /// serialized graph cannot silently execute with wrong shapes.
    pub fn validate(&self) -> Result<()> {
        if self.format_version != SERIALIZATION_FORMAT_VERSION {
            return Err(Error::Serialization(format!(
                "unsupported graph format version {} (this build supports {})",
                self.format_version, SERIALIZATION_FORMAT_VERSION
            )));
        }
        if self.ir_version != IR_VERSION {
            return Err(Error::Serialization(format!(
                "unsupported graph IR version {} (this build supports {})",
                self.ir_version, IR_VERSION
            )));
        }

        let mut value_names = HashSet::new();
        for (index, value) in self.values.iter().enumerate() {
            if value.id.0 as usize != index {
                return Err(Error::invalid_graph(format!(
                    "value table is not dense: slot {index} contains {}",
                    value.id
                )));
            }
            if let Some(name) = &value.name
                && !value_names.insert(name.clone())
            {
                return Err(Error::invalid_graph(format!(
                    "duplicate value name {name:?}"
                )));
            }
        }

        let mut symbol_names = HashSet::new();
        for (index, symbol) in self.symbols.iter().enumerate() {
            if symbol.id.0 as usize != index {
                return Err(Error::invalid_graph(format!(
                    "symbol table is not dense at slot {index}"
                )));
            }
            if symbol.name.is_empty() {
                return Err(Error::invalid_graph("symbol name must not be empty"));
            }
            if !symbol_names.insert(symbol.name.clone()) {
                return Err(Error::invalid_graph(format!(
                    "duplicate symbol name {:?}",
                    symbol.name
                )));
            }
        }

        let mut parameter_names = HashSet::new();
        for (index, parameter) in self.parameters.iter().enumerate() {
            if parameter.id.0 as usize != index {
                return Err(Error::invalid_graph(format!(
                    "parameter table is not dense at slot {index}"
                )));
            }
            if !parameter_names.insert(parameter.name.clone()) {
                return Err(Error::invalid_graph(format!(
                    "duplicate parameter name {:?}",
                    parameter.name
                )));
            }
            let value = self.value(parameter.value)?;
            if value.shape != parameter.shape || value.dtype != parameter.dtype {
                return Err(Error::invalid_graph(format!(
                    "parameter {:?} metadata disagrees with its graph value {}",
                    parameter.name, parameter.value
                )));
            }
            match value.kind {
                ValueKind::Parameter(id) if id == parameter.id => {}
                _ => {
                    return Err(Error::invalid_graph(format!(
                        "parameter {:?} is not referenced by a Parameter value",
                        parameter.name
                    )));
                }
            }
            if parameter.shape.numel_static().is_none() {
                return Err(Error::invalid_graph(format!(
                    "parameter {:?} has a non-static shape {}",
                    parameter.name, parameter.shape
                )));
            }
        }

        let mut defined: HashSet<ValueId> = HashSet::new();
        let mut input_names = HashSet::new();
        for &input in &self.inputs {
            let value = self.value(input)?;
            if let Some(name) = &value.name
                && !input_names.insert(name.clone())
            {
                return Err(Error::invalid_graph(format!(
                    "duplicate input name {name:?}"
                )));
            }
            if !matches!(value.kind, ValueKind::Input) {
                return Err(Error::invalid_graph(format!(
                    "graph input {input} does not have Input value kind"
                )));
            }
            if !defined.insert(input) {
                return Err(Error::invalid_graph(format!(
                    "graph input {input} appears more than once"
                )));
            }
        }
        for parameter in &self.parameters {
            defined.insert(parameter.value);
        }

        for (node_index, node) in self.nodes.iter().enumerate() {
            if node.id != NodeId(node_index as u32) {
                return Err(Error::invalid_graph(format!(
                    "node table is not dense at slot {node_index}: {}",
                    node.id
                )));
            }
            let mut input_infos = Vec::with_capacity(node.inputs.len());
            for &input in &node.inputs {
                if !defined.contains(&input) {
                    return Err(Error::invalid_graph(format!(
                        "{} uses {input} before it is defined",
                        node.id
                    )));
                }
                input_infos.push(self.value(input)?.clone());
            }
            let inferred = infer_outputs(&node.op, &node.attrs, &input_infos)?;
            if inferred.len() != node.outputs.len() {
                return Err(Error::invalid_graph(format!(
                    "{} produced {} outputs, expected {}",
                    node.id,
                    node.outputs.len(),
                    inferred.len()
                )));
            }
            for (output, expected) in node.outputs.iter().zip(inferred) {
                if !defined.insert(*output) {
                    return Err(Error::invalid_graph(format!(
                        "value {output} is defined more than once"
                    )));
                }
                let value = self.value(*output)?;
                if value.shape != expected.shape {
                    return Err(Error::invalid_graph(format!(
                        "{} output {output} has shape {}, inference says {}",
                        node.id, value.shape, expected.shape
                    )));
                }
                if value.dtype != expected.dtype {
                    return Err(Error::invalid_graph(format!(
                        "{} output {output} has dtype {}, inference says {}",
                        node.id, value.dtype, expected.dtype
                    )));
                }
            }
        }

        if self.outputs.is_empty() {
            return Err(Error::invalid_graph(
                "graph must declare at least one output",
            ));
        }
        for &output in &self.outputs {
            if !defined.contains(&output) {
                return Err(Error::invalid_graph(format!(
                    "declared output {output} is not defined"
                )));
            }
        }

        // Reject parameter ids that are out of range even if no ParameterInfo
        // points at them; this catches malformed deserialized data early.
        for value in &self.values {
            if let ValueKind::Parameter(id) = value.kind
                && self.parameters.get(id.0 as usize).map(|p| p.id) != Some(ParameterId(id.0))
            {
                return Err(Error::invalid_graph(format!(
                    "value {} references missing parameter {id}",
                    value.id
                )));
            }
        }

        Ok(())
    }
}
