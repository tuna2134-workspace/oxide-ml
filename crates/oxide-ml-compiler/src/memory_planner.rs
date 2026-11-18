use std::collections::HashMap;

use oxide_ml_core::{
    BufferAssignment, BufferId, BufferSlot, LifetimePoint, MemoryPlan, Result, ValueId,
    ValueLifetime,
};
use oxide_ml_graph::{Graph, ValueKind};

#[derive(Clone, Debug)]
pub struct MemoryPlanner {
    pub alignment: usize,
}

impl Default for MemoryPlanner {
    fn default() -> Self {
        MemoryPlanner { alignment: 64 }
    }
}

impl MemoryPlanner {
    pub fn new(alignment: usize) -> Result<Self> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(oxide_ml_core::Error::invalid_shape(
                "MemoryPlanner",
                "alignment must be a non-zero power of two",
            ));
        }
        Ok(MemoryPlanner { alignment })
    }

    pub fn plan(&self, graph: &Graph) -> Result<MemoryPlan> {
        let node_count = graph.nodes().len();
        let mut definition: HashMap<ValueId, usize> = HashMap::new();
        for node in graph.nodes() {
            for &output in &node.outputs {
                definition.insert(output, node.id.0 as usize);
            }
        }
        let mut last_use: HashMap<ValueId, usize> = definition.clone();
        for node in graph.nodes() {
            for &input in &node.inputs {
                last_use.insert(input, node.id.0 as usize);
            }
        }
        let output_set: std::collections::HashSet<ValueId> =
            graph.outputs.iter().copied().collect();
        for &output in &graph.outputs {
            if definition.contains_key(&output) {
                last_use.insert(output, node_count);
            }
        }

        let mut lifetimes = Vec::with_capacity(definition.len());
        let mut candidates = Vec::new();
        let mut total_value_bytes = 0usize;
        let mut unplanned_values = Vec::new();
        for (&value, &defined) in &definition {
            let info = graph.value(value)?;
            if matches!(info.kind, ValueKind::Input | ValueKind::Parameter(_)) {
                continue;
            }
            let size = info
                .shape
                .numel_static()
                .map(|count| count * info.dtype.size_in_bytes());
            let used = last_use.get(&value).copied().unwrap_or(defined);
            let lifetime = ValueLifetime {
                value,
                defined_at: LifetimePoint::Node(defined as u32),
                last_used_at: if output_set.contains(&value) || used == node_count {
                    LifetimePoint::End
                } else {
                    LifetimePoint::Node(used as u32)
                },
                size_bytes: size,
                dtype: info.dtype,
                shape: info.shape.clone(),
            };
            if let Some(bytes) = size {
                total_value_bytes += bytes;
                candidates.push((value, defined, used, bytes, info.dtype, info.shape.clone()));
            } else {
                unplanned_values.push(value);
            }
            lifetimes.push(lifetime);
        }
        lifetimes.sort_by_key(|lifetime| lifetime.value);

        // Largest-first greedy interval allocation. A slot is reusable when its
        // previous lifetime ended before this value is defined. Offsets are
        // zero for now: each value owns a whole aligned slot.
        candidates.sort_by(|a, b| b.3.cmp(&a.3).then(a.1.cmp(&b.1)).then(a.0.cmp(&b.0)));
        let mut slots: Vec<BufferSlot> = Vec::new();
        let mut free_at: Vec<usize> = Vec::new();
        let mut assignments = HashMap::new();
        for (value, defined, used, bytes, dtype, shape) in candidates {
            let required = align_up(bytes.max(1), self.alignment);
            let mut choice = None;
            for index in 0..slots.len() {
                if free_at[index] <= defined
                    && slots[index].size_bytes >= required
                    && choice
                        .map(|current: usize| slots[index].size_bytes < slots[current].size_bytes)
                        .unwrap_or(true)
                {
                    choice = Some(index);
                }
            }
            let slot_id = match choice {
                Some(index) => {
                    free_at[index] = used + 1;
                    slots[index].id
                }
                None => {
                    let id = BufferId(slots.len() as u32);
                    slots.push(BufferSlot {
                        id,
                        size_bytes: required,
                        alignment: self.alignment,
                    });
                    free_at.push(used + 1);
                    id
                }
            };
            assignments.insert(
                value,
                BufferAssignment {
                    slot: slot_id,
                    offset: 0,
                    size_bytes: bytes,
                    dtype,
                    shape,
                },
            );
        }

        Ok(MemoryPlan {
            slots,
            assignments,
            lifetimes,
            unplanned_values,
            total_value_bytes,
        })
    }
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use oxide_ml_core::{DType, Shape};
    use oxide_ml_graph::GraphBuilder;

    use super::*;

    #[test]
    fn reuses_buffer_for_disjoint_lifetimes() {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::from_static([4]), DType::F32);
        let a = builder.relu(x).unwrap();
        let b = builder.sigmoid(a).unwrap();
        let c = builder.tanh(b).unwrap();
        let graph = builder.finish(&[c]).unwrap();
        let plan = MemoryPlanner::default().plan(&graph).unwrap();
        assert!(!plan.slots.is_empty());
        assert!(plan.assignments.len() >= 3);
        assert!(plan.peak_bytes() < plan.total_value_bytes * 3);
    }
}
