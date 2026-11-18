use std::collections::HashMap;

use crate::{DType, Shape, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LifetimePoint {
    Node(u32),
    End,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueLifetime {
    pub value: ValueId,
    pub defined_at: LifetimePoint,
    pub last_used_at: LifetimePoint,
    pub size_bytes: Option<usize>,
    pub dtype: DType,
    pub shape: Shape,
}

impl ValueLifetime {
    pub fn overlaps(&self, other: &ValueLifetime) -> bool {
        !(self.last_used_at < other.defined_at || other.last_used_at < self.defined_at)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferSlot {
    pub id: BufferId,
    pub size_bytes: usize,
    pub alignment: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferAssignment {
    pub slot: BufferId,
    pub offset: usize,
    pub size_bytes: usize,
    pub dtype: DType,
    pub shape: Shape,
}

#[derive(Clone, Debug, Default)]
pub struct MemoryPlan {
    pub slots: Vec<BufferSlot>,
    pub assignments: HashMap<ValueId, BufferAssignment>,
    pub lifetimes: Vec<ValueLifetime>,
    pub unplanned_values: Vec<ValueId>,
    pub total_value_bytes: usize,
}

impl MemoryPlan {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn assignment(&self, value: ValueId) -> Option<&BufferAssignment> {
        self.assignments.get(&value)
    }

    pub fn slot(&self, id: BufferId) -> Option<&BufferSlot> {
        self.slots.get(id.0 as usize)
    }

    pub fn peak_bytes(&self) -> usize {
        self.slots.iter().map(|slot| slot.size_bytes).sum()
    }

    pub fn allocation_count(&self) -> usize {
        self.slots.len()
    }

    pub fn reused_value_count(&self) -> usize {
        let used_slots = self
            .slots
            .iter()
            .filter(|slot| {
                self.assignments
                    .values()
                    .any(|assignment| assignment.slot == slot.id)
            })
            .count();
        self.assignments.len().saturating_sub(used_slots)
    }
}
