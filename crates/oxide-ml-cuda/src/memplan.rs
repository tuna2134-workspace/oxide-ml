use std::collections::HashMap;
use std::sync::Arc;

use oxide_ml_compiler::{CanonicalPlan, MEMORY_PLAN_VERSION, PlanHash, canonicalize_memory_plan};
use oxide_ml_core::{DType, MemoryPlan, ValueId};
use oxide_ml_graph::Graph;

use crate::error::{CudaError, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaSlotPlan {
    pub id: u32,
    pub size_bytes: usize,
    pub alignment: usize,
    pub elements_f32: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaSlotAssignment {
    pub slot: u32,
    pub offset: usize,
    pub size_bytes: usize,
    pub shape: Vec<usize>,
}

/// CUDA view of a compiler `MemoryPlan`.
///
/// The source plan and its graph-relative canonical plan are retained for
/// arena identity. `CudaMemoryPlan` itself remains the execution layout and
/// keeps raw `ValueId` assignments.
#[derive(Clone, Debug, Default)]
pub struct CudaMemoryPlan {
    pub slots: Vec<CudaSlotPlan>,
    pub assignments: HashMap<ValueId, CudaSlotAssignment>,
    pub unplanned: Vec<ValueId>,
    pub logical_bytes: usize,
    source_plan: Option<Arc<MemoryPlan>>,
    canonical_plan: Option<Arc<CanonicalPlan>>,
    plan_hash: Option<PlanHash>,
    source_graph_raw_hash: Option<[u8; 32]>,
    /// Symbol -> maximum runtime dimension for a bounded dynamic plan.
    capacity_bindings: Vec<(u32, usize)>,
}

impl CudaMemoryPlan {
    pub fn from_memory_plan(graph: &Graph, plan: &MemoryPlan) -> Result<Self> {
        let source_graph_raw_hash = graph
            .raw_identity_hash()
            .map_err(|error| CudaError::InvalidPlan(error.to_string()))?;
        let canonical_plan = canonicalize_memory_plan(graph, plan)
            .map_err(|error| CudaError::InvalidPlan(error.to_string()))?;
        let plan_hash = canonical_plan.hash();

        let mut slots = Vec::with_capacity(plan.slots.len());
        for (index, slot) in plan.slots.iter().enumerate() {
            if slot.id.0 as usize != index {
                return Err(CudaError::InvalidPlan(format!(
                    "slot table is not dense at {index}"
                )));
            }
            if slot.alignment == 0 || !slot.alignment.is_power_of_two() {
                return Err(CudaError::InvalidPlan(format!(
                    "slot {} has invalid alignment {}",
                    slot.id.0, slot.alignment
                )));
            }
            if slot.size_bytes % 4 != 0 {
                return Err(CudaError::InvalidPlan(format!(
                    "slot {} size {} is not a multiple of F32",
                    slot.id.0, slot.size_bytes
                )));
            }
            slots.push(CudaSlotPlan {
                id: slot.id.0,
                size_bytes: slot.size_bytes,
                alignment: slot.alignment,
                elements_f32: slot.size_bytes / 4,
            });
        }

        let mut assignments = HashMap::new();
        let mut logical_bytes = 0usize;
        for (&value, assignment) in &plan.assignments {
            let info = graph.value(value)?;
            if info.dtype != DType::F32 {
                return Err(CudaError::Unsupported(format!(
                    "planned value {value} has dtype {}; CUDA adapter supports F32",
                    info.dtype
                )));
            }
            let shape = info.shape.static_dims().ok_or_else(|| {
                CudaError::InvalidPlan(format!(
                    "planned value {value} has a dynamic shape and must fall back"
                ))
            })?;
            let slot = slots.get(assignment.slot.0 as usize).ok_or_else(|| {
                CudaError::InvalidPlan(format!(
                    "value {value} references missing slot {}",
                    assignment.slot.0
                ))
            })?;
            let bytes = shape.iter().product::<usize>() * 4;
            if bytes > slot.size_bytes {
                return Err(CudaError::InvalidPlan(format!(
                    "value {value} needs {bytes} bytes but slot {} has {}",
                    slot.id, slot.size_bytes
                )));
            }
            if assignment.offset + bytes > slot.size_bytes {
                return Err(CudaError::InvalidPlan(format!(
                    "value {value} overflows slot {} at offset {}",
                    slot.id, assignment.offset
                )));
            }
            logical_bytes += bytes;
            assignments.insert(
                value,
                CudaSlotAssignment {
                    slot: slot.id,
                    offset: assignment.offset,
                    size_bytes: bytes,
                    shape,
                },
            );
        }

        let assignments = crate::memplan::filter_cuda_assignments(graph, assignments)?;
        Ok(CudaMemoryPlan {
            slots,
            assignments,
            unplanned: plan.unplanned_values.clone(),
            logical_bytes,
            source_plan: Some(Arc::new(plan.clone())),
            canonical_plan: Some(Arc::new(canonical_plan)),
            plan_hash: Some(plan_hash),
            source_graph_raw_hash: Some(source_graph_raw_hash),
            capacity_bindings: Vec::new(),
        })
    }

    /// Builds a plan from a graph specialized at maximum capacity.
    ///
    /// The plan's slot sizes are the maximum capacity; runtime execution must
    /// validate its actual required bytes against these capacities.
    pub fn from_bounded_memory_plan(
        graph: &Graph,
        plan: &MemoryPlan,
        bindings: &std::collections::HashMap<oxide_ml_core::SymbolId, usize>,
    ) -> Result<Self> {
        let mut bounded = Self::from_memory_plan(graph, plan)?;
        let mut capacity = bindings
            .iter()
            .map(|(symbol, value)| (symbol.0, *value))
            .collect::<Vec<_>>();
        capacity.sort_unstable();
        bounded.capacity_bindings = capacity;
        Ok(bounded)
    }

    pub fn capacity_bindings(&self) -> &[(u32, usize)] {
        &self.capacity_bindings
    }

    pub fn is_bounded_dynamic(&self) -> bool {
        !self.capacity_bindings.is_empty()
    }

    pub fn max_capacity_bytes(&self) -> usize {
        self.slots.iter().map(|slot| slot.size_bytes).sum()
    }

    pub fn assignment(&self, value: ValueId) -> Option<&CudaSlotAssignment> {
        self.assignments.get(&value)
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn plan_hash(&self) -> Option<PlanHash> {
        self.plan_hash
    }

    pub fn memory_plan_version(&self) -> u32 {
        MEMORY_PLAN_VERSION
    }

    pub fn canonical_plan(&self) -> Option<&Arc<CanonicalPlan>> {
        self.canonical_plan.as_ref()
    }

    pub fn source_plan(&self) -> Option<&Arc<MemoryPlan>> {
        self.source_plan.as_ref()
    }

    /// Raw graph numbering this plan was built from.
    pub fn source_graph_raw_hash(&self) -> Option<[u8; 32]> {
        self.source_graph_raw_hash
    }

    /// Exact execution-layout comparison including raw ValueId bindings.
    ///
    /// Arena reuse across two graphs that are canonical-equal but keep
    /// different ValueId numbering is unsafe because the persistent arena
    /// stores ValueId-keyed assignments. This check forces a separate arena in
    /// that case.
    pub fn layout_eq(&self, other: &CudaMemoryPlan) -> bool {
        if self.slots != other.slots
            || self.logical_bytes != other.logical_bytes
            || self.unplanned != other.unplanned
            || self.capacity_bindings != other.capacity_bindings
        {
            return false;
        }
        if self.assignments.len() != other.assignments.len() {
            return false;
        }
        let mut left = self.assignments.iter().collect::<Vec<_>>();
        let mut right = other.assignments.iter().collect::<Vec<_>>();
        left.sort_by_key(|(value, _)| value.0);
        right.sort_by_key(|(value, _)| value.0);
        left == right
    }

    /// Deterministic opaque encoding used in `ArenaDescriptor.backend_layout`.
    pub fn raw_layout_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.slots.len() as u64).to_le_bytes());
        for slot in &self.slots {
            out.extend_from_slice(&slot.id.to_le_bytes());
            out.extend_from_slice(&(slot.size_bytes as u64).to_le_bytes());
            out.extend_from_slice(&(slot.alignment as u64).to_le_bytes());
            out.extend_from_slice(&(slot.elements_f32 as u64).to_le_bytes());
        }
        let mut assignments = self.assignments.iter().collect::<Vec<_>>();
        assignments.sort_by_key(|(value, _)| value.0);
        out.extend_from_slice(&(assignments.len() as u64).to_le_bytes());
        for (value, assignment) in assignments {
            out.extend_from_slice(&value.0.to_le_bytes());
            out.extend_from_slice(&assignment.slot.to_le_bytes());
            out.extend_from_slice(&(assignment.size_bytes as u64).to_le_bytes());
            out.extend_from_slice(&(assignment.shape.len() as u64).to_le_bytes());
            for dim in &assignment.shape {
                out.extend_from_slice(&(*dim as u64).to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.logical_bytes as u64).to_le_bytes());
        out
    }
}

fn filter_cuda_assignments(
    _graph: &Graph,
    assignments: HashMap<ValueId, CudaSlotAssignment>,
) -> Result<HashMap<ValueId, CudaSlotAssignment>> {
    Ok(assignments)
}

/// Memory accounting for a planned execution. Counters are intentionally
/// separated: pool capacity, checked-out bytes, pending bytes, and the
/// compiler's logical plan are different quantities.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CudaMemoryStats {
    pub allocation_count: u64,
    pub reuse_count: u64,
    pub release_count: u64,
    pub planned_slot_count: u64,
    pub planned_reuse_count: u64,
    pub fallback_allocation_count: u64,
    pub pending_release_count: u64,
    pub pending_release_bytes: u64,
    pub pool_reserved_bytes: u64,
    pub pool_in_use_bytes: u64,
    pub peak_pool_in_use_bytes: u64,
    pub memory_plan_logical_bytes: u64,
}

impl From<crate::buffer::PoolStats> for CudaMemoryStats {
    fn from(stats: crate::buffer::PoolStats) -> Self {
        CudaMemoryStats {
            allocation_count: stats.allocations as u64,
            reuse_count: stats.reuses as u64,
            release_count: stats.released as u64,
            planned_slot_count: stats.planned_slot_count as u64,
            planned_reuse_count: stats.planned_reuse_count as u64,
            fallback_allocation_count: stats.fallback_allocation_count as u64,
            pending_release_count: stats.pending_release_count as u64,
            pending_release_bytes: stats.pending_release_bytes as u64,
            pool_reserved_bytes: stats.reserved_bytes as u64,
            pool_in_use_bytes: stats.current_bytes as u64,
            peak_pool_in_use_bytes: stats.peak_bytes as u64,
            memory_plan_logical_bytes: stats.memory_plan_logical_bytes as u64,
        }
    }
}
