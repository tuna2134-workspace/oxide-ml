use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use oxide_ml_compiler::{
    ArenaContextId, ArenaDescriptor, ArenaKey, ArenaRegistry, ArenaRegistryError,
    ArenaRegistryStats, DEFAULT_MAX_ARENAS, MEMORY_PLAN_VERSION,
};
use oxide_ml_core::DType;
use oxide_ml_graph::Graph;

use crate::arena::CudaPlannedArena;
use crate::arena_key::{ContextIdentity, DeviceIdentity, ensure_context_device};
use crate::buffer::CudaBufferPool;
use crate::error::{CudaError, Result};
use crate::memplan::CudaMemoryPlan;

/// CUDA arena registry.
///
/// The registry key is `ArenaKey`; exact reuse additionally requires the
/// stored `ArenaDescriptor` (canonical graph, canonical plan, device, context,
/// dtype, version and the opaque CUDA layout bytes) to match. A graph that is
/// canonical-equal but has different raw ValueIds produces a different
/// `backend_layout` and therefore a separate arena.
pub struct CudaArenaRegistry {
    inner: ArenaRegistry<CudaPlannedArena>,
    context_tokens: Mutex<HashMap<ArenaContextId, ContextIdentity>>,
}

impl CudaArenaRegistry {
    pub fn new(max_arenas: usize) -> Self {
        CudaArenaRegistry {
            inner: ArenaRegistry::new(max_arenas),
            context_tokens: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or_create(
        &self,
        graph: &Graph,
        plan: &CudaMemoryPlan,
        pool: CudaBufferPool,
        device: &DeviceIdentity,
        context: &ContextIdentity,
        capacity: usize,
    ) -> Result<Arc<CudaPlannedArena>> {
        ensure_context_device(context.context(), device)?;

        let raw_graph_hash = graph
            .raw_identity_hash()
            .map_err(|error| CudaError::InvalidPlan(error.to_string()))?;
        if plan.source_graph_raw_hash() != Some(raw_graph_hash) {
            return Err(CudaError::InvalidPlan(
                "memory plan was produced from a different graph numbering".into(),
            ));
        }

        let canonical_graph = graph
            .canonicalize()
            .map_err(|error| CudaError::InvalidPlan(error.to_string()))?;
        let plan_hash = plan
            .plan_hash()
            .ok_or_else(|| CudaError::InvalidPlan("CUDA plan has no canonical hash".into()))?;
        let canonical_plan = plan
            .canonical_plan()
            .ok_or_else(|| CudaError::InvalidPlan("CUDA plan has no canonical form".into()))?;
        if canonical_plan.hash() != plan_hash {
            return Err(CudaError::InvalidPlan(
                "CUDA plan hash does not match its canonical form".into(),
            ));
        }

        let device_id = device.id().clone();
        let context_id = context.id();
        let dtype = DType::F32;
        let key = ArenaKey::new(
            canonical_graph.hash(),
            plan_hash,
            device_id.clone(),
            context_id,
            dtype,
            MEMORY_PLAN_VERSION,
        );
        let descriptor = ArenaDescriptor::new(
            canonical_graph,
            (**canonical_plan).clone(),
            device_id,
            context_id,
            dtype,
        )
        .with_backend_layout(plan.raw_layout_bytes());

        self.remember_context(context);

        self.inner
            .get_or_create(key, descriptor, capacity, || {
                CudaPlannedArena::new(plan.clone(), pool, capacity)
            })
            .map_err(map_registry_error)
    }

    /// Explicit key/descriptor lookup, used by collision and integration tests.
    pub fn get_or_create_explicit(
        &self,
        key: ArenaKey,
        descriptor: ArenaDescriptor,
        capacity: usize,
        plan: &CudaMemoryPlan,
        pool: CudaBufferPool,
    ) -> Result<Arc<CudaPlannedArena>> {
        self.inner
            .get_or_create(key, descriptor, capacity, || {
                CudaPlannedArena::new(plan.clone(), pool, capacity)
            })
            .map_err(map_registry_error)
    }

    fn remember_context(&self, context: &ContextIdentity) {
        let mut tokens = match self.context_tokens.lock() {
            Ok(tokens) => tokens,
            Err(poisoned) => poisoned.into_inner(),
        };
        tokens.insert(context.id(), context.clone());
    }

    pub fn stats(&self) -> ArenaRegistryStats {
        self.inner.stats()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn max_arenas(&self) -> usize {
        self.inner.max_arenas()
    }
}

impl Default for CudaArenaRegistry {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ARENAS)
    }
}

fn map_registry_error(error: ArenaRegistryError<CudaError>) -> CudaError {
    match error {
        ArenaRegistryError::Factory(error) => error,
        ArenaRegistryError::CapacityExceeded { limit, arena_count } => CudaError::ArenaBusy(
            format!("arena registry capacity limit {limit} reached ({arena_count} arenas)"),
        ),
        ArenaRegistryError::InvalidCapacity(capacity) => CudaError::InvalidPlan(format!(
            "arena slot-set capacity must be positive, got {capacity}"
        )),
    }
}
