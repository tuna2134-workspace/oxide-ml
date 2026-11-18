use std::sync::{Arc, Mutex};

use cuda_core::CudaStream;
use oxide_ml_graph::Graph;

use crate::buffer::{CudaBufferPool, CudaTensor};
use crate::error::{CudaError, Result};
use crate::event::CudaEvent;
use crate::memplan::CudaMemoryPlan;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaStats {
    pub arena_count: u64,
    pub active_arena_count: u64,
    pub slot_set_count: u64,
    pub active_slot_set_count: u64,
    pub pending_slot_set_count: u64,
    pub slot_count: u64,
    pub logical_bytes: u64,
    pub physical_reserved_bytes: u64,
    pub arena_reuse_count: u64,
    pub slot_set_reuse_count: u64,
    pub planned_slot_reuse_count: u64,
    pub allocation_count: u64,
    pub peak_reserved_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotSetState {
    Free,
    Leased,
    PendingCompletion,
    Reclaimable,
}

struct SlotSet {
    slots: Vec<CudaTensor>,
    events: Vec<Vec<Arc<CudaEvent>>>,
    streams: Vec<Option<Arc<CudaStream>>>,
    state: SlotSetState,
}

struct ArenaInner {
    plan: CudaMemoryPlan,
    pool: CudaBufferPool,
    sets: Mutex<Vec<SlotSet>>,
    capacity: usize,
    stats: Mutex<ArenaStats>,
}

#[derive(Clone)]
pub struct CudaPlannedArena {
    inner: Arc<ArenaInner>,
}

impl CudaPlannedArena {
    pub fn new(plan: CudaMemoryPlan, pool: CudaBufferPool, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(CudaError::InvalidPlan(
                "arena capacity must be positive".into(),
            ));
        }
        let logical_bytes = plan.logical_bytes as u64;
        let slot_count = plan.slot_count() as u64;
        let arena = CudaPlannedArena {
            inner: Arc::new(ArenaInner {
                plan,
                pool,
                sets: Mutex::new(Vec::new()),
                capacity,
                stats: Mutex::new(ArenaStats {
                    arena_count: 1,
                    active_arena_count: 1,
                    slot_count,
                    logical_bytes,
                    ..ArenaStats::default()
                }),
            }),
        };
        Ok(arena)
    }

    pub fn plan(&self) -> &CudaMemoryPlan {
        &self.inner.plan
    }

    pub fn capacity(&self) -> usize {
        self.inner.capacity
    }

    pub fn lease(&self, stream: &Arc<CudaStream>) -> Result<SlotSetLease> {
        self.poll()?;
        let mut sets = self
            .inner
            .sets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut index = sets.iter().position(|set| set.state == SlotSetState::Free);
        let reused_existing = index.is_some();
        if index.is_none() && sets.len() < self.inner.capacity {
            sets.push(self.allocate_set()?);
            index = Some(sets.len() - 1);
        }
        if index.is_none() {
            // Reuse a pending set only by enqueueing event waits, never by
            // silently overwriting an in-flight buffer.
            let pending = sets.iter().position(|set| {
                set.state == SlotSetState::PendingCompletion
                    || set.state == SlotSetState::Reclaimable
            });
            if let Some(candidate) = pending {
                index = Some(candidate);
            }
        }
        let index = index.ok_or_else(|| {
            CudaError::ArenaBusy(format!("all {} slot sets are leased", sets.len()))
        })?;
        let set = &mut sets[index];
        for (slot_index, events) in set.events.iter().enumerate() {
            if set.streams[slot_index]
                .as_ref()
                .is_some_and(|previous| Arc::ptr_eq(previous, stream))
            {
                continue;
            }
            for event in events {
                event.wait_on(stream)?;
            }
        }
        let slots = std::mem::take(&mut set.slots);
        let events = std::mem::take(&mut set.events);
        let streams = std::mem::take(&mut set.streams);
        set.state = SlotSetState::Leased;
        set.events = vec![Vec::new(); slots.len()];
        set.streams = vec![None; slots.len()];
        set.slots = Vec::new();
        let mut stats = self
            .inner
            .stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stats.active_slot_set_count += 1;
        stats.slot_set_reuse_count += u64::from(reused_existing);
        drop(stats);
        drop(sets);
        Ok(SlotSetLease {
            arena: self.inner.clone(),
            index,
            slots: Some(slots),
            events: Some(events),
            streams: Some(streams),
            finished: false,
        })
    }

    fn allocate_set(&self) -> Result<SlotSet> {
        let mut slots = Vec::with_capacity(self.inner.plan.slots.len());
        for slot in &self.inner.plan.slots {
            let storage = self.inner.pool.acquire(slot.elements_f32)?;
            slots.push(CudaTensor::from_storage(
                vec![slot.elements_f32],
                Arc::new(storage),
            )?);
        }
        let mut stats = self
            .inner
            .stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stats.slot_set_count += 1;
        stats.allocation_count += slots.len() as u64;
        stats.physical_reserved_bytes += slots
            .iter()
            .map(|slot| slot.capacity() as u64 * 4)
            .sum::<u64>();
        stats.peak_reserved_bytes = stats.peak_reserved_bytes.max(stats.physical_reserved_bytes);
        drop(stats);
        let len = slots.len();
        Ok(SlotSet {
            slots,
            events: vec![Vec::new(); len],
            streams: vec![None; len],
            state: SlotSetState::Free,
        })
    }

    pub fn poll(&self) -> Result<usize> {
        let mut sets = self
            .inner
            .sets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut freed = 0;
        for set in &mut *sets {
            if set.state != SlotSetState::PendingCompletion {
                continue;
            }
            let complete = set
                .events
                .iter()
                .flatten()
                .all(|event| event.is_complete().unwrap_or(false));
            if complete {
                set.state = SlotSetState::Reclaimable;
                set.state = SlotSetState::Free;
                freed += 1;
            }
        }
        if freed > 0 {
            let mut stats = self
                .inner
                .stats
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.active_slot_set_count = stats.active_slot_set_count.saturating_sub(freed as u64);
        }
        Ok(freed)
    }

    pub fn stats(&self) -> ArenaStats {
        let _ = self.poll();
        *self
            .inner
            .stats
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

pub struct SlotSetLease {
    arena: Arc<ArenaInner>,
    index: usize,
    slots: Option<Vec<CudaTensor>>,
    events: Option<Vec<Vec<Arc<CudaEvent>>>>,
    streams: Option<Vec<Option<Arc<CudaStream>>>>,
    finished: bool,
}

impl SlotSetLease {
    pub fn take(
        &mut self,
    ) -> Result<(
        Vec<CudaTensor>,
        Vec<Vec<Arc<CudaEvent>>>,
        Vec<Option<Arc<CudaStream>>>,
    )> {
        let slots = self
            .slots
            .take()
            .ok_or_else(|| CudaError::ArenaBusy("lease already finished".into()))?;
        let events = self
            .events
            .take()
            .ok_or_else(|| CudaError::ArenaBusy("lease already finished".into()))?;
        let streams = self
            .streams
            .take()
            .ok_or_else(|| CudaError::ArenaBusy("lease already finished".into()))?;
        Ok((slots, events, streams))
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn finish(
        &mut self,
        slots: Vec<CudaTensor>,
        events: Vec<Vec<Arc<CudaEvent>>>,
        streams: Vec<Option<Arc<CudaStream>>>,
    ) -> Result<()> {
        self.finish_or_return(slots, events, streams)
            .map_err(|_| CudaError::ArenaBusy("lease already finished".into()))
    }

    /// Finishes the lease without dropping the resources on error.
    ///
    /// This is used by `ExecutionResources::Drop`; if bookkeeping fails, the
    /// caller can keep the slot buffers alive instead of prematurely freeing
    /// memory that the GPU may still be using.
    pub fn finish_or_return(
        &mut self,
        slots: Vec<CudaTensor>,
        events: Vec<Vec<Arc<CudaEvent>>>,
        streams: Vec<Option<Arc<CudaStream>>>,
    ) -> std::result::Result<
        (),
        (
            Vec<CudaTensor>,
            Vec<Vec<Arc<CudaEvent>>>,
            Vec<Option<Arc<CudaStream>>>,
        ),
    > {
        if self.finished {
            return Err((slots, events, streams));
        }
        let mut sets = self
            .arena
            .sets
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let Some(set) = sets.get_mut(self.index) else {
            return Err((slots, events, streams));
        };
        set.slots = slots;
        set.events = events;
        set.streams = streams;
        set.state = SlotSetState::PendingCompletion;
        self.finished = true;
        Ok(())
    }
}

impl Drop for SlotSetLease {
    fn drop(&mut self) {
        if self.slots.is_some() {
            // Lease dropped without finish: never return buffers to Free here.
            if let Some(slots) = self.slots.take() {
                let mut sets = self
                    .arena
                    .sets
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(set) = sets.get_mut(self.index) {
                    set.slots = slots;
                    set.events = self.events.take().unwrap_or_default();
                    set.streams = self.streams.take().unwrap_or_default();
                    set.state = SlotSetState::PendingCompletion;
                }
            }
        }
    }
}

pub fn arena_plan_for_graph(graph: &Graph, plan: &CudaMemoryPlan) -> Result<CudaMemoryPlan> {
    let _ = graph;
    Ok(plan.clone())
}
