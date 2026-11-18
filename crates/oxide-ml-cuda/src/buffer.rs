use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use oxide_ml_event_safety::{
    ContextId, EventDiagnostics, EventError, EventState, ReclaimDecision, evaluate_completion,
    lock_recover,
};

use crate::error::{CudaError, Result};
use crate::event::CudaEvent;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Number of driver allocations made by the pool.
    pub allocations: usize,
    /// Number of successful free-list reuses.
    pub reuses: usize,
    /// Buffers currently leased to callers.
    pub current_bytes: usize,
    /// Peak of `current_bytes`.
    pub peak_bytes: usize,
    /// Buffers returned to the free list.
    pub released: usize,
    /// Buffers held in the pending-release queue.
    pub pending_release_count: usize,
    /// Bytes held in the pending-release queue.
    pub pending_release_bytes: usize,
    /// Bytes retained by the pool (free + leased + pending + quarantined).
    pub reserved_bytes: usize,
    /// Peak of `reserved_bytes`.
    pub peak_reserved_bytes: usize,
    /// Planned slots supplied by the latest planned execution.
    pub planned_slot_count: usize,
    /// Planned slot reuses in the latest planned execution.
    pub planned_reuse_count: usize,
    /// Values that fell back to ordinary allocation in the latest plan.
    pub fallback_allocation_count: usize,
    /// Compiler logical bytes for the latest plan.
    pub memory_plan_logical_bytes: usize,
    /// Buffers that could not be proven safe to reuse.
    pub quarantined_count: usize,
    /// Bytes held by quarantined buffers.
    pub quarantined_bytes: usize,
    /// Cumulative number of buffers quarantined.
    pub quarantine_total: usize,
    /// Mutex poisoning recoveries observed by buffer bookkeeping.
    pub lock_poison_recoveries: usize,
}

/// Buffer quarantine statistics (Phase 7F-13).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferPoolStats {
    pub quarantined_count: usize,
    pub quarantined_bytes: usize,
    pub quarantine_total: usize,
}

/// Why a buffer was quarantined.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferQuarantineRecord {
    pub len: usize,
    pub bytes: usize,
    pub event_states: Vec<EventState>,
    pub event_errors: Vec<Option<EventError>>,
    pub error: Option<EventError>,
}

struct Pending {
    len: usize,
    buffer: DeviceBuffer<f32>,
    events: Vec<Arc<CudaEvent>>,
}

struct QuarantinedBuffer {
    len: usize,
    /// Kept alive (never reused) so the quarantined allocation cannot be
    /// handed back to a kernel.
    _buffer: DeviceBuffer<f32>,
    record: BufferQuarantineRecord,
}

#[derive(Default)]
struct PoolState {
    free: HashMap<usize, Vec<DeviceBuffer<f32>>>,
    pending: Vec<Pending>,
    quarantine: Vec<QuarantinedBuffer>,
    stats: PoolStats,
}

impl PoolState {
    fn refresh_derived_stats(&mut self) {
        self.stats.pending_release_count = self.pending.len();
        self.stats.pending_release_bytes = self.pending.iter().map(|entry| entry.len * 4).sum();
        self.stats.quarantined_count = self.quarantine.len();
        self.stats.quarantined_bytes = self.quarantine.iter().map(|entry| entry.len * 4).sum();
    }
}

struct PoolInner {
    context: Arc<CudaContext>,
    context_id: ContextId,
    stream: Arc<CudaStream>,
    state: Mutex<PoolState>,
    diagnostics: Arc<EventDiagnostics>,
}

impl PoolInner {
    fn quarantine(
        &self,
        len: usize,
        buffer: DeviceBuffer<f32>,
        events: Vec<Arc<CudaEvent>>,
        error: Option<EventError>,
        decrement_current_bytes: bool,
    ) {
        let event_states = events.iter().map(|event| event.state()).collect();
        let event_errors = events
            .iter()
            .map(|event| event.last_error())
            .collect::<Vec<_>>();
        let record = BufferQuarantineRecord {
            len,
            bytes: len * 4,
            event_states,
            event_errors,
            error,
        };
        let mut state = lock_recover(&self.state, &self.diagnostics);
        if decrement_current_bytes {
            state.stats.current_bytes = state.stats.current_bytes.saturating_sub(len * 4);
        }
        state.stats.quarantine_total = state.stats.quarantine_total.saturating_add(1);
        state.quarantine.push(QuarantinedBuffer {
            len,
            _buffer: buffer,
            record,
        });
        state.refresh_derived_stats();
    }

    fn poll(&self) {
        let pending = {
            let mut state = lock_recover(&self.state, &self.diagnostics);
            std::mem::take(&mut state.pending)
        };
        if pending.is_empty() {
            let mut state = lock_recover(&self.state, &self.diagnostics);
            state.refresh_derived_stats();
            return;
        }

        let mut ready = Vec::new();
        let mut remaining = Vec::new();
        for mut entry in pending {
            let verdict = evaluate_completion(&entry.events);
            match verdict.decision {
                ReclaimDecision::Reclaim => {
                    ready.push((entry.len, entry.buffer));
                    entry.events.clear();
                }
                ReclaimDecision::Pending => remaining.push(entry),
                ReclaimDecision::Quarantine => {
                    let events = std::mem::take(&mut entry.events);
                    self.quarantine(entry.len, entry.buffer, events, verdict.error, false);
                }
            }
        }

        let mut state = lock_recover(&self.state, &self.diagnostics);
        state.pending.extend(remaining);
        for (len, buffer) in ready {
            state.stats.released = state.stats.released.saturating_add(1);
            state.free.entry(len).or_default().push(buffer);
        }
        state.refresh_derived_stats();
    }

    fn release_immediate(&self, len: usize, buffer: DeviceBuffer<f32>) {
        let mut state = lock_recover(&self.state, &self.diagnostics);
        state.stats.current_bytes = state.stats.current_bytes.saturating_sub(len * 4);
        state.stats.released = state.stats.released.saturating_add(1);
        state.free.entry(len).or_default().push(buffer);
        state.refresh_derived_stats();
    }

    fn release_deferred(&self, len: usize, buffer: DeviceBuffer<f32>, events: Vec<Arc<CudaEvent>>) {
        if events.is_empty() {
            self.release_immediate(len, buffer);
            return;
        }
        let verdict = evaluate_completion(&events);
        match verdict.decision {
            ReclaimDecision::Reclaim => self.release_immediate(len, buffer),
            ReclaimDecision::Pending => {
                let mut state = lock_recover(&self.state, &self.diagnostics);
                state.stats.current_bytes = state.stats.current_bytes.saturating_sub(len * 4);
                state.pending.push(Pending {
                    len,
                    buffer,
                    events,
                });
                state.refresh_derived_stats();
            }
            ReclaimDecision::Quarantine => {
                self.quarantine(len, buffer, events, verdict.error, true);
            }
        }
    }
}

impl Drop for PoolInner {
    fn drop(&mut self) {
        // Drop must not panic. Taking the state through `get_mut` recovers from
        // a poisoned mutex, and dropping `DeviceBuffer` is safe because
        // cuda-core retains the allocation until its access leases drain.
        let state = match self.state.get_mut() {
            Ok(state) => state,
            Err(poisoned) => {
                self.diagnostics.record_poison_recovery();
                poisoned.into_inner()
            }
        };
        // Quarantined and pending buffers are never returned to the free list;
        // they are released only here, when nothing can reuse them.
        let _ = std::mem::take(&mut state.pending);
        let _ = std::mem::take(&mut state.quarantine);
        let _ = std::mem::take(&mut state.free);
    }
}

#[derive(Clone)]
pub struct CudaBufferPool {
    inner: Arc<PoolInner>,
}

impl CudaBufferPool {
    pub fn new(context: Arc<CudaContext>, stream: Arc<CudaStream>) -> Self {
        let context_id = ContextId::new(Arc::as_ptr(&context) as usize);
        CudaBufferPool {
            inner: Arc::new(PoolInner {
                context,
                context_id,
                stream,
                state: Mutex::new(PoolState::default()),
                diagnostics: Arc::new(EventDiagnostics::new()),
            }),
        }
    }

    pub fn acquire(&self, len: usize) -> Result<PooledBuffer> {
        self.inner.poll();
        let reused = {
            let mut state = lock_recover(&self.inner.state, &self.inner.diagnostics);
            state.free.get_mut(&len).and_then(Vec::pop)
        };
        let buffer = match reused {
            Some(buffer) => {
                let mut state = lock_recover(&self.inner.state, &self.inner.diagnostics);
                state.stats.reuses = state.stats.reuses.saturating_add(1);
                state.stats.current_bytes = state.stats.current_bytes.saturating_add(len * 4);
                state.stats.peak_bytes = state.stats.peak_bytes.max(state.stats.current_bytes);
                buffer
            }
            None => {
                let buffer = DeviceBuffer::<f32>::zeroed(&self.inner.stream, len)?;
                let mut state = lock_recover(&self.inner.state, &self.inner.diagnostics);
                state.stats.allocations = state.stats.allocations.saturating_add(1);
                state.stats.current_bytes = state.stats.current_bytes.saturating_add(len * 4);
                state.stats.peak_bytes = state.stats.peak_bytes.max(state.stats.current_bytes);
                state.stats.reserved_bytes = state.stats.reserved_bytes.saturating_add(len * 4);
                state.stats.peak_reserved_bytes = state
                    .stats
                    .peak_reserved_bytes
                    .max(state.stats.reserved_bytes);
                buffer
            }
        };
        Ok(PooledBuffer {
            len,
            pool: Arc::clone(&self.inner),
            buffer: Some(buffer),
            events: Mutex::new(Vec::new()),
        })
    }

    pub fn poll_pending(&self) {
        self.inner.poll();
    }

    pub fn stats(&self) -> PoolStats {
        self.inner.poll();
        let mut stats = {
            let state = lock_recover(&self.inner.state, &self.inner.diagnostics);
            state.stats.clone()
        };
        stats.lock_poison_recoveries = self.inner.diagnostics.snapshot().poison_recoveries;
        stats
    }

    pub fn quarantine_stats(&self) -> BufferPoolStats {
        let stats = self.stats();
        BufferPoolStats {
            quarantined_count: stats.quarantined_count,
            quarantined_bytes: stats.quarantined_bytes,
            quarantine_total: stats.quarantine_total,
        }
    }

    pub fn quarantine_diagnostics(&self) -> Vec<BufferQuarantineRecord> {
        let state = lock_recover(&self.inner.state, &self.inner.diagnostics);
        state
            .quarantine
            .iter()
            .map(|entry| entry.record.clone())
            .collect()
    }

    pub fn record_plan(&self, slots: usize, reuse: usize, fallback: usize, logical_bytes: usize) {
        let mut state = lock_recover(&self.inner.state, &self.inner.diagnostics);
        state.stats.planned_slot_count = slots;
        state.stats.planned_reuse_count = reuse;
        state.stats.fallback_allocation_count = fallback;
        state.stats.memory_plan_logical_bytes = logical_bytes;
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.inner.context
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.inner.stream
    }

    pub fn context_id(&self) -> ContextId {
        self.inner.context_id
    }
}

pub struct PooledBuffer {
    len: usize,
    pool: Arc<PoolInner>,
    buffer: Option<DeviceBuffer<f32>>,
    events: Mutex<Vec<Arc<CudaEvent>>>,
}

impl PooledBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn buffer(&self) -> Result<&DeviceBuffer<f32>> {
        self.buffer
            .as_ref()
            .ok_or_else(|| CudaError::Internal("pooled buffer was already released".into()))
    }

    pub fn buffer_mut(&mut self) -> Result<&mut DeviceBuffer<f32>> {
        self.buffer
            .as_mut()
            .ok_or_else(|| CudaError::Internal("pooled buffer was already released".into()))
    }

    pub fn pointer(&self) -> Result<u64> {
        Ok(self.buffer()?.cu_deviceptr())
    }

    /// Registers a completion event.
    ///
    /// Rejects events created in another context and events already marked
    /// failed: neither can establish completion for this buffer.
    pub fn mark_used(&self, event: Arc<CudaEvent>) -> Result<()> {
        if event.context_id() != self.pool.context_id {
            return Err(CudaError::Event(format!(
                "cannot attach an event from context {:?} to a buffer from context {:?}",
                event.context_id(),
                self.pool.context_id
            )));
        }
        if event.state() == EventState::Failed {
            return Err(CudaError::Event(
                "cannot attach a failed event to a buffer".into(),
            ));
        }
        let mut events = lock_recover(&self.events, &self.pool.diagnostics);
        events.push(event);
        Ok(())
    }

    pub fn completion_events(&self) -> Vec<Arc<CudaEvent>> {
        lock_recover(&self.events, &self.pool.diagnostics).clone()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        let events = std::mem::take(&mut *lock_recover(&self.events, &self.pool.diagnostics));
        self.pool.release_deferred(self.len, buffer, events);
    }
}

#[derive(Clone)]
pub struct CudaTensor {
    pub shape: Vec<usize>,
    storage: Arc<PooledBuffer>,
}

impl CudaTensor {
    pub fn from_pool(pool: &CudaBufferPool, shape: Vec<usize>) -> Result<Self> {
        let len = shape.iter().product();
        let storage = pool.acquire(len)?;
        Ok(CudaTensor {
            shape,
            storage: Arc::new(storage),
        })
    }

    pub fn from_host(pool: &CudaBufferPool, shape: Vec<usize>, data: &[f32]) -> Result<Self> {
        let len = shape.iter().product::<usize>();
        if data.len() != len {
            return Err(CudaError::Shape(format!(
                "host tensor has {} elements, shape requires {len}",
                data.len()
            )));
        }
        let storage = pool.acquire(len)?;
        let mut tensor = CudaTensor {
            shape,
            storage: Arc::new(storage),
        };
        Arc::get_mut(&mut tensor.storage)
            .ok_or_else(|| CudaError::Internal("newly created pooled tensor is shared".into()))?
            .buffer_mut()?
            .copy_from_host(pool.stream(), data)?;
        Ok(tensor)
    }

    pub fn zeros(pool: &CudaBufferPool, shape: Vec<usize>) -> Result<Self> {
        Self::from_pool(pool, shape)
    }

    pub fn from_storage(shape: Vec<usize>, storage: Arc<PooledBuffer>) -> Result<Self> {
        if shape.iter().product::<usize>() != storage.len() {
            return Err(CudaError::Shape(format!(
                "shape {shape:?} does not match pooled buffer length {}",
                storage.len()
            )));
        }
        Ok(CudaTensor { shape, storage })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn pointer(&self) -> Result<u64> {
        self.storage.pointer()
    }

    pub fn storage(&self) -> &Arc<PooledBuffer> {
        &self.storage
    }

    pub fn buffer(&self) -> Result<&DeviceBuffer<f32>> {
        self.storage.buffer()
    }

    pub fn buffer_mut(&mut self) -> Result<&mut DeviceBuffer<f32>> {
        Arc::get_mut(&mut self.storage)
            .ok_or_else(|| {
                CudaError::Internal(
                    "tensor storage is shared; cannot obtain a unique output buffer".into(),
                )
            })?
            .buffer_mut()
    }

    pub fn to_host(&self, stream: &CudaStream) -> Result<Vec<f32>> {
        Ok(self.storage.buffer()?.to_host_vec(stream)?)
    }

    pub fn mark_used(&self, event: Arc<CudaEvent>) -> Result<()> {
        self.storage.mark_used(event)
    }

    pub fn set_shape(&mut self, shape: Vec<usize>) -> Result<()> {
        if shape.iter().product::<usize>() > self.storage.len() {
            return Err(CudaError::Shape(format!(
                "shape {shape:?} needs more elements than slot capacity {}",
                self.storage.len()
            )));
        }
        self.shape = shape;
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        self.storage.len()
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        if shape.iter().product::<usize>() != self.numel() {
            return Err(CudaError::Shape("reshape element count mismatch".into()));
        }
        Ok(CudaTensor {
            shape,
            storage: self.storage.clone(),
        })
    }

    pub fn copy_from_host(&mut self, stream: &CudaStream, data: &[f32]) -> Result<()> {
        if data.len() != self.numel() {
            return Err(CudaError::Shape("copy_from_host length mismatch".into()));
        }
        Arc::get_mut(&mut self.storage)
            .ok_or_else(|| CudaError::Internal("tensor storage is shared; cannot mutate".into()))?
            .buffer_mut()?
            .copy_from_host(stream, data)?;
        Ok(())
    }
}
