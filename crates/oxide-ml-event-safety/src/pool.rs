use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::diagnostics::{EventDiagnostics, EventDiagnosticsSnapshot, lock_recover};
use crate::error::{EventError, EventOp};
use crate::state::{ContextId, EventState};

/// Why a completion event is not reclaimable.
pub trait CompletionEvent: Send + Sync + 'static {
    fn state(&self) -> EventState;
    fn generation(&self) -> u64;
    fn last_error(&self) -> Option<EventError>;
    fn context_id(&self) -> ContextId;
    /// Queries completion. Only `Ok(true)` is a confirmed completion.
    fn is_complete(&self) -> Result<bool, EventError>;
    fn synchronize(&self) -> Result<(), EventError>;
}

/// An event that may be returned to a pool free list after its state was
/// reset. Implementations must never reset a `Failed` event.
pub trait ReusableEvent: CompletionEvent {
    fn reset_for_reuse(&self) -> Result<(), EventError>;
}

pub trait RecordEvent<Stream: ?Sized>: CompletionEvent {
    fn record(&self, stream: &Stream) -> Result<(), EventError>;
}

/// Creates events bound to one context.
///
/// The pool validates the context of every created event; a factory that
/// returns an event from another context is rejected instead of being
/// registered.
pub trait EventFactory<E>: Send + Sync + 'static {
    fn context_id(&self) -> ContextId;
    fn create(&self, diagnostics: Arc<EventDiagnostics>) -> Result<E, EventError>;
}

/// Event pool statistics.
///
/// `quarantine_count` is the current quarantine size; `quarantine_total` is
/// cumulative. `destroy_failure_count` counts event destruction failures that
/// were observed by the shared diagnostics sink (including the final `Drop`
/// of events that outlived the pool).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EventPoolStats {
    pub allocation_count: usize,
    pub reuse_count: usize,
    pub pending_count: usize,
    pub quarantine_count: usize,
    pub quarantine_total: usize,
    pub destroy_failure_count: usize,
    pub destroy_attempt_count: usize,
    pub destroy_skipped_count: usize,
    pub poison_recovery_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuarantineRecord {
    pub event_id: u64,
    pub generation: u64,
    pub state: EventState,
    pub error: Option<EventError>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PollReport {
    pub completed: usize,
    pub quarantined: usize,
    pub still_pending: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CleanupReport {
    pub free_events: usize,
    pub pending_events: usize,
    pub quarantined_events: usize,
    pub destroy_attempts: usize,
    pub destroy_failures: usize,
    pub destroy_skipped: usize,
    pub poison_recoveries: usize,
}

struct PooledEvent<E> {
    id: u64,
    event: Arc<E>,
}

struct PoolLists<E> {
    free: Vec<Arc<PooledEvent<E>>>,
    pending: Vec<Arc<PooledEvent<E>>>,
    quarantine: Vec<Arc<PooledEvent<E>>>,
    next_id: u64,
    stats: EventPoolStats,
}

impl<E> Default for PoolLists<E> {
    fn default() -> Self {
        PoolLists {
            free: Vec::new(),
            pending: Vec::new(),
            quarantine: Vec::new(),
            next_id: 1,
            stats: EventPoolStats::default(),
        }
    }
}

impl<E> PoolLists<E> {
    fn sync_counts(&mut self) {
        self.stats.pending_count = self.pending.len();
        self.stats.quarantine_count = self.quarantine.len();
    }

    fn quarantine(&mut self, event: Arc<PooledEvent<E>>) {
        self.pending
            .retain(|candidate| !Arc::ptr_eq(candidate, &event));
        self.free
            .retain(|candidate| !Arc::ptr_eq(candidate, &event));
        self.quarantine.push(event);
        self.stats.quarantine_total = self.stats.quarantine_total.saturating_add(1);
        self.sync_counts();
    }
}

struct EventPoolInner<F, E> {
    factory: F,
    context_id: ContextId,
    diagnostics: Arc<EventDiagnostics>,
    lists: Mutex<PoolLists<E>>,
}

impl<F, E> Drop for EventPoolInner<F, E> {
    fn drop(&mut self) {
        // `Drop` must not panic. `get_mut` is available because the pool inner
        // is uniquely owned at this point; recover from poisoning by taking
        // the inner value.
        let lists = match self.lists.get_mut() {
            Ok(lists) => lists,
            Err(poisoned) => {
                self.diagnostics.record_poison_recovery();
                poisoned.into_inner()
            }
        };
        let free = std::mem::take(&mut lists.free);
        let pending = std::mem::take(&mut lists.pending);
        let quarantine = std::mem::take(&mut lists.quarantine);
        // Events still referenced by buffers or user code stay alive; they
        // destroy their driver handle when the last `Arc` is dropped. The
        // driver defers destruction of an event captured by unfinished work,
        // so this is safe even for pending/quarantined events.
        drop(free);
        drop(pending);
        drop(quarantine);
    }
}

/// Context-bound event pool with explicit quarantine.
///
/// `S` is the backend stream type used by [`EventLease::record`]. It is part
/// of the type so the lease can stay free of dynamic dispatch.
pub struct EventPool<F, E, S: ?Sized = ()> {
    inner: Arc<EventPoolInner<F, E>>,
    _stream: PhantomData<fn() -> S>,
}

impl<F, E, S: ?Sized> Clone for EventPool<F, E, S> {
    fn clone(&self) -> Self {
        EventPool {
            inner: Arc::clone(&self.inner),
            _stream: PhantomData,
        }
    }
}

impl<F, E, S: ?Sized> fmt::Debug for EventPool<F, E, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventPool")
            .field("context_id", &self.inner.context_id)
            .finish_non_exhaustive()
    }
}

impl<F, E, S: ?Sized> EventPool<F, E, S>
where
    F: EventFactory<E>,
    E: ReusableEvent,
{
    pub fn new(factory: F) -> Self {
        Self::with_diagnostics(factory, Arc::new(EventDiagnostics::new()))
    }

    pub fn with_diagnostics(factory: F, diagnostics: Arc<EventDiagnostics>) -> Self {
        let context_id = factory.context_id();
        EventPool {
            inner: Arc::new(EventPoolInner {
                factory,
                context_id,
                diagnostics,
                lists: Mutex::new(PoolLists::default()),
            }),
            _stream: PhantomData,
        }
    }

    pub fn context_id(&self) -> ContextId {
        self.inner.context_id
    }

    pub fn diagnostics(&self) -> Arc<EventDiagnostics> {
        Arc::clone(&self.inner.diagnostics)
    }

    /// Acquires a lease, reusing a free event when possible.
    ///
    /// A failed creation increments no allocation statistic and never
    /// registers a partially initialized event.
    pub fn acquire(&self) -> Result<EventLease<F, E, S>, EventError> {
        let _ = self.poll();

        let reused = {
            let mut lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
            lists.free.pop()
        };

        let pooled = match reused {
            Some(event) => {
                if let Err(error) = event.event.reset_for_reuse() {
                    // A failed/recorded event must never count as a reuse.
                    let mut lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
                    lists.quarantine(event);
                    return Err(error);
                }
                let mut lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
                lists.stats.reuse_count = lists.stats.reuse_count.saturating_add(1);
                event
            }
            None => {
                let event = self
                    .inner
                    .factory
                    .create(Arc::clone(&self.inner.diagnostics))?;
                if event.context_id() != self.inner.context_id {
                    // Do not register an event from another context. Dropping
                    // it here destroys only this rejected handle.
                    drop(event);
                    return Err(EventError::new(
                        EventOp::Context,
                        "event factory returned an event bound to a different context",
                    ));
                }
                let mut lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
                lists.stats.allocation_count = lists.stats.allocation_count.saturating_add(1);
                let id = lists.next_id;
                lists.next_id = lists.next_id.saturating_add(1);
                Arc::new(PooledEvent {
                    id,
                    event: Arc::new(event),
                })
            }
        };

        Ok(EventLease {
            inner: Arc::clone(&self.inner),
            event: Some(pooled),
            _stream: PhantomData,
        })
    }

    /// Queries every pending event.
    ///
    /// Pending events are removed under the pool lock, queried without
    /// holding it, then committed. Concurrent releases simply append to the
    /// (possibly freshly emptied) pending list.
    pub fn poll(&self) -> PollReport {
        let pending = {
            let mut lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
            std::mem::take(&mut lists.pending)
        };
        let mut completed = Vec::new();
        let mut failed = Vec::new();
        let mut still_pending = Vec::new();
        for event in pending {
            match event.event.state() {
                EventState::Failed => failed.push(event),
                EventState::Completed => completed.push(event),
                EventState::Unrecorded => still_pending.push(event),
                EventState::Recorded => match event.event.is_complete() {
                    Ok(true) => completed.push(event),
                    Ok(false) => still_pending.push(event),
                    Err(_) => failed.push(event),
                },
            }
        }

        let completed_count = completed.len();
        let failed_count = failed.len();
        let pending_count = still_pending.len();
        {
            let mut lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
            lists.free.extend(completed);
            for event in failed {
                lists.quarantine(event);
            }
            lists.pending.extend(still_pending);
            lists.sync_counts();
        }
        PollReport {
            completed: completed_count,
            quarantined: failed_count,
            still_pending: pending_count,
        }
    }

    pub fn stats(&self) -> EventPoolStats {
        self.poll();
        let mut stats = {
            let lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
            lists.stats
        };
        let diagnostics = self.inner.diagnostics.snapshot();
        stats.destroy_failure_count = diagnostics.destroy_failures;
        stats.destroy_attempt_count = diagnostics.destroy_attempts;
        stats.destroy_skipped_count = diagnostics.destroy_skipped;
        stats.poison_recovery_count = diagnostics.poison_recoveries;
        stats
    }

    pub fn diagnostics_snapshot(&self) -> EventDiagnosticsSnapshot {
        self.inner.diagnostics.snapshot()
    }

    /// Returns diagnostics for every currently quarantined event.
    pub fn quarantine_diagnostics(&self) -> Vec<QuarantineRecord> {
        let events: Vec<Arc<PooledEvent<E>>> = {
            let lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
            lists.quarantine.clone()
        };
        events
            .iter()
            .map(|event| QuarantineRecord {
                event_id: event.id,
                generation: event.event.generation(),
                state: event.event.state(),
                error: event.event.last_error(),
            })
            .collect()
    }

    /// Drains all pool lists and destroys the handles whose last reference is
    /// owned by the pool.
    ///
    /// Events still referenced by buffers or user code are not destroyed yet;
    /// their `Drop` records the destruction result in the shared diagnostics.
    /// Repeating `shutdown` is harmless.
    pub fn shutdown(&self) -> CleanupReport {
        let before = self.inner.diagnostics.snapshot();
        let (free, pending, quarantine) = {
            let mut lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
            (
                std::mem::take(&mut lists.free),
                std::mem::take(&mut lists.pending),
                std::mem::take(&mut lists.quarantine),
            )
        };
        let report = CleanupReport {
            free_events: free.len(),
            pending_events: pending.len(),
            quarantined_events: quarantine.len(),
            ..CleanupReport::default()
        };
        drop(free);
        drop(pending);
        drop(quarantine);
        {
            let mut lists = lock_recover(&self.inner.lists, &self.inner.diagnostics);
            lists.sync_counts();
        }
        let after = self.inner.diagnostics.snapshot();
        CleanupReport {
            destroy_attempts: after
                .destroy_attempts
                .saturating_sub(before.destroy_attempts),
            destroy_failures: after
                .destroy_failures
                .saturating_sub(before.destroy_failures),
            destroy_skipped: after.destroy_skipped.saturating_sub(before.destroy_skipped),
            poison_recoveries: after
                .poison_recoveries
                .saturating_sub(before.poison_recoveries),
            ..report
        }
    }
}

impl<F, E> EventPoolInner<F, E>
where
    F: EventFactory<E>,
    E: ReusableEvent,
{
    fn release(&self, event: Arc<PooledEvent<E>>) {
        let destination = match event.event.state() {
            EventState::Failed => Destination::Quarantine,
            EventState::Unrecorded | EventState::Completed => Destination::Free,
            EventState::Recorded => match event.event.is_complete() {
                Ok(true) => Destination::Free,
                Ok(false) => Destination::Pending,
                // `is_complete` marks the event Failed before returning an
                // error, so a failed query can never be treated as complete.
                Err(_) => Destination::Quarantine,
            },
        };
        let mut lists = lock_recover(&self.lists, &self.diagnostics);
        match destination {
            Destination::Free => lists.free.push(event),
            Destination::Pending => lists.pending.push(event),
            Destination::Quarantine => lists.quarantine(event),
        }
        lists.sync_counts();
    }
}

#[derive(Clone, Copy)]
enum Destination {
    Free,
    Pending,
    Quarantine,
}

/// RAII lease returned by [`EventPool::acquire`].
pub struct EventLease<F, E, S: ?Sized = ()>
where
    F: EventFactory<E>,
    E: ReusableEvent,
{
    inner: Arc<EventPoolInner<F, E>>,
    event: Option<Arc<PooledEvent<E>>>,
    _stream: PhantomData<fn() -> S>,
}

impl<F, E, S: ?Sized> fmt::Debug for EventLease<F, E, S>
where
    F: EventFactory<E>,
    E: ReusableEvent,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventLease")
            .field("id", &self.id())
            .field("state", &self.state())
            .finish()
    }
}

impl<F, E, S: ?Sized> EventLease<F, E, S>
where
    F: EventFactory<E>,
    E: ReusableEvent,
{
    fn event(&self) -> Result<&Arc<PooledEvent<E>>, EventError> {
        self.event
            .as_ref()
            .ok_or_else(|| EventError::new(EventOp::Context, "event lease was already released"))
    }

    pub fn id(&self) -> Option<u64> {
        self.event.as_ref().map(|event| event.id)
    }

    pub fn state(&self) -> Option<EventState> {
        self.event.as_ref().map(|event| event.event.state())
    }

    pub fn last_error(&self) -> Option<EventError> {
        self.event
            .as_ref()
            .and_then(|event| event.event.last_error())
    }

    pub fn arc(&self) -> Result<Arc<E>, EventError> {
        Ok(Arc::clone(&self.event()?.event))
    }

    pub fn is_complete(&self) -> Result<bool, EventError> {
        self.event()?.event.is_complete()
    }

    pub fn synchronize(&self) -> Result<(), EventError> {
        self.event()?.event.synchronize()
    }
}

impl<F, E, S: ?Sized> EventLease<F, E, S>
where
    F: EventFactory<E>,
    E: ReusableEvent + RecordEvent<S>,
{
    pub fn record(&self, stream: &S) -> Result<(), EventError> {
        self.event()?.event.record(stream)
    }
}

impl<F, E, S: ?Sized> Drop for EventLease<F, E, S>
where
    F: EventFactory<E>,
    E: ReusableEvent,
{
    fn drop(&mut self) {
        if let Some(event) = self.event.take() {
            self.inner.release(event);
        }
    }
}
