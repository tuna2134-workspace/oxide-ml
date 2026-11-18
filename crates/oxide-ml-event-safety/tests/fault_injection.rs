use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use oxide_ml_event_safety::{
    CompletionEvent, ContextId, EventDiagnostics, EventError, EventFactory, EventOp, EventPool,
    EventState, EventStateMachine, ReclaimDecision, RecordEvent, ReusableEvent,
    evaluate_completion, lock_recover,
};

// ---------------------------------------------------------------------------
// Mock backend with injectable event-operation failures.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct MockBackend {
    ready: AtomicBool,
    create_failures: AtomicUsize,
    record_failures: AtomicUsize,
    query_failures: AtomicUsize,
    synchronize_failures: AtomicUsize,
    destroy_failures: AtomicUsize,
    context_lost: AtomicBool,
    creates: AtomicUsize,
    destroys: AtomicUsize,
}

impl MockBackend {
    fn new(ready: bool) -> Self {
        MockBackend {
            ready: AtomicBool::new(ready),
            ..MockBackend::default()
        }
    }

    fn fail_next(counter: &AtomicUsize) -> bool {
        let mut current = counter.load(Ordering::SeqCst);
        loop {
            if current == 0 {
                return false;
            }
            match counter.compare_exchange(current, current - 1, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    fn destroy_failure(&self) -> bool {
        Self::fail_next(&self.destroy_failures)
    }
}

struct MockEvent {
    backend: Arc<MockBackend>,
    context_id: ContextId,
    machine: Mutex<EventStateMachine>,
    diagnostics: Arc<EventDiagnostics>,
}

impl MockEvent {
    fn error(op: EventOp, message: &str) -> EventError {
        EventError::with_code(op, 999, message)
    }

    fn machine(&self) -> std::sync::MutexGuard<'_, EventStateMachine> {
        lock_recover(&self.machine, &self.diagnostics)
    }
}

impl CompletionEvent for MockEvent {
    fn state(&self) -> EventState {
        self.machine().state()
    }

    fn generation(&self) -> u64 {
        self.machine().generation()
    }

    fn last_error(&self) -> Option<EventError> {
        self.machine().last_error()
    }

    fn context_id(&self) -> ContextId {
        self.context_id
    }

    fn is_complete(&self) -> Result<bool, EventError> {
        let mut machine = self.machine();
        match machine.state() {
            EventState::Unrecorded => Ok(false),
            EventState::Completed => Ok(true),
            EventState::Failed => Err(machine
                .last_error()
                .unwrap_or_else(|| Self::error(EventOp::Query, "event is failed"))),
            EventState::Recorded => {
                if MockBackend::fail_next(&self.backend.query_failures) {
                    let error = Self::error(EventOp::Query, "injected query failure");
                    machine.query_failure(error.clone());
                    Err(error)
                } else {
                    let ready = self.backend.ready.load(Ordering::SeqCst);
                    machine.query_success(ready);
                    Ok(ready)
                }
            }
        }
    }

    fn synchronize(&self) -> Result<(), EventError> {
        let mut machine = self.machine();
        match machine.state() {
            EventState::Unrecorded => Err(Self::error(
                EventOp::Synchronize,
                "cannot synchronize an unrecorded event",
            )),
            EventState::Failed => Err(machine
                .last_error()
                .unwrap_or_else(|| Self::error(EventOp::Synchronize, "event is failed"))),
            EventState::Completed => Ok(()),
            EventState::Recorded => {
                if MockBackend::fail_next(&self.backend.synchronize_failures) {
                    let error = Self::error(EventOp::Synchronize, "injected synchronize failure");
                    machine.synchronize_failure(error.clone());
                    Err(error)
                } else {
                    machine.synchronize_success();
                    Ok(())
                }
            }
        }
    }
}

impl ReusableEvent for MockEvent {
    fn reset_for_reuse(&self) -> Result<(), EventError> {
        self.machine().reset_for_reuse()
    }
}

impl RecordEvent<()> for MockEvent {
    fn record(&self, _stream: &()) -> Result<(), EventError> {
        let mut machine = self.machine();
        machine.begin_record()?;
        if MockBackend::fail_next(&self.backend.record_failures) {
            let error = Self::error(EventOp::Record, "injected record failure");
            machine.record_failure(error.clone());
            Err(error)
        } else {
            machine.record_success();
            Ok(())
        }
    }
}

impl Drop for MockEvent {
    fn drop(&mut self) {
        self.backend.destroys.fetch_add(1, Ordering::SeqCst);
        if self.backend.context_lost.load(Ordering::SeqCst) {
            self.diagnostics.record_destroy_skipped(Some(Self::error(
                EventOp::Destroy,
                "context is no longer valid",
            )));
        } else if self.backend.destroy_failure() {
            self.diagnostics
                .record_destroy_failure(Self::error(EventOp::Destroy, "injected destroy failure"));
        } else {
            self.diagnostics.record_destroy_success();
        }
    }
}

struct MockFactory {
    backend: Arc<MockBackend>,
    context_id: ContextId,
    event_context_id: ContextId,
}

impl EventFactory<MockEvent> for MockFactory {
    fn context_id(&self) -> ContextId {
        self.context_id
    }

    fn create(&self, diagnostics: Arc<EventDiagnostics>) -> Result<MockEvent, EventError> {
        if MockBackend::fail_next(&self.backend.create_failures) {
            return Err(MockEvent::error(EventOp::Create, "injected create failure"));
        }
        self.backend.creates.fetch_add(1, Ordering::SeqCst);
        Ok(MockEvent {
            backend: Arc::clone(&self.backend),
            context_id: self.event_context_id,
            machine: Mutex::new(EventStateMachine::new()),
            diagnostics,
        })
    }
}

struct Harness {
    pool: EventPool<MockFactory, MockEvent>,
    backend: Arc<MockBackend>,
    diagnostics: Arc<EventDiagnostics>,
}

fn harness_with_context(ready: bool, context_id: u64, event_context_id: u64) -> Harness {
    let backend = Arc::new(MockBackend::new(ready));
    let diagnostics = Arc::new(EventDiagnostics::new());
    let factory = MockFactory {
        backend: Arc::clone(&backend),
        context_id: ContextId(context_id),
        event_context_id: ContextId(event_context_id),
    };
    let pool = EventPool::with_diagnostics(factory, Arc::clone(&diagnostics));
    Harness {
        pool,
        backend,
        diagnostics,
    }
}

fn harness(ready: bool) -> Harness {
    harness_with_context(ready, 7, 7)
}

fn acquire_recorded(
    harness: &Harness,
) -> oxide_ml_event_safety::EventLease<MockFactory, MockEvent> {
    let lease = harness.pool.acquire().expect("acquire");
    lease.record(&()).expect("record");
    lease
}

// ---------------------------------------------------------------------------
// 1-5: state machine success paths.
// ---------------------------------------------------------------------------

#[test]
fn unrecorded_event_is_never_complete() {
    let harness = harness(true);
    let lease = harness.pool.acquire().expect("acquire");
    assert_eq!(lease.state(), Some(EventState::Unrecorded));
    assert!(!lease.is_complete().expect("query"));
    assert_eq!(lease.state(), Some(EventState::Unrecorded));

    let event = lease.arc().expect("arc");
    let verdict = evaluate_completion(&[event]);
    assert_eq!(verdict.decision, ReclaimDecision::Pending);
}

#[test]
fn record_success_moves_to_recorded() {
    let harness = harness(true);
    let lease = harness.pool.acquire().expect("acquire");
    lease.record(&()).expect("record");
    assert_eq!(lease.state(), Some(EventState::Recorded));
}

#[test]
fn query_incomplete_keeps_recorded() {
    let harness = harness(false);
    let lease = acquire_recorded(&harness);
    assert!(!lease.is_complete().expect("query"));
    assert_eq!(lease.state(), Some(EventState::Recorded));
}

#[test]
fn query_complete_moves_to_completed() {
    let harness = harness(true);
    let lease = acquire_recorded(&harness);
    assert!(lease.is_complete().expect("query"));
    assert_eq!(lease.state(), Some(EventState::Completed));
}

#[test]
fn synchronize_success_moves_to_completed() {
    let harness = harness(false);
    let lease = acquire_recorded(&harness);
    lease.synchronize().expect("synchronize");
    assert_eq!(lease.state(), Some(EventState::Completed));
}

// ---------------------------------------------------------------------------
// 6-8: operation failures mark Failed.
// ---------------------------------------------------------------------------

#[test]
fn record_failure_marks_failed_and_cannot_rerecord() {
    let harness = harness(true);
    harness.backend.record_failures.store(1, Ordering::SeqCst);
    let lease = harness.pool.acquire().expect("acquire");
    let error = lease.record(&()).expect_err("record must fail");
    assert_eq!(error.op, EventOp::Record);
    assert_eq!(lease.state(), Some(EventState::Failed));
    assert!(lease.last_error().is_some());

    let second = lease
        .record(&())
        .expect_err("failed event must not re-record");
    assert_eq!(second.op, EventOp::Record);
    assert_eq!(lease.state(), Some(EventState::Failed));
}

#[test]
fn query_failure_marks_failed() {
    let harness = harness(true);
    let lease = acquire_recorded(&harness);
    harness.backend.query_failures.store(1, Ordering::SeqCst);
    let error = lease.is_complete().expect_err("query must fail");
    assert_eq!(error.op, EventOp::Query);
    assert_eq!(lease.state(), Some(EventState::Failed));
}

#[test]
fn synchronize_failure_marks_failed() {
    let harness = harness(false);
    let lease = acquire_recorded(&harness);
    harness
        .backend
        .synchronize_failures
        .store(1, Ordering::SeqCst);
    let error = lease.synchronize().expect_err("synchronize must fail");
    assert_eq!(error.op, EventOp::Synchronize);
    assert_eq!(lease.state(), Some(EventState::Failed));
}

// ---------------------------------------------------------------------------
// 9-11: quarantine.
// ---------------------------------------------------------------------------

#[test]
fn failed_event_is_quarantined_and_not_reused() {
    let harness = harness(true);
    harness.backend.record_failures.store(1, Ordering::SeqCst);
    let failed = harness.pool.acquire().expect("acquire");
    let failed_id = failed.id().expect("id");
    failed.record(&()).expect_err("record must fail");
    drop(failed);

    let stats = harness.pool.stats();
    assert_eq!(stats.quarantine_count, 1);
    assert_eq!(stats.quarantine_total, 1);
    assert_eq!(stats.pending_count, 0);
    assert_eq!(stats.allocation_count, 1);
    assert_eq!(stats.reuse_count, 0);

    let records = harness.pool.quarantine_diagnostics();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].event_id, failed_id);
    assert_eq!(records[0].state, EventState::Failed);
    assert_eq!(
        records[0].error.as_ref().map(|e| e.op),
        Some(EventOp::Record)
    );

    // The next acquisition must allocate a fresh event, not reuse the
    // quarantined one.
    let next = harness.pool.acquire().expect("acquire after quarantine");
    assert_ne!(next.id(), Some(failed_id));
    let stats = harness.pool.stats();
    assert_eq!(stats.allocation_count, 2);
    assert_eq!(stats.reuse_count, 0);
    assert_eq!(stats.quarantine_count, 1);
}

#[test]
fn quarantine_event_never_enters_free_list() {
    let harness = harness(true);
    harness.backend.record_failures.store(1, Ordering::SeqCst);
    let failed = harness.pool.acquire().expect("acquire");
    let failed_id = failed.id().expect("id");
    failed.record(&()).expect_err("record must fail");
    drop(failed);

    for _ in 0..8 {
        let lease = harness.pool.acquire().expect("acquire");
        assert_ne!(lease.id(), Some(failed_id));
        // Unrecorded leases are safe to return to the free list.
        drop(lease);
    }
    let stats = harness.pool.stats();
    assert_eq!(stats.quarantine_count, 1);
    assert_eq!(stats.quarantine_total, 1);
}

#[test]
fn query_failure_on_release_moves_pending_to_quarantine() {
    let harness = harness(false);
    let lease = acquire_recorded(&harness);
    drop(lease);

    let stats = harness.pool.stats();
    assert_eq!(stats.pending_count, 1);
    harness.backend.query_failures.store(1, Ordering::SeqCst);
    let report = harness.pool.poll();
    assert_eq!(report.quarantined, 1);
    let stats = harness.pool.stats();
    assert_eq!(stats.pending_count, 0);
    assert_eq!(stats.quarantine_count, 1);
    assert_eq!(stats.quarantine_total, 1);
}

// ---------------------------------------------------------------------------
// 12-13: destroy / create statistics.
// ---------------------------------------------------------------------------

#[test]
fn destroy_failure_is_reflected_in_statistics() {
    let harness = harness(true);
    harness.backend.destroy_failures.store(1, Ordering::SeqCst);
    let lease = harness.pool.acquire().expect("acquire");
    drop(lease);

    let report = harness.pool.shutdown();
    assert_eq!(report.free_events, 1);
    assert_eq!(report.destroy_attempts, 1);
    assert_eq!(report.destroy_failures, 1);

    let stats = harness.pool.stats();
    assert_eq!(stats.destroy_failure_count, 1);
    assert_eq!(stats.destroy_attempt_count, 1);
    assert_eq!(stats.destroy_skipped_count, 0);
}

#[test]
fn destroy_is_skipped_when_context_is_lost() {
    let harness = harness(true);
    harness.backend.context_lost.store(true, Ordering::SeqCst);
    let lease = harness.pool.acquire().expect("acquire");
    drop(lease);

    let report = harness.pool.shutdown();
    assert_eq!(report.destroy_attempts, 0);
    assert_eq!(report.destroy_skipped, 1);

    let stats = harness.pool.stats();
    assert_eq!(stats.destroy_failure_count, 0);
    assert_eq!(stats.destroy_skipped_count, 1);
}

#[test]
fn create_failure_does_not_corrupt_statistics() {
    let harness = harness(true);
    harness.backend.create_failures.store(1, Ordering::SeqCst);
    let error = harness.pool.acquire().expect_err("create must fail");
    assert_eq!(error.op, EventOp::Create);

    let stats = harness.pool.stats();
    assert_eq!(stats.allocation_count, 0);
    assert_eq!(stats.reuse_count, 0);
    assert_eq!(stats.pending_count, 0);
    assert_eq!(stats.quarantine_count, 0);
    assert_eq!(stats.destroy_attempt_count, 0);
    assert_eq!(harness.backend.creates.load(Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------------------
// 14: pending events stay out of the free list.
// ---------------------------------------------------------------------------

#[test]
fn incomplete_pending_event_is_not_reused() {
    let harness = harness(false);
    let first = acquire_recorded(&harness);
    let first_id = first.id().expect("id");
    drop(first);

    let stats = harness.pool.stats();
    assert_eq!(stats.pending_count, 1);

    let second = harness.pool.acquire().expect("acquire");
    assert_ne!(second.id(), Some(first_id));
    let stats = harness.pool.stats();
    assert_eq!(stats.allocation_count, 2);
    assert_eq!(stats.reuse_count, 0);
    assert_eq!(stats.pending_count, 1);

    // Once completion is confirmed the pending event becomes reusable.
    harness.backend.ready.store(true, Ordering::SeqCst);
    let stats = harness.pool.stats();
    assert_eq!(stats.pending_count, 0);
    let third = harness.pool.acquire().expect("acquire");
    assert_eq!(third.id(), Some(first_id));
    assert_eq!(harness.pool.stats().reuse_count, 1);
}

// ---------------------------------------------------------------------------
// 15-16: buffer reclamation policy.
// ---------------------------------------------------------------------------

#[test]
fn buffer_is_not_reclaimed_before_all_events_complete() {
    let harness = harness(true);
    let completed = acquire_recorded(&harness);
    completed.synchronize().expect("synchronize");
    let completed_event = completed.arc().expect("arc");

    harness.backend.ready.store(false, Ordering::SeqCst);
    let in_flight = acquire_recorded(&harness);
    let in_flight_event = in_flight.arc().expect("arc");

    let verdict = evaluate_completion(&[completed_event.clone(), in_flight_event.clone()]);
    assert_eq!(verdict.decision, ReclaimDecision::Pending);

    // Unrecorded events also hold the buffer.
    let fresh = harness.pool.acquire().expect("acquire");
    let fresh_event = fresh.arc().expect("arc");
    let verdict = evaluate_completion(&[completed_event, fresh_event]);
    assert_eq!(verdict.decision, ReclaimDecision::Pending);
}

#[test]
fn buffer_with_failed_event_is_quarantined() {
    let harness = harness(true);
    let completed = acquire_recorded(&harness);
    completed.synchronize().expect("synchronize");
    let completed_event = completed.arc().expect("arc");

    harness.backend.record_failures.store(1, Ordering::SeqCst);
    let failed = harness.pool.acquire().expect("acquire");
    failed.record(&()).expect_err("record must fail");
    let failed_event = failed.arc().expect("arc");

    let verdict = evaluate_completion(&[completed_event, failed_event]);
    assert_eq!(verdict.decision, ReclaimDecision::Quarantine);
    assert_eq!(verdict.error.map(|error| error.op), Some(EventOp::Record));
}

#[test]
fn buffer_with_query_error_is_quarantined() {
    let harness = harness(true);
    let in_flight = acquire_recorded(&harness);
    let event = in_flight.arc().expect("arc");

    harness.backend.query_failures.store(1, Ordering::SeqCst);
    let verdict = evaluate_completion(&[event]);
    assert_eq!(verdict.decision, ReclaimDecision::Quarantine);
    assert_eq!(verdict.error.map(|error| error.op), Some(EventOp::Query));
}

// ---------------------------------------------------------------------------
// 17: context mismatch.
// ---------------------------------------------------------------------------

#[test]
fn event_from_different_context_is_rejected() {
    let harness = harness_with_context(true, 1, 2);
    let error = harness
        .pool
        .acquire()
        .expect_err("context mismatch must fail");
    assert_eq!(error.op, EventOp::Context);
    let stats = harness.pool.stats();
    assert_eq!(stats.allocation_count, 0);
    assert_eq!(stats.reuse_count, 0);
    assert_eq!(stats.quarantine_count, 0);
}

// ---------------------------------------------------------------------------
// 18: Drop safety.
// ---------------------------------------------------------------------------

#[test]
fn event_pool_drop_does_not_panic_with_failed_pending_and_quarantined_events() {
    let harness = harness(false);
    harness.backend.destroy_failures.store(32, Ordering::SeqCst);

    // Free event.
    let free = harness.pool.acquire().expect("acquire");
    drop(free);

    // Pending event.
    let pending = acquire_recorded(&harness);
    drop(pending);

    // Quarantined event.
    harness.backend.record_failures.store(1, Ordering::SeqCst);
    let failed = harness.pool.acquire().expect("acquire");
    failed.record(&()).expect_err("record must fail");
    drop(failed);

    let stats = harness.pool.stats();
    assert_eq!(stats.quarantine_count, 1);
    assert_eq!(stats.pending_count, 1);

    let diagnostics = Arc::clone(&harness.diagnostics);
    let result = catch_unwind(AssertUnwindSafe(move || drop(harness)));
    assert!(result.is_ok(), "EventPool Drop panicked");
    // The free event is reused for the failed acquisition, so the pool owns
    // two distinct events: one pending and one quarantined.
    assert_eq!(diagnostics.snapshot().destroy_failures, 2);
}

#[test]
fn state_machine_never_promotes_failed_or_unrecorded() {
    let mut machine = EventStateMachine::new();
    machine.query_success(true);
    assert_eq!(machine.state(), EventState::Unrecorded);
    assert!(machine.reset_for_reuse().is_ok());

    machine.record_success();
    assert_eq!(machine.state(), EventState::Recorded);
    machine.record_failure(EventError::new(EventOp::Record, "boom"));
    assert_eq!(machine.state(), EventState::Failed);
    assert!(machine.reset_for_reuse().is_err());
    assert!(machine.begin_record().is_err());
    machine.synchronize_success();
    assert_eq!(machine.state(), EventState::Failed);
}

#[test]
fn poisoned_event_state_mutex_is_recovered() {
    let harness = harness(true);
    let lease = harness.pool.acquire().expect("acquire");
    let event = lease.arc().expect("arc");
    let poisoned = catch_unwind(AssertUnwindSafe(|| {
        let _guard = event.machine.lock().expect("test lock");
        panic!("poison the event state mutex");
    }));
    assert!(poisoned.is_err());

    lease.record(&()).expect("record after poison recovery");
    assert_eq!(lease.state(), Some(EventState::Recorded));
    assert!(harness.diagnostics.snapshot().poison_recoveries >= 1);
}
