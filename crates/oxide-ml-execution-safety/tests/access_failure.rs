use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use oxide_ml_event_safety::{EventDiagnostics, lock_recover};
use oxide_ml_execution_safety::{
    AccessMode, BufferAccessState, BufferContextId, BufferId, BufferIdentity,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct MockStream(u64);

#[derive(Debug)]
struct MockEvent {
    complete: AtomicBool,
    failed: AtomicBool,
}

impl MockEvent {
    fn new() -> Arc<Self> {
        Arc::new(MockEvent {
            complete: AtomicBool::new(false),
            failed: AtomicBool::new(false),
        })
    }
}

struct Harness {
    state: BufferAccessState<Arc<MockEvent>, MockStream>,
    fail_wait: bool,
    fail_acquire: bool,
    fail_launch: bool,
    fail_record: bool,
}

impl Harness {
    fn new() -> Self {
        Harness {
            state: BufferAccessState::new(BufferIdentity::new(BufferId(1), 0, BufferContextId(1))),
            fail_wait: false,
            fail_acquire: false,
            fail_launch: false,
            fail_record: false,
        }
    }

    fn launch(
        &mut self,
        stream: MockStream,
        mode: AccessMode,
    ) -> Result<Arc<MockEvent>, &'static str> {
        let plan = self.state.plan_access(&stream, mode, |_| false);
        if self.fail_wait && !plan.waits.is_empty() {
            return Err("stream wait");
        }
        if self.fail_acquire {
            return Err("event acquire");
        }
        if self.fail_launch {
            return Err("kernel launch");
        }
        let event = MockEvent::new();
        if self.fail_record {
            event.failed.store(true, Ordering::SeqCst);
            return Err("event record");
        }
        self.state.record(stream, mode, Arc::clone(&event));
        Ok(event)
    }
}

#[test]
fn stream_wait_failure_does_not_record_access() {
    let mut harness = Harness::new();
    harness.launch(MockStream(1), AccessMode::Write).unwrap();
    harness.fail_wait = true;
    assert_eq!(
        harness.launch(MockStream(2), AccessMode::Read).unwrap_err(),
        "stream wait"
    );
    assert_eq!(harness.state.pending_readers(), 0);
    assert!(harness.state.has_writer());
}

#[test]
fn kernel_launch_failure_does_not_record_access() {
    let mut harness = Harness::new();
    harness.fail_launch = true;
    assert_eq!(
        harness
            .launch(MockStream(1), AccessMode::Write)
            .unwrap_err(),
        "kernel launch"
    );
    assert_eq!(harness.state.pending_readers(), 0);
    assert!(!harness.state.has_writer());
}

#[test]
fn event_record_failure_is_not_installed() {
    let mut harness = Harness::new();
    harness.fail_record = true;
    let failed = harness.launch(MockStream(1), AccessMode::Read).unwrap_err();
    assert_eq!(failed, "event record");
    assert_eq!(harness.state.pending_readers(), 0);
    // A later access must not see a phantom completion token.
    harness.fail_record = false;
    harness.launch(MockStream(2), AccessMode::Read).unwrap();
    assert_eq!(harness.state.pending_readers(), 1);
}

#[test]
fn guard_token_survives_host_drop() {
    let mut harness = Harness::new();
    let token = harness.launch(MockStream(1), AccessMode::Read).unwrap();
    drop(token);
    let plan = harness
        .state
        .plan_access(&MockStream(2), AccessMode::Write, |_| false);
    assert_eq!(plan.waits.len(), 1);
    assert_eq!(harness.state.pending_readers(), 1);
}

#[test]
fn generation_reset_rejects_old_identity() {
    let mut harness = Harness::new();
    let old = harness.state.identity();
    harness
        .state
        .reset(BufferIdentity::new(BufferId(1), 1, BufferContextId(1)));
    assert_ne!(harness.state.identity(), old);
    assert_eq!(harness.state.identity().generation, 1);
}

#[test]
fn context_mismatch_is_detectable() {
    let a = BufferIdentity::new(BufferId(1), 0, BufferContextId(1));
    let b = BufferIdentity::new(BufferId(1), 0, BufferContextId(2));
    assert_ne!(a.context_id, b.context_id);
}

#[test]
fn poisoned_access_lock_recovers() {
    let state = Mutex::new(Harness::new());
    let diagnostics = Arc::new(EventDiagnostics::new());
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = state.lock().unwrap();
        panic!("poison");
    }));
    let guard = lock_recover(&state, &diagnostics);
    assert_eq!(guard.state.identity().generation, 0);
}

#[test]
fn completed_pending_access_is_pruned_safely() {
    let mut harness = Harness::new();
    let token = harness.launch(MockStream(1), AccessMode::Read).unwrap();
    token.complete.store(true, Ordering::SeqCst);
    let plan = harness
        .state
        .plan_access(&MockStream(2), AccessMode::Write, |event| {
            event.complete.load(Ordering::SeqCst)
        });
    assert!(plan.waits.is_empty());
    assert_eq!(harness.state.pending_readers(), 0);
}
