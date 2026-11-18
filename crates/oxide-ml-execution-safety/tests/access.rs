use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use oxide_ml_execution_safety::{
    AccessMode, BufferAccessState, BufferContextId, BufferId, BufferIdentity,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct MockStream(u64);

#[derive(Debug)]
struct MockEvent {
    complete: AtomicBool,
}

impl MockEvent {
    fn new() -> Arc<Self> {
        Arc::new(MockEvent {
            complete: AtomicBool::new(false),
        })
    }

    fn complete(&self) {
        self.complete.store(true, Ordering::SeqCst);
    }
}

type State = BufferAccessState<Arc<MockEvent>, MockStream>;

fn identity(generation: u64) -> BufferIdentity {
    BufferIdentity::new(BufferId(1), generation, BufferContextId(1))
}

fn state() -> State {
    BufferAccessState::new(identity(0))
}

fn never_complete(_: &Arc<MockEvent>) -> bool {
    false
}

#[test]
fn read_read_does_not_wait() {
    let mut state = state();
    let first = MockEvent::new();
    let plan = state.plan_access(&MockStream(1), AccessMode::Read, never_complete);
    assert!(plan.waits.is_empty());
    state.record(MockStream(1), AccessMode::Read, first);

    let plan = state.plan_access(&MockStream(2), AccessMode::Read, never_complete);
    assert!(plan.waits.is_empty(), "read/read must not wait");
    let second = MockEvent::new();
    state.record(MockStream(2), AccessMode::Read, second);
    assert_eq!(state.stats().read_read_concurrent, 1);
}

#[test]
fn read_then_write_waits_reader() {
    let mut state = state();
    let read = MockEvent::new();
    state.record(MockStream(1), AccessMode::Read, Arc::clone(&read));
    let plan = state.plan_access(&MockStream(2), AccessMode::Write, never_complete);
    assert_eq!(plan.waits.len(), 1);
    assert_eq!(plan.cross_stream_waits, 1);
    assert!(Arc::ptr_eq(&plan.waits[0], &read));
}

#[test]
fn write_then_read_waits_writer() {
    let mut state = state();
    let write = MockEvent::new();
    state.record(MockStream(1), AccessMode::Write, Arc::clone(&write));
    let plan = state.plan_access(&MockStream(2), AccessMode::Read, never_complete);
    assert_eq!(plan.waits.len(), 1);
    assert!(Arc::ptr_eq(&plan.waits[0], &write));
}

#[test]
fn write_then_write_waits_writer() {
    let mut state = state();
    let write = MockEvent::new();
    state.record(MockStream(1), AccessMode::Write, Arc::clone(&write));
    let plan = state.plan_access(&MockStream(2), AccessMode::Write, never_complete);
    assert_eq!(plan.waits.len(), 1);
    assert!(Arc::ptr_eq(&plan.waits[0], &write));
}

#[test]
fn read_write_waits_like_write() {
    let mut state = state();
    let read = MockEvent::new();
    let write = MockEvent::new();
    state.record(MockStream(1), AccessMode::Read, Arc::clone(&read));
    state.record(MockStream(2), AccessMode::Write, Arc::clone(&write));

    let plan = state.plan_access(&MockStream(3), AccessMode::Read, never_complete);
    assert_eq!(plan.waits.len(), 1, "only the writer remains after a write");
    assert!(Arc::ptr_eq(&plan.waits[0], &write));

    let plan = state.plan_access(&MockStream(3), AccessMode::ReadWrite, never_complete);
    assert_eq!(plan.waits.len(), 1);
}

#[test]
fn same_stream_hazards_are_elided() {
    let mut state = state();
    let write = MockEvent::new();
    state.record(MockStream(1), AccessMode::Write, write);
    let plan = state.plan_access(&MockStream(1), AccessMode::Read, never_complete);
    assert!(plan.waits.is_empty());
    assert_eq!(plan.same_stream_elided, 1);

    let read = MockEvent::new();
    state.record(MockStream(1), AccessMode::Read, read);
    let plan = state.plan_access(&MockStream(1), AccessMode::Write, never_complete);
    assert!(plan.waits.is_empty());
    assert_eq!(plan.same_stream_elided, 1);
}

#[test]
fn multiple_readers_are_all_waited_before_write() {
    let mut state = state();
    let r1 = MockEvent::new();
    let r2 = MockEvent::new();
    let r3 = MockEvent::new();
    state.record(MockStream(1), AccessMode::Read, Arc::clone(&r1));
    state.record(MockStream(2), AccessMode::Read, Arc::clone(&r2));
    state.record(MockStream(1), AccessMode::Read, Arc::clone(&r3));

    let plan = state.plan_access(&MockStream(3), AccessMode::Write, never_complete);
    assert_eq!(plan.waits.len(), 3);
    assert_eq!(plan.cross_stream_waits, 3);
    assert_eq!(state.pending_readers(), 3);
}

#[test]
fn three_stream_dependency_chain() {
    let mut state = state();
    let r1 = MockEvent::new();
    let r2 = MockEvent::new();
    state.record(MockStream(1), AccessMode::Read, Arc::clone(&r1));
    state.record(MockStream(2), AccessMode::Read, Arc::clone(&r2));

    let plan = state.plan_access(&MockStream(3), AccessMode::Write, never_complete);
    assert_eq!(plan.waits.len(), 2);
    let w = MockEvent::new();
    state.record(MockStream(3), AccessMode::Write, Arc::clone(&w));

    let plan = state.plan_access(&MockStream(4), AccessMode::Read, never_complete);
    assert_eq!(plan.waits.len(), 1);
    assert!(Arc::ptr_eq(&plan.waits[0], &w));
    assert_eq!(state.pending_readers(), 0);
}

#[test]
fn completed_readers_are_pruned() {
    let mut state = state();
    let read = MockEvent::new();
    state.record(MockStream(1), AccessMode::Read, Arc::clone(&read));
    read.complete();
    let plan = state.plan_access(&MockStream(2), AccessMode::Write, |event| {
        event.complete.load(Ordering::SeqCst)
    });
    assert!(plan.waits.is_empty());
    assert_eq!(state.pending_readers(), 0);
}

#[test]
fn completed_writer_is_pruned() {
    let mut state = state();
    let write = MockEvent::new();
    state.record(MockStream(1), AccessMode::Write, Arc::clone(&write));
    write.complete();
    let plan = state.plan_access(&MockStream(2), AccessMode::Read, |event| {
        event.complete.load(Ordering::SeqCst)
    });
    assert!(plan.waits.is_empty());
    assert!(!state.has_writer());
}

#[test]
fn identity_generation_and_context_are_distinct() {
    assert_ne!(identity(0), identity(1));
    let other_context = BufferIdentity::new(BufferId(1), 0, BufferContextId(2));
    assert_ne!(identity(0), other_context);
    let other_buffer = BufferIdentity::new(BufferId(2), 0, BufferContextId(1));
    assert_ne!(identity(0), other_buffer);
}

#[test]
fn reset_clears_old_access_state() {
    let mut state = state();
    let read = MockEvent::new();
    state.record(MockStream(1), AccessMode::Read, read);
    state.reset(identity(1));
    assert_eq!(state.pending_readers(), 0);
    assert!(!state.has_writer());
    assert_eq!(state.identity(), identity(1));
}

#[test]
fn access_mode_merge() {
    assert_eq!(AccessMode::Read.merge(AccessMode::Read), AccessMode::Read);
    assert_eq!(AccessMode::Read.merge(AccessMode::Write), AccessMode::Write);
    assert_eq!(
        AccessMode::ReadWrite.merge(AccessMode::Read),
        AccessMode::ReadWrite
    );
    assert_eq!(
        AccessMode::Write.merge(AccessMode::ReadWrite),
        AccessMode::Write
    );
}
