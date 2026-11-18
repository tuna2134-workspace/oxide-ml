use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use oxide_ml_execution_safety::{ExecutionDiagnostics, ExecutionState, SharedExecutionLifecycle};

#[derive(Default)]
struct Counters {
    context_drops: AtomicUsize,
    buffer_frees: AtomicUsize,
    buffer_pending: AtomicUsize,
    arena_pending: AtomicUsize,
    execution_drops: AtomicUsize,
    arena_finish_failures: AtomicUsize,
}

struct MockContext {
    counters: Arc<Counters>,
    binding_ok: AtomicBool,
}

impl MockContext {
    fn new(counters: Arc<Counters>) -> Self {
        MockContext {
            counters,
            binding_ok: AtomicBool::new(true),
        }
    }

    fn bind(&self) -> Result<(), &'static str> {
        if self.binding_ok.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err("context binding failure")
        }
    }
}

impl Drop for MockContext {
    fn drop(&mut self) {
        self.counters.context_drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct MockBuffer {
    counters: Arc<Counters>,
    complete: AtomicBool,
}

impl MockBuffer {
    fn new(counters: Arc<Counters>) -> Self {
        MockBuffer {
            counters,
            complete: AtomicBool::new(false),
        }
    }

    fn mark_complete(&self) {
        self.complete.store(true, Ordering::SeqCst);
    }
}

impl Drop for MockBuffer {
    fn drop(&mut self) {
        if self.complete.load(Ordering::SeqCst) {
            self.counters.buffer_frees.fetch_add(1, Ordering::SeqCst);
        } else {
            self.counters.buffer_pending.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[derive(Clone)]
struct MockOutput {
    buffer: Arc<MockBuffer>,
}

struct MockCompletion {
    ready: AtomicBool,
    query_fail: AtomicBool,
    sync_fail: AtomicBool,
}

impl Default for MockCompletion {
    fn default() -> Self {
        MockCompletion {
            ready: AtomicBool::new(false),
            query_fail: AtomicBool::new(false),
            sync_fail: AtomicBool::new(false),
        }
    }
}

impl MockCompletion {
    fn query(&self) -> Result<bool, &'static str> {
        if self.query_fail.load(Ordering::SeqCst) {
            Err("query failure")
        } else {
            Ok(self.ready.load(Ordering::SeqCst))
        }
    }

    fn synchronize(&self) -> Result<(), &'static str> {
        if self.sync_fail.load(Ordering::SeqCst) {
            Err("synchronize failure")
        } else {
            self.ready.store(true, Ordering::SeqCst);
            Ok(())
        }
    }
}

struct MockArena {
    counters: Arc<Counters>,
    released: AtomicUsize,
}

impl MockArena {
    fn new(counters: Arc<Counters>) -> Self {
        MockArena {
            counters,
            released: AtomicUsize::new(0),
        }
    }
}

struct MockArenaLease {
    arena: Arc<MockArena>,
    finished: bool,
    fail_finish: bool,
}

impl MockArenaLease {
    fn finish(&mut self) -> Result<(), &'static str> {
        if self.finished {
            return Err("lease already finished");
        }
        self.finished = true;
        if self.fail_finish {
            self.arena
                .counters
                .arena_finish_failures
                .fetch_add(1, Ordering::SeqCst);
            return Err("arena finish failure");
        }
        self.arena.released.fetch_add(1, Ordering::SeqCst);
        self.arena
            .counters
            .arena_pending
            .fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct MockExecutor {
    context: Arc<MockContext>,
}

impl Drop for MockExecutor {
    fn drop(&mut self) {
        let _ = &self.context;
    }
}

struct MockExecution {
    lifecycle: SharedExecutionLifecycle,
    diagnostics: Arc<ExecutionDiagnostics>,
    context: Arc<MockContext>,
    outputs: Vec<MockOutput>,
    arena: Option<MockArenaLease>,
    completion: MockCompletion,
    counters: Arc<Counters>,
    fail_transfer: bool,
}

impl MockExecution {
    fn new(context: Arc<MockContext>, counters: Arc<Counters>, with_arena: bool) -> Self {
        let lifecycle = SharedExecutionLifecycle::new();
        let diagnostics = Arc::new(ExecutionDiagnostics::new());
        let output = MockOutput {
            buffer: Arc::new(MockBuffer::new(Arc::clone(&counters))),
        };
        let arena = with_arena.then(|| {
            let arena = Arc::new(MockArena::new(Arc::clone(&counters)));
            MockArenaLease {
                arena,
                finished: false,
                fail_finish: false,
            }
        });
        MockExecution {
            lifecycle,
            diagnostics,
            context,
            outputs: vec![output],
            arena,
            completion: MockCompletion::default(),
            counters,
            fail_transfer: false,
        }
    }

    fn finish_arena(&mut self) {
        if let Some(mut lease) = self.arena.take()
            && let Err(error) = lease.finish()
        {
            self.diagnostics.record_cleanup_failure(error);
        }
    }

    fn poll(&self) -> Result<bool, &'static str> {
        self.lifecycle
            .note_pending()
            .map_err(|_| "execution is not pollable")?;
        match self.completion.query() {
            Ok(true) => {
                self.lifecycle
                    .mark_completed()
                    .map_err(|_| "execution completion state failure")?;
                Ok(true)
            }
            Ok(false) => Ok(false),
            Err(error) => {
                let _ = self.lifecycle.mark_failed();
                Err(error)
            }
        }
    }

    fn wait(&mut self) -> Result<(), &'static str> {
        if let Err(error) = self.context.bind() {
            self.diagnostics.record_context_binding_failure();
            let _ = self.lifecycle.mark_failed();
            return Err(error);
        }
        if let Err(error) = self.completion.synchronize() {
            let _ = self.lifecycle.mark_failed();
            return Err(error);
        }
        self.lifecycle
            .mark_completed()
            .map_err(|_| "execution completion state failure")?;
        for output in &self.outputs {
            output.buffer.mark_complete();
        }
        self.finish_arena();
        Ok(())
    }

    fn collect(&mut self) -> Result<Option<usize>, &'static str> {
        if self.fail_transfer {
            self.diagnostics.record_d2h_failure();
            let _ = self.lifecycle.mark_failed();
            return Err("D2H failure");
        }
        self.lifecycle
            .begin_collect()
            .map_err(|_| "execution is not collectable")?;
        self.lifecycle
            .mark_collected()
            .map_err(|_| "execution collect state failure")?;
        Ok(Some(self.outputs.len()))
    }
}

impl Drop for MockExecution {
    fn drop(&mut self) {
        // Emulate ExecutionResources::drop: defer arena slots, then let
        // buffers and context drop in their own order. No synchronization.
        self.finish_arena();
        self.counters.execution_drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn executor_drop_does_not_destroy_execution_context() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let executor = MockExecutor {
        context: Arc::clone(&context),
    };
    let mut execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), false);
    assert!(!execution.poll().unwrap());
    assert_eq!(execution.lifecycle.state(), ExecutionState::Running);
    drop(executor);
    assert_eq!(counters.context_drops.load(Ordering::SeqCst), 0);
    execution.wait().unwrap();
    assert_eq!(execution.lifecycle.state(), ExecutionState::Completed);
    drop(execution);
    drop(context);
    assert_eq!(counters.context_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn execution_drop_while_running_defers_buffer_and_arena() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), true);
    assert!(!execution.poll().unwrap());
    assert_eq!(execution.lifecycle.state(), ExecutionState::Running);
    drop(execution);
    assert_eq!(counters.execution_drops.load(Ordering::SeqCst), 1);
    assert_eq!(counters.buffer_pending.load(Ordering::SeqCst), 1);
    assert_eq!(counters.buffer_frees.load(Ordering::SeqCst), 0);
    assert_eq!(counters.arena_pending.load(Ordering::SeqCst), 1);
}

#[test]
fn external_output_outlives_execution() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let mut execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), false);
    execution.wait().unwrap();
    let output = execution.outputs[0].clone();
    drop(execution);
    assert_eq!(counters.buffer_frees.load(Ordering::SeqCst), 0);
    drop(output);
    assert_eq!(counters.buffer_frees.load(Ordering::SeqCst), 1);
}

#[test]
fn multiple_output_owners_release_once() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let mut execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), false);
    execution.wait().unwrap();
    let output_a = execution.outputs[0].clone();
    let output_b = output_a.clone();
    execution.outputs.clear();
    drop(output_a);
    assert_eq!(counters.buffer_frees.load(Ordering::SeqCst), 0);
    drop(output_b);
    assert_eq!(counters.buffer_frees.load(Ordering::SeqCst), 1);
}

#[test]
fn arena_lease_outlives_registry_entry() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let mut execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), true);
    // The registry's entry is the only other owner of the arena. Dropping it
    // must not invalidate the lease held by the running execution.
    let registry = Arc::clone(&execution.arena.as_ref().unwrap().arena);
    drop(registry);
    execution.wait().unwrap();
    assert_eq!(counters.arena_pending.load(Ordering::SeqCst), 1);
}

#[test]
fn query_failure_marks_failed_and_keeps_resources() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), false);
    execution
        .completion
        .query_fail
        .store(true, Ordering::SeqCst);
    assert!(execution.poll().is_err());
    assert_eq!(execution.lifecycle.state(), ExecutionState::Failed);
    assert_eq!(counters.buffer_pending.load(Ordering::SeqCst), 0);
    drop(execution);
    assert_eq!(counters.buffer_pending.load(Ordering::SeqCst), 1);
}

#[test]
fn synchronize_failure_marks_failed() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let mut execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), false);
    execution.completion.sync_fail.store(true, Ordering::SeqCst);
    assert!(execution.wait().is_err());
    assert_eq!(execution.lifecycle.state(), ExecutionState::Failed);
}

#[test]
fn context_binding_failure_marks_failed() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let mut execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), false);
    context.binding_ok.store(false, Ordering::SeqCst);
    assert!(execution.wait().is_err());
    assert_eq!(execution.lifecycle.state(), ExecutionState::Failed);
}

#[test]
fn d2h_failure_keeps_output_ownership_and_marks_failed() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let mut execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), false);
    execution.wait().unwrap();
    execution.fail_transfer = true;
    assert!(execution.collect().is_err());
    assert_eq!(execution.lifecycle.state(), ExecutionState::Failed);
    assert_eq!(execution.diagnostics.snapshot().d2h_failures, 1);
    assert_eq!(counters.buffer_frees.load(Ordering::SeqCst), 0);
}

#[test]
fn double_collect_is_rejected() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let mut execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), false);
    execution.wait().unwrap();
    assert_eq!(execution.collect().unwrap(), Some(1));
    assert!(execution.collect().is_err());
    assert_eq!(execution.lifecycle.state(), ExecutionState::Collected);
}

#[test]
fn arena_finish_failure_is_diagnosed_without_panic() {
    let counters = Arc::new(Counters::default());
    let context = Arc::new(MockContext::new(Arc::clone(&counters)));
    let mut execution = MockExecution::new(Arc::clone(&context), Arc::clone(&counters), true);
    execution.arena.as_mut().unwrap().fail_finish = true;
    execution.wait().unwrap();
    assert_eq!(counters.arena_finish_failures.load(Ordering::SeqCst), 1);
    assert_eq!(execution.diagnostics.snapshot().cleanup_failures, 1);
}
