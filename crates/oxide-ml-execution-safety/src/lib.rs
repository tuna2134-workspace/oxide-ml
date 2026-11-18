//! CUDA-independent async execution lifecycle and ownership helpers.
//!
//! The CUDA backend keeps the concrete resources (`CudaRun`, stream, context,
//! arena lease, event lease) in `CudaExecution`; this crate owns the state
//! machine and diagnostics shared with the CPU tests.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use oxide_ml_event_safety::{EventDiagnostics, lock_recover};

/// Lifecycle of one submitted asynchronous execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionState {
    Submitted,
    Running,
    Completed,
    Collected,
    Failed,
}

impl ExecutionState {
    pub fn is_finished(self) -> bool {
        matches!(
            self,
            ExecutionState::Completed | ExecutionState::Collected | ExecutionState::Failed
        )
    }
}

impl fmt::Display for ExecutionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionState::Submitted => f.write_str("submitted"),
            ExecutionState::Running => f.write_str("running"),
            ExecutionState::Completed => f.write_str("completed"),
            ExecutionState::Collected => f.write_str("collected"),
            ExecutionState::Failed => f.write_str("failed"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionStateError {
    AlreadyCollected,
    Failed,
    InvalidTransition {
        from: ExecutionState,
        to: ExecutionState,
    },
}

impl fmt::Display for ExecutionStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutionStateError::AlreadyCollected => {
                f.write_str("execution outputs were already collected")
            }
            ExecutionStateError::Failed => f.write_str("execution is in the Failed state"),
            ExecutionStateError::InvalidTransition { from, to } => {
                write!(f, "invalid execution transition {from} -> {to}")
            }
        }
    }
}

impl std::error::Error for ExecutionStateError {}

/// Pure execution lifecycle state machine.
#[derive(Clone, Debug)]
pub struct ExecutionLifecycle {
    state: ExecutionState,
}

impl Default for ExecutionLifecycle {
    fn default() -> Self {
        ExecutionLifecycle {
            state: ExecutionState::Submitted,
        }
    }
}

impl ExecutionLifecycle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> ExecutionState {
        self.state
    }

    /// Called when a non-blocking poll finds the execution still in flight.
    pub fn note_pending(&mut self) -> Result<ExecutionState, ExecutionStateError> {
        match self.state {
            ExecutionState::Submitted => {
                self.state = ExecutionState::Running;
                Ok(self.state)
            }
            ExecutionState::Running => Ok(self.state),
            ExecutionState::Completed => Ok(self.state),
            ExecutionState::Collected => Err(ExecutionStateError::AlreadyCollected),
            ExecutionState::Failed => Err(ExecutionStateError::Failed),
        }
    }

    pub fn mark_completed(&mut self) -> Result<(), ExecutionStateError> {
        match self.state {
            ExecutionState::Submitted | ExecutionState::Running | ExecutionState::Completed => {
                self.state = ExecutionState::Completed;
                Ok(())
            }
            ExecutionState::Collected => Err(ExecutionStateError::AlreadyCollected),
            ExecutionState::Failed => Err(ExecutionStateError::Failed),
        }
    }

    /// Marks a failed execution. GPU resources may still be in use, so this
    /// state never implies that buffers can be released.
    pub fn mark_failed(&mut self) -> Result<(), ExecutionStateError> {
        match self.state {
            ExecutionState::Failed => Ok(()),
            ExecutionState::Collected => Err(ExecutionStateError::AlreadyCollected),
            _ => {
                self.state = ExecutionState::Failed;
                Ok(())
            }
        }
    }

    /// Validates that outputs may be collected. The caller performs the D2H
    /// transfer and only then calls [`Self::mark_collected`].
    pub fn begin_collect(&mut self) -> Result<(), ExecutionStateError> {
        match self.state {
            ExecutionState::Completed => Ok(()),
            ExecutionState::Collected => Err(ExecutionStateError::AlreadyCollected),
            ExecutionState::Failed => Err(ExecutionStateError::Failed),
            ExecutionState::Submitted | ExecutionState::Running => {
                Err(ExecutionStateError::InvalidTransition {
                    from: self.state,
                    to: ExecutionState::Collected,
                })
            }
        }
    }

    pub fn mark_collected(&mut self) -> Result<(), ExecutionStateError> {
        match self.state {
            ExecutionState::Completed => {
                self.state = ExecutionState::Collected;
                Ok(())
            }
            ExecutionState::Collected => Err(ExecutionStateError::AlreadyCollected),
            ExecutionState::Failed => Err(ExecutionStateError::Failed),
            ExecutionState::Submitted | ExecutionState::Running => {
                Err(ExecutionStateError::InvalidTransition {
                    from: self.state,
                    to: ExecutionState::Collected,
                })
            }
        }
    }
}

/// Shared, poison-recovering lifecycle wrapper used by `CudaExecution`.
#[derive(Clone)]
pub struct SharedExecutionLifecycle {
    inner: Arc<Mutex<ExecutionLifecycle>>,
    diagnostics: Arc<EventDiagnostics>,
}

impl SharedExecutionLifecycle {
    pub fn new() -> Self {
        SharedExecutionLifecycle {
            inner: Arc::new(Mutex::new(ExecutionLifecycle::new())),
            diagnostics: Arc::new(EventDiagnostics::new()),
        }
    }

    pub fn with_diagnostics(diagnostics: Arc<EventDiagnostics>) -> Self {
        SharedExecutionLifecycle {
            inner: Arc::new(Mutex::new(ExecutionLifecycle::new())),
            diagnostics,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ExecutionLifecycle> {
        lock_recover(&self.inner, &self.diagnostics)
    }

    pub fn state(&self) -> ExecutionState {
        self.lock().state()
    }

    pub fn note_pending(&self) -> Result<ExecutionState, ExecutionStateError> {
        self.lock().note_pending()
    }

    pub fn mark_completed(&self) -> Result<(), ExecutionStateError> {
        self.lock().mark_completed()
    }

    pub fn mark_failed(&self) -> Result<(), ExecutionStateError> {
        self.lock().mark_failed()
    }

    pub fn begin_collect(&self) -> Result<(), ExecutionStateError> {
        self.lock().begin_collect()
    }

    pub fn mark_collected(&self) -> Result<(), ExecutionStateError> {
        self.lock().mark_collected()
    }

    pub fn diagnostics(&self) -> Arc<EventDiagnostics> {
        Arc::clone(&self.diagnostics)
    }
}

impl Default for SharedExecutionLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

/// Cleanup/error counters that must remain observable without panicking.
#[derive(Debug, Default)]
pub struct ExecutionDiagnostics {
    cleanup_failures: AtomicUsize,
    context_binding_failures: AtomicUsize,
    d2h_failures: AtomicUsize,
    last_cleanup_error: Mutex<Option<String>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionDiagnosticsSnapshot {
    pub cleanup_failures: usize,
    pub context_binding_failures: usize,
    pub d2h_failures: usize,
    pub last_cleanup_error: Option<String>,
}

impl ExecutionDiagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_cleanup_failure(&self, message: impl Into<String>) {
        self.cleanup_failures.fetch_add(1, Ordering::Relaxed);
        let mut last = self
            .last_cleanup_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *last = Some(message.into());
    }

    pub fn record_context_binding_failure(&self) {
        self.context_binding_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_d2h_failure(&self) {
        self.d2h_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> ExecutionDiagnosticsSnapshot {
        let last_cleanup_error = self
            .last_cleanup_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        ExecutionDiagnosticsSnapshot {
            cleanup_failures: self.cleanup_failures.load(Ordering::Relaxed),
            context_binding_failures: self.context_binding_failures.load(Ordering::Relaxed),
            d2h_failures: self.d2h_failures.load(Ordering::Relaxed),
            last_cleanup_error,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    #[test]
    fn state_machine_transitions() {
        let mut lifecycle = ExecutionLifecycle::new();
        assert_eq!(lifecycle.state(), ExecutionState::Submitted);
        assert_eq!(lifecycle.note_pending().unwrap(), ExecutionState::Running);
        lifecycle.mark_completed().unwrap();
        assert_eq!(lifecycle.state(), ExecutionState::Completed);
        lifecycle.begin_collect().unwrap();
        lifecycle.mark_collected().unwrap();
        assert_eq!(lifecycle.state(), ExecutionState::Collected);
        assert_eq!(
            lifecycle.begin_collect().unwrap_err(),
            ExecutionStateError::AlreadyCollected
        );
    }

    #[test]
    fn failure_from_submitted_or_running() {
        let mut submitted = ExecutionLifecycle::new();
        submitted.mark_failed().unwrap();
        assert_eq!(submitted.state(), ExecutionState::Failed);

        let mut running = ExecutionLifecycle::new();
        running.note_pending().unwrap();
        running.mark_failed().unwrap();
        assert_eq!(running.state(), ExecutionState::Failed);
        assert_eq!(
            running.begin_collect().unwrap_err(),
            ExecutionStateError::Failed
        );
    }

    #[test]
    fn poisoned_shared_lifecycle_is_recovered() {
        let shared = SharedExecutionLifecycle::new();
        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            let _guard = shared.inner.lock().expect("test lock");
            panic!("poison lifecycle");
        }));
        assert!(poisoned.is_err());
        shared.note_pending().unwrap();
        assert_eq!(shared.state(), ExecutionState::Running);
        assert!(shared.diagnostics().snapshot().poison_recoveries >= 1);
    }
}
