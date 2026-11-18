use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::error::EventError;

const MAX_RECENT_DESTROY_RECORDS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DestroyOutcome {
    Destroyed,
    Failed,
    Skipped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DestroyRecord {
    pub outcome: DestroyOutcome,
    pub error: Option<EventError>,
}

/// Shared diagnostic sink for event resource destruction and poison recovery.
///
/// Every event holds an `Arc` to the pool's sink, so a failure that happens
/// when the last event reference is dropped (possibly after the pool itself
/// has gone away) is still observable by anyone holding the sink.
#[derive(Debug, Default)]
pub struct EventDiagnostics {
    destroy_attempts: AtomicUsize,
    destroy_failures: AtomicUsize,
    destroy_skipped: AtomicUsize,
    poison_recoveries: AtomicUsize,
    recent: Mutex<Vec<DestroyRecord>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventDiagnosticsSnapshot {
    pub destroy_attempts: usize,
    pub destroy_failures: usize,
    pub destroy_skipped: usize,
    pub poison_recoveries: usize,
    pub recent_destroy_records: Vec<DestroyRecord>,
}

impl EventDiagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_destroy_success(&self) {
        self.destroy_attempts.fetch_add(1, Ordering::Relaxed);
        self.push_recent(DestroyRecord {
            outcome: DestroyOutcome::Destroyed,
            error: None,
        });
    }

    pub fn record_destroy_failure(&self, error: EventError) {
        self.destroy_attempts.fetch_add(1, Ordering::Relaxed);
        self.destroy_failures.fetch_add(1, Ordering::Relaxed);
        self.push_recent(DestroyRecord {
            outcome: DestroyOutcome::Failed,
            error: Some(error),
        });
    }

    /// Records that destroy was deliberately not attempted because the
    /// context is no longer known to be valid.
    pub fn record_destroy_skipped(&self, error: Option<EventError>) {
        self.destroy_skipped.fetch_add(1, Ordering::Relaxed);
        self.push_recent(DestroyRecord {
            outcome: DestroyOutcome::Skipped,
            error,
        });
    }

    pub fn record_poison_recovery(&self) {
        self.poison_recoveries.fetch_add(1, Ordering::Relaxed);
    }

    fn push_recent(&self, record: DestroyRecord) {
        let mut recent = match self.recent.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                self.poison_recoveries.fetch_add(1, Ordering::Relaxed);
                poisoned.into_inner()
            }
        };
        if recent.len() >= MAX_RECENT_DESTROY_RECORDS {
            recent.remove(0);
        }
        recent.push(record);
    }

    pub fn snapshot(&self) -> EventDiagnosticsSnapshot {
        let recent = match self.recent.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => {
                self.poison_recoveries.fetch_add(1, Ordering::Relaxed);
                poisoned.into_inner().clone()
            }
        };
        EventDiagnosticsSnapshot {
            destroy_attempts: self.destroy_attempts.load(Ordering::Relaxed),
            destroy_failures: self.destroy_failures.load(Ordering::Relaxed),
            destroy_skipped: self.destroy_skipped.load(Ordering::Relaxed),
            poison_recoveries: self.poison_recoveries.load(Ordering::Relaxed),
            recent_destroy_records: recent,
        }
    }
}

/// Locks a mutex without panicking on poisoning.
///
/// Recovery is recorded in the diagnostics sink. This is used by event and
/// quarantine bookkeeping, which must remain usable during `Drop` even if a
/// previous holder panicked.
pub fn lock_recover<'a, T>(
    mutex: &'a Mutex<T>,
    diagnostics: &EventDiagnostics,
) -> MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            diagnostics.record_poison_recovery();
            poisoned.into_inner()
        }
    }
}
