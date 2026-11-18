//! CUDA-independent event safety primitives for oxide-ml.
//!
//! This crate contains no CUDA dependency. It provides:
//!
//! - the `Unrecorded -> Recorded -> Completed` / `Failed` state machine,
//! - a context-bound event pool with a quarantine list that failed events can
//!   never leave,
//! - structured event errors and destroy diagnostics,
//! - the deferred buffer reclamation policy used by `CudaBufferPool`.
//!
//! The CUDA backend supplies a production [`EventFactory`] / [`CompletionEvent`]
//! implementation and the CPU tests supply an injectable mock. Both are
//! monomorphized generic backends; no dynamic dispatch is used.

pub mod buffer;
pub mod diagnostics;
pub mod error;
pub mod pool;
pub mod state;

pub use buffer::{ReclaimDecision, ReclaimVerdict, evaluate_completion};
pub use diagnostics::{
    DestroyOutcome, DestroyRecord, EventDiagnostics, EventDiagnosticsSnapshot, lock_recover,
};
pub use error::{EventError, EventOp};
pub use pool::{
    CleanupReport, CompletionEvent, EventFactory, EventLease, EventPool, EventPoolStats,
    PollReport, QuarantineRecord, RecordEvent, ReusableEvent,
};
pub use state::{ContextId, EventState, EventStateMachine};
