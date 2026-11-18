use std::fmt;

/// The CUDA event operation that produced an [`EventError`].
///
/// This is intentionally limited to event operations. The rest of the CUDA
/// runtime keeps using the backend's native error type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventOp {
    Create,
    Record,
    Query,
    Synchronize,
    Destroy,
    Context,
}

impl EventOp {
    pub fn as_str(self) -> &'static str {
        match self {
            EventOp::Create => "create",
            EventOp::Record => "record",
            EventOp::Query => "query",
            EventOp::Synchronize => "synchronize",
            EventOp::Destroy => "destroy",
            EventOp::Context => "context",
        }
    }
}

impl fmt::Display for EventOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A structured event failure.
///
/// `code` is the raw CUDA driver result when the failure came from the driver.
/// Logical state-machine failures (for example re-recording a failed event)
/// leave it as `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EventError {
    pub op: EventOp,
    pub code: Option<i64>,
    pub message: String,
}

impl EventError {
    pub fn new(op: EventOp, message: impl Into<String>) -> Self {
        EventError {
            op,
            code: None,
            message: message.into(),
        }
    }

    pub fn with_code(op: EventOp, code: i64, message: impl Into<String>) -> Self {
        EventError {
            op,
            code: Some(code),
            message: message.into(),
        }
    }

    /// Conservative classification of context-loss errors.
    ///
    /// `CUDA_ERROR_INVALID_CONTEXT` (201) and
    /// `CUDA_ERROR_CONTEXT_IS_DESTROYED` (709) mean the driver context is no
    /// longer usable. Cleanup paths must not call further CUDA APIs for such
    /// errors; they record a skipped destroy instead.
    pub fn is_context_lost(&self) -> bool {
        matches!(self.code, Some(201) | Some(709))
    }

    pub fn summary(&self) -> String {
        match self.code {
            Some(code) => format!("{} failure (CUDA error {code}): {}", self.op, self.message),
            None => format!("{} failure: {}", self.op, self.message),
        }
    }
}

impl fmt::Display for EventError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

impl std::error::Error for EventError {}
