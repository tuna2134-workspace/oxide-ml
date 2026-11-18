use std::sync::Arc;

use crate::error::EventError;
use crate::pool::CompletionEvent;
use crate::state::EventState;

/// Result of evaluating the completion events attached to a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReclaimDecision {
    /// Every completion event is explicitly `Completed` (or there are no
    /// events at all, meaning no GPU work was registered for the buffer).
    Reclaim,
    /// At least one event is unrecorded or still in flight. Keep the buffer
    /// out of the free list.
    Pending,
    /// At least one event failed or its completion could not be confirmed.
    /// The buffer must be quarantined and never reused.
    Quarantine,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReclaimVerdict {
    pub decision: ReclaimDecision,
    pub error: Option<EventError>,
}

impl ReclaimVerdict {
    pub fn reclaim() -> Self {
        ReclaimVerdict {
            decision: ReclaimDecision::Reclaim,
            error: None,
        }
    }

    pub fn pending() -> Self {
        ReclaimVerdict {
            decision: ReclaimDecision::Pending,
            error: None,
        }
    }

    pub fn quarantine(error: Option<EventError>) -> Self {
        ReclaimVerdict {
            decision: ReclaimDecision::Quarantine,
            error,
        }
    }
}

/// The single source of truth for deferred buffer reclamation.
///
/// A buffer is reclaimable iff every completion event is confirmed
/// `Completed`. `Unrecorded` and `Recorded` hold the buffer; `Failed` and any
/// query error quarantine it. Completion is never inferred from an event that
/// was not recorded.
pub fn evaluate_completion<E: CompletionEvent + ?Sized>(events: &[Arc<E>]) -> ReclaimVerdict {
    if events.is_empty() {
        return ReclaimVerdict::reclaim();
    }

    let mut incomplete = false;
    for event in events {
        match event.state() {
            EventState::Failed => {
                return ReclaimVerdict::quarantine(event.last_error());
            }
            EventState::Unrecorded => {
                // Never recorded means no confirmed completion. It is not a
                // failure, but the buffer must stay out of the free list.
                incomplete = true;
            }
            EventState::Completed => {}
            EventState::Recorded => match event.is_complete() {
                Ok(true) => {}
                Ok(false) => incomplete = true,
                Err(error) => {
                    // `is_complete` must have marked the event Failed. Even
                    // if a backend implementation did not, the error itself
                    // is enough to quarantine the buffer.
                    return ReclaimVerdict::quarantine(Some(error));
                }
            },
        }
    }

    if incomplete {
        ReclaimVerdict::pending()
    } else {
        ReclaimVerdict::reclaim()
    }
}
