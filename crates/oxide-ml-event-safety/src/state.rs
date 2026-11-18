use crate::error::{EventError, EventOp};

/// Identity of the CUDA context an event was created in.
///
/// The value is opaque to this crate. The CUDA backend uses the allocation
/// address of its `Arc<CudaContext>`; two live `Arc`s to the same context have
/// the same id and different contexts have different ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContextId(pub u64);

impl ContextId {
    pub fn new(value: usize) -> Self {
        ContextId(value as u64)
    }
}

/// Event lifecycle state.
///
/// State transitions are listed in `docs/phase7f13-final-report.md` and
/// enforced by [`EventStateMachine`]. The important safety property is that a
/// `Failed` event is terminal: it can never be recorded or treated as
/// completed again.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventState {
    Unrecorded,
    Recorded,
    Completed,
    Failed,
}

impl EventState {
    /// `true` only for an explicitly confirmed completion.
    ///
    /// `Unrecorded` is deliberately not complete: an event that was never
    /// recorded has no completion to confirm.
    pub fn is_completed(self) -> bool {
        matches!(self, EventState::Completed)
    }

    pub fn is_failed(self) -> bool {
        matches!(self, EventState::Failed)
    }

    /// Whether the event may enter a pool free list.
    ///
    /// `Failed` is never reusable. `Recorded` may only become reusable after
    /// a successful query marks it `Completed`.
    pub fn is_free_list_candidate(self) -> bool {
        matches!(self, EventState::Unrecorded | EventState::Completed)
    }
}

/// Pure state machine shared by the production CUDA event and the fault
/// injection mock.
///
/// The caller performs the CUDA API call while holding the state lock, then
/// reports the actual result through `record_success`/`record_failure` (and
/// the query/synchronize equivalents). State is therefore only advanced after
/// the API result is known.
#[derive(Clone, Debug)]
pub struct EventStateMachine {
    state: EventState,
    generation: u64,
    last_error: Option<EventError>,
}

impl Default for EventStateMachine {
    fn default() -> Self {
        EventStateMachine {
            state: EventState::Unrecorded,
            generation: 0,
            last_error: None,
        }
    }
}

impl EventStateMachine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> EventState {
        self.state
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn last_error(&self) -> Option<EventError> {
        self.last_error.clone()
    }

    /// Validates that a record call may start.
    ///
    /// A failed event is terminal and cannot be re-recorded. `Unrecorded`,
    /// `Recorded` and `Completed` events may be recorded; recording a
    /// `Completed` event reuses it (the pool normally resets it to
    /// `Unrecorded` first).
    pub fn begin_record(&self) -> Result<(), EventError> {
        if self.state == EventState::Failed {
            Err(EventError::new(
                EventOp::Record,
                "failed event cannot be re-recorded",
            ))
        } else {
            Ok(())
        }
    }

    pub fn record_success(&mut self) {
        debug_assert!(self.state != EventState::Failed);
        self.state = EventState::Recorded;
        self.generation = self.generation.saturating_add(1);
        self.last_error = None;
    }

    pub fn record_failure(&mut self, error: EventError) {
        self.state = EventState::Failed;
        self.last_error = Some(error);
    }

    /// Records a successful `query` result.
    ///
    /// `complete == false` leaves a `Recorded` event in `Recorded`.
    pub fn query_success(&mut self, complete: bool) {
        if self.state == EventState::Recorded {
            self.state = if complete {
                EventState::Completed
            } else {
                EventState::Recorded
            };
        } else if complete && self.state == EventState::Completed {
            self.state = EventState::Completed;
        }
        // Unrecorded is not promoted to Completed: querying an unrecorded
        // event must never fabricate a completion.
    }

    pub fn query_failure(&mut self, error: EventError) {
        // A query error is an unconfirmed completion. The event is failed and
        // must not be reused or treated as completed.
        if self.state != EventState::Failed {
            self.state = EventState::Failed;
            self.last_error = Some(error);
        }
    }

    pub fn synchronize_success(&mut self) {
        if self.state != EventState::Failed {
            self.state = EventState::Completed;
            self.last_error = None;
        }
    }

    pub fn synchronize_failure(&mut self, error: EventError) {
        if self.state != EventState::Failed {
            self.state = EventState::Failed;
            self.last_error = Some(error);
        }
    }

    /// Moves a completed (or never-recorded) event back to `Unrecorded` for
    /// reuse. A failed event is never reset.
    pub fn reset_for_reuse(&mut self) -> Result<(), EventError> {
        match self.state {
            EventState::Unrecorded | EventState::Completed => {
                self.state = EventState::Unrecorded;
                self.last_error = None;
                Ok(())
            }
            EventState::Failed => Err(EventError::new(
                EventOp::Record,
                "failed event cannot be reused",
            )),
            EventState::Recorded => Err(EventError::new(
                EventOp::Record,
                "recorded event was not confirmed complete before reuse",
            )),
        }
    }

    /// Marks the event failed for a failure that happened outside record /
    /// query / synchronize (for example a failed stream wait).
    pub fn mark_failed(&mut self, error: EventError) {
        if self.state != EventState::Failed {
            self.state = EventState::Failed;
            self.last_error = Some(error);
        }
    }
}
