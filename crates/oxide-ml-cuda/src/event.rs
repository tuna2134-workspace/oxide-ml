use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex};

use cuda_core::{CudaContext, CudaStream, IntoResult};

use oxide_ml_event_safety::{
    CompletionEvent, ContextId, EventDiagnostics, EventError, EventFactory,
    EventLease as CoreEventLease, EventOp, EventPool as CoreEventPool, EventStateMachine,
    RecordEvent, ReusableEvent, lock_recover,
};

use crate::error::{CudaError, Result};

pub use oxide_ml_event_safety::{
    CleanupReport, EventPoolStats, EventState, PollReport, QuarantineRecord,
};

/// Context-bound CUDA event wrapper with an explicit failure state.
///
/// The raw driver handle is owned by this type and destroyed exactly once in
/// `Drop`. The state machine only advances after the corresponding CUDA call
/// succeeds. A `Failed` event is terminal.
#[derive(Clone)]
pub struct CudaEvent {
    inner: Arc<CudaEventInner>,
}

struct CudaEventInner {
    context: Arc<CudaContext>,
    context_id: ContextId,
    raw: RawCudaEvent,
    machine: Mutex<EventStateMachine>,
    diagnostics: Arc<EventDiagnostics>,
}

impl CudaEvent {
    pub fn new(context: &Arc<CudaContext>) -> Result<Self> {
        Self::new_with_diagnostics(context, Arc::new(EventDiagnostics::new()))
    }

    fn new_with_diagnostics(
        context: &Arc<CudaContext>,
        diagnostics: Arc<EventDiagnostics>,
    ) -> Result<Self> {
        let raw = RawCudaEvent::create(context, Arc::clone(&diagnostics))?;
        Ok(CudaEvent {
            inner: Arc::new(CudaEventInner {
                context: Arc::clone(context),
                context_id: ContextId::new(Arc::as_ptr(context) as usize),
                raw,
                machine: Mutex::new(EventStateMachine::new()),
                diagnostics,
            }),
        })
    }

    fn machine(&self) -> std::sync::MutexGuard<'_, EventStateMachine> {
        lock_recover(&self.inner.machine, &self.inner.diagnostics)
    }

    fn driver_error(op: EventOp, error: cuda_core::DriverError) -> EventError {
        EventError::with_code(op, error.0 as i64, format!("{error}"))
    }

    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        let mut machine = self.machine();
        machine.begin_record()?;
        if !Arc::ptr_eq(&self.inner.context, stream.context()) {
            let error = EventError::new(
                EventOp::Record,
                "stream belongs to a different CUDA context",
            );
            machine.record_failure(error.clone());
            return Err(error.into());
        }
        match self.inner.raw.record(stream) {
            Ok(()) => {
                machine.record_success();
                Ok(())
            }
            Err(error) => {
                machine.record_failure(error.clone());
                Err(error.into())
            }
        }
    }

    pub fn is_complete(&self) -> Result<bool> {
        let mut machine = self.machine();
        match machine.state() {
            EventState::Unrecorded => Ok(false),
            EventState::Completed => Ok(true),
            EventState::Failed => Err(machine
                .last_error()
                .unwrap_or_else(|| EventError::new(EventOp::Query, "event is failed"))
                .into()),
            EventState::Recorded => match self.inner.raw.query() {
                Ok(true) => {
                    machine.query_success(true);
                    Ok(true)
                }
                Ok(false) => {
                    machine.query_success(false);
                    Ok(false)
                }
                Err(error) => {
                    machine.query_failure(error.clone());
                    Err(error.into())
                }
            },
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        let mut machine = self.machine();
        match machine.state() {
            EventState::Unrecorded => Err(EventError::new(
                EventOp::Synchronize,
                "cannot synchronize an unrecorded event",
            )
            .into()),
            EventState::Failed => Err(machine
                .last_error()
                .unwrap_or_else(|| EventError::new(EventOp::Synchronize, "event is failed"))
                .into()),
            EventState::Completed => Ok(()),
            EventState::Recorded => match self.inner.raw.synchronize() {
                Ok(()) => {
                    machine.synchronize_success();
                    Ok(())
                }
                Err(error) => {
                    machine.synchronize_failure(error.clone());
                    Err(error.into())
                }
            },
        }
    }

    /// Enqueues a wait for this event on `stream`.
    ///
    /// `Unrecorded` and `Completed` events are already complete or represent
    /// no work, so no CUDA call is needed. A failed wait marks the event
    /// failed: the caller must not assume the requested ordering happened.
    pub fn wait_on(&self, stream: &CudaStream) -> Result<()> {
        if !Arc::ptr_eq(&self.inner.context, stream.context()) {
            return Err(EventError::new(
                EventOp::Context,
                "stream belongs to a different CUDA context",
            )
            .into());
        }
        let mut machine = self.machine();
        match machine.state() {
            EventState::Unrecorded | EventState::Completed => Ok(()),
            EventState::Failed => Err(machine
                .last_error()
                .unwrap_or_else(|| EventError::new(EventOp::Query, "event is failed"))
                .into()),
            EventState::Recorded => match self.inner.raw.wait(stream) {
                Ok(()) => Ok(()),
                Err(error) => {
                    machine.mark_failed(error.clone());
                    Err(error.into())
                }
            },
        }
    }

    pub fn state(&self) -> EventState {
        self.machine().state()
    }

    pub fn generation(&self) -> u64 {
        self.machine().generation()
    }

    pub fn last_error(&self) -> Option<EventError> {
        self.machine().last_error()
    }

    pub fn context_id(&self) -> ContextId {
        self.inner.context_id
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.inner.context
    }

    pub fn cu_event(&self) -> cuda_core::sys::CUevent {
        self.inner.raw.handle()
    }
}

impl ReusableEvent for CudaEvent {
    fn reset_for_reuse(&self) -> std::result::Result<(), EventError> {
        self.machine().reset_for_reuse()
    }
}

impl CompletionEvent for CudaEvent {
    fn state(&self) -> EventState {
        CudaEvent::state(self)
    }

    fn generation(&self) -> u64 {
        CudaEvent::generation(self)
    }

    fn last_error(&self) -> Option<EventError> {
        CudaEvent::last_error(self)
    }

    fn context_id(&self) -> ContextId {
        self.inner.context_id
    }

    fn is_complete(&self) -> std::result::Result<bool, EventError> {
        CudaEvent::is_complete(self).map_err(|error| match error {
            CudaError::EventFailure(error) => error,
            other => EventError::new(EventOp::Query, other.to_string()),
        })
    }

    fn synchronize(&self) -> std::result::Result<(), EventError> {
        CudaEvent::synchronize(self).map_err(|error| match error {
            CudaError::EventFailure(error) => error,
            other => EventError::new(EventOp::Synchronize, other.to_string()),
        })
    }
}

impl RecordEvent<CudaStream> for CudaEvent {
    fn record(&self, stream: &CudaStream) -> std::result::Result<(), EventError> {
        CudaEvent::record(self, stream).map_err(|error| match error {
            CudaError::EventFailure(error) => error,
            other => EventError::new(EventOp::Record, other.to_string()),
        })
    }
}

/// Raw CUDA event handle plus its owning context.
///
/// The driver contract for `cuEventDestroy` is explicit: an event may be
/// destroyed before it is complete; the call does not block and the driver
/// releases resources asynchronously at completion. The event also holds
/// `Arc<CudaContext>`, so the context cannot be destroyed first.
struct RawCudaEvent {
    handle: cuda_core::sys::CUevent,
    context: Arc<CudaContext>,
    diagnostics: Arc<EventDiagnostics>,
    destroyed: bool,
}

// SAFETY: `CUevent` is an opaque driver handle that is safe to pass between
// threads as long as the owning context is bound before each use. Every
// operation below calls `bind_to_thread` first and the event is `Sync` by
// interior driver design; the Rust-side state is protected by `CudaEventInner`.
unsafe impl Send for RawCudaEvent {}
unsafe impl Sync for RawCudaEvent {}

impl RawCudaEvent {
    fn create(context: &Arc<CudaContext>, diagnostics: Arc<EventDiagnostics>) -> Result<Self> {
        context.bind_to_thread()?;
        let mut handle = MaybeUninit::uninit();
        unsafe {
            cuda_core::sys::cuEventCreate(
                handle.as_mut_ptr(),
                cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT,
            )
            .result()?;
            Ok(RawCudaEvent {
                handle: handle.assume_init(),
                context: Arc::clone(context),
                diagnostics,
                destroyed: false,
            })
        }
    }

    fn handle(&self) -> cuda_core::sys::CUevent {
        self.handle
    }

    fn record(&self, stream: &CudaStream) -> std::result::Result<(), EventError> {
        self.context
            .bind_to_thread()
            .map_err(|error| CudaEvent::driver_error(EventOp::Record, error))?;
        // SAFETY: the handle is valid and the owning context is current. The
        // stream context was checked by the caller.
        unsafe { cuda_core::sys::cuEventRecord(self.handle, stream.cu_stream()) }
            .result()
            .map_err(|error| CudaEvent::driver_error(EventOp::Record, error))
    }

    fn query(&self) -> std::result::Result<bool, EventError> {
        self.context
            .bind_to_thread()
            .map_err(|error| CudaEvent::driver_error(EventOp::Query, error))?;
        match unsafe { cuda_core::sys::cuEventQuery(self.handle) }.result() {
            Ok(()) => Ok(true),
            Err(error) if error.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_NOT_READY => {
                Ok(false)
            }
            Err(error) => Err(CudaEvent::driver_error(EventOp::Query, error)),
        }
    }

    fn synchronize(&self) -> std::result::Result<(), EventError> {
        self.context
            .bind_to_thread()
            .map_err(|error| CudaEvent::driver_error(EventOp::Synchronize, error))?;
        // SAFETY: the handle is valid and the owning context is current.
        unsafe { cuda_core::sys::cuEventSynchronize(self.handle) }
            .result()
            .map_err(|error| CudaEvent::driver_error(EventOp::Synchronize, error))
    }

    fn wait(&self, stream: &CudaStream) -> std::result::Result<(), EventError> {
        self.context
            .bind_to_thread()
            .map_err(|error| CudaEvent::driver_error(EventOp::Query, error))?;
        // SAFETY: both handles are valid for this context, which is current.
        unsafe {
            cuda_core::sys::cuStreamWaitEvent(
                stream.cu_stream(),
                self.handle,
                cuda_core::sys::CUevent_wait_flags_enum_CU_EVENT_WAIT_DEFAULT,
            )
        }
        .result()
        .map_err(|error| CudaEvent::driver_error(EventOp::Query, error))
    }
}

impl Drop for RawCudaEvent {
    fn drop(&mut self) {
        if self.destroyed {
            return;
        }
        self.destroyed = true;
        if let Err(error) = self.context.bind_to_thread() {
            // Do not call into a context that cannot be made current.
            self.diagnostics
                .record_destroy_skipped(Some(CudaEvent::driver_error(EventOp::Destroy, error)));
            return;
        }
        // SAFETY: the handle was created by this context and is destroyed
        // exactly once. CUDA explicitly permits destroying an event before it
        // is complete; resource release is deferred by the driver.
        match unsafe { cuda_core::sys::cuEventDestroy_v2(self.handle) }.result() {
            Ok(()) => self.diagnostics.record_destroy_success(),
            Err(error) => self
                .diagnostics
                .record_destroy_failure(CudaEvent::driver_error(EventOp::Destroy, error)),
        }
    }
}

struct CudaEventFactory {
    context: Arc<CudaContext>,
    context_id: ContextId,
}

impl CudaEventFactory {
    fn new(context: Arc<CudaContext>) -> Self {
        let context_id = ContextId::new(Arc::as_ptr(&context) as usize);
        CudaEventFactory {
            context,
            context_id,
        }
    }
}

impl EventFactory<CudaEvent> for CudaEventFactory {
    fn context_id(&self) -> ContextId {
        self.context_id
    }

    fn create(
        &self,
        diagnostics: Arc<EventDiagnostics>,
    ) -> std::result::Result<CudaEvent, EventError> {
        CudaEvent::new_with_diagnostics(&self.context, diagnostics).map_err(|error| match error {
            CudaError::EventFailure(error) => error,
            other => EventError::new(EventOp::Create, other.to_string()),
        })
    }
}

type CorePool = CoreEventPool<CudaEventFactory, CudaEvent, CudaStream>;
type CoreLease = CoreEventLease<CudaEventFactory, CudaEvent, CudaStream>;

/// Event pool bound to a CUDA context, with free / pending / quarantine lists.
pub struct EventPool {
    inner: CorePool,
}

impl EventPool {
    pub fn new(context: Arc<CudaContext>) -> Self {
        EventPool {
            inner: CorePool::new(CudaEventFactory::new(context)),
        }
    }

    pub fn context_id(&self) -> ContextId {
        self.inner.context_id()
    }

    pub fn acquire(&self) -> Result<EventLease> {
        Ok(EventLease {
            inner: self.inner.acquire()?,
        })
    }

    pub fn poll(&self) -> PollReport {
        self.inner.poll()
    }

    pub fn stats(&self) -> EventPoolStats {
        self.inner.stats()
    }

    pub fn quarantine_diagnostics(&self) -> Vec<QuarantineRecord> {
        self.inner.quarantine_diagnostics()
    }

    pub fn shutdown(&self) -> CleanupReport {
        self.inner.shutdown()
    }

    pub fn diagnostics(&self) -> oxide_ml_event_safety::EventDiagnosticsSnapshot {
        self.inner.diagnostics_snapshot()
    }
}

pub struct EventLease {
    inner: CoreLease,
}

impl EventLease {
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        Ok(self.inner.record(stream)?)
    }

    pub fn is_complete(&self) -> Result<bool> {
        Ok(self.inner.is_complete()?)
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.inner.synchronize()?)
    }

    pub fn arc(&self) -> Result<Arc<CudaEvent>> {
        Ok(self.inner.arc()?)
    }

    pub fn id(&self) -> Option<u64> {
        self.inner.id()
    }

    pub fn state(&self) -> Option<EventState> {
        self.inner.state()
    }

    pub fn last_error(&self) -> Option<EventError> {
        self.inner.last_error()
    }
}
