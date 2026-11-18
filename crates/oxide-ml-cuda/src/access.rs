use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cuda_core::CudaStream;
use oxide_ml_event_safety::{EventDiagnostics, lock_recover};
pub use oxide_ml_execution_safety::AccessMode;
use oxide_ml_execution_safety::{
    AccessPlan, AccessStats, BufferAccessState as CoreAccessState, BufferId, BufferIdentity,
};

use crate::buffer::CudaTensor;
use crate::error::{CudaError, Result};
use crate::event::{CudaEvent, EventLease, EventPool};

/// Stream identity token.
///
/// The token keeps the stream alive, so a destroyed stream cannot have its
/// address reused while an access is still pending.
#[derive(Clone)]
pub struct StreamToken {
    id: u64,
    stream: Arc<CudaStream>,
}

impl StreamToken {
    pub fn new(stream: &Arc<CudaStream>) -> Self {
        StreamToken {
            id: Arc::as_ptr(stream) as usize as u64,
            stream: Arc::clone(stream),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl PartialEq for StreamToken {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for StreamToken {}

/// Completion token for one access. It owns an EventLease so the event pool
/// cannot reuse the event while a hazard still references it.
pub struct AccessCompletion {
    event: Arc<CudaEvent>,
    _lease: EventLease,
}

impl AccessCompletion {
    pub fn event(&self) -> &Arc<CudaEvent> {
        &self.event
    }
}

pub type AccessState = CoreAccessState<Arc<AccessCompletion>, StreamToken>;

/// One buffer access request for a kernel launch.
///
/// The request holds no `Arc<PooledBuffer>`/`Weak<PooledBuffer>`: those would
/// make `CudaTensor::buffer_mut()` see shared storage during the launch.
/// Completion events are attached to tensors by the kernel wrapper after the
/// launch closure returns.
#[derive(Clone)]
pub struct AccessRequest {
    state: Arc<Mutex<AccessState>>,
    diagnostics: Arc<EventDiagnostics>,
    identity: BufferIdentity,
    mode: AccessMode,
}

impl AccessRequest {
    fn new(tensor: &CudaTensor, mode: AccessMode) -> Self {
        let storage = tensor.storage();
        AccessRequest {
            state: Arc::clone(storage.access_state()),
            diagnostics: Arc::clone(storage.pool_diagnostics()),
            identity: storage.identity(),
            mode,
        }
    }

    pub fn read(tensor: &CudaTensor) -> Self {
        Self::new(tensor, AccessMode::Read)
    }

    pub fn write(tensor: &CudaTensor) -> Self {
        Self::new(tensor, AccessMode::Write)
    }

    pub fn read_write(tensor: &CudaTensor) -> Self {
        Self::new(tensor, AccessMode::ReadWrite)
    }
}

/// RAII guard for one recorded access.
///
/// Dropping the guard does not remove the hazard: the buffer access state owns
/// an `Arc<AccessCompletion>` until a later conflicting access waits for it.
pub struct AccessGuard {
    identity: BufferIdentity,
    mode: AccessMode,
    token: Arc<AccessCompletion>,
}

impl AccessGuard {
    pub fn identity(&self) -> BufferIdentity {
        self.identity
    }

    pub fn mode(&self) -> AccessMode {
        self.mode
    }

    pub fn event(&self) -> &Arc<CudaEvent> {
        self.token.event()
    }

    pub fn completion(&self) -> &Arc<AccessCompletion> {
        &self.token
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaunchAccessReport {
    pub buffers: usize,
    pub event_waits: usize,
    pub same_stream_elided: usize,
    pub cross_stream_waits: usize,
    pub total_waits: u64,
    pub total_same_stream_elided: u64,
    pub total_cross_stream_waits: u64,
    pub read_read_concurrent: u64,
    pub pending_readers: usize,
    pub pending_writers: usize,
}

pub struct LaunchCompletion<R> {
    pub value: R,
    pub report: LaunchAccessReport,
    pub guards: Vec<AccessGuard>,
    pub event: Arc<CudaEvent>,
}

/// Executes one kernel launch under buffer access guards.
///
/// Buffer strong references are upgraded only after the launch closure has
/// returned; this keeps `CudaTensor::buffer_mut()` valid during the launch.
pub fn launch_with_access<R>(
    event_pool: &EventPool,
    stream: &Arc<CudaStream>,
    requests: &[AccessRequest],
    launch: impl FnOnce() -> Result<R>,
) -> Result<LaunchCompletion<R>> {
    let mut merged: HashMap<
        BufferIdentity,
        (Arc<Mutex<AccessState>>, Arc<EventDiagnostics>, AccessMode),
    > = HashMap::new();
    for request in requests {
        merged
            .entry(request.identity)
            .and_modify(|(_, _, mode)| *mode = mode.merge(request.mode))
            .or_insert_with(|| {
                (
                    Arc::clone(&request.state),
                    Arc::clone(&request.diagnostics),
                    request.mode,
                )
            });
    }
    let mut ordered = merged.into_iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(identity, _)| (identity.id, identity.generation));

    let identities = ordered
        .iter()
        .map(|(identity, _)| *identity)
        .collect::<Vec<_>>();
    let states = ordered
        .iter()
        .map(|(_, (state, _, _))| Arc::clone(state))
        .collect::<Vec<_>>();
    let diagnostics = ordered
        .iter()
        .map(|(_, (_, diagnostics, _))| Arc::clone(diagnostics))
        .collect::<Vec<_>>();
    let modes = ordered
        .iter()
        .map(|(_, (_, _, mode))| *mode)
        .collect::<Vec<_>>();
    let stream_token = StreamToken::new(stream);

    let mut state_guards = Vec::with_capacity(identities.len());
    let mut plans: Vec<AccessPlan<Arc<AccessCompletion>>> = Vec::with_capacity(identities.len());
    let mut report = LaunchAccessReport {
        buffers: identities.len(),
        ..LaunchAccessReport::default()
    };

    for index in 0..identities.len() {
        let mut state = lock_recover(&states[index], &diagnostics[index]);
        if state.identity() != identities[index] {
            return Err(CudaError::Internal(format!(
                "buffer generation mismatch: state {:?}, request {:?}",
                state.identity(),
                identities[index]
            )));
        }
        let plan = state.plan_access(&stream_token, modes[index], |token| {
            token.event().is_complete().unwrap_or(false)
        });
        for event in &plan.waits {
            event.event().wait_on(stream)?;
        }
        report.event_waits += plan.waits.len();
        report.same_stream_elided += plan.same_stream_elided;
        report.cross_stream_waits += plan.cross_stream_waits;
        plans.push(plan);
        state_guards.push(state);
    }

    let completion = match event_pool.acquire() {
        Ok(completion) => completion,
        Err(error) => {
            return Err(error);
        }
    };

    let value = match launch() {
        Ok(value) => value,
        Err(error) => {
            return Err(error);
        }
    };

    completion.record(stream)?;
    let event = match completion.arc() {
        Ok(event) => event,
        Err(error) => {
            return Err(error);
        }
    };

    let token = Arc::new(AccessCompletion {
        event: Arc::clone(&event),
        _lease: completion,
    });
    let mut guards = Vec::with_capacity(identities.len());
    for (index, state) in state_guards.iter_mut().enumerate() {
        state.record(stream_token.clone(), modes[index], Arc::clone(&token));
        let stats = state.stats();
        report.total_waits = report.total_waits.saturating_add(stats.event_waits);
        report.total_same_stream_elided = report
            .total_same_stream_elided
            .saturating_add(stats.same_stream_elided_waits);
        report.total_cross_stream_waits = report
            .total_cross_stream_waits
            .saturating_add(stats.cross_stream_waits);
        report.read_read_concurrent = report
            .read_read_concurrent
            .saturating_add(stats.read_read_concurrent);
        report.pending_readers += stats.pending_readers;
        report.pending_writers += stats.pending_writer;
        guards.push(AccessGuard {
            identity: identities[index],
            mode: modes[index],
            token: Arc::clone(&token),
        });
    }
    Ok(LaunchCompletion {
        value,
        report,
        guards,
        event,
    })
}

pub fn access_stats(state: &AccessState) -> AccessStats {
    state.stats()
}

pub fn access_identity(state: &AccessState) -> BufferIdentity {
    state.identity()
}

pub fn access_id(identity: BufferIdentity) -> BufferId {
    identity.id
}

/// Backwards-compatible name for the existing access mode.
pub type BufferAccess = AccessMode;
