use std::collections::HashMap;
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use cuda_core::{CudaContext, IntoResult};
use oxide_ml_compiler::{ArenaContextId, DeviceId};

use crate::device::CudaDevice;
use crate::error::{CudaError, Result};

/// Process-local CUDA device identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceIdentity {
    id: DeviceId,
}

impl DeviceIdentity {
    pub fn from_device(device: &CudaDevice) -> Result<Self> {
        let info = device.info();
        let uuid = query_device_uuid(device.context()).ok();
        Ok(DeviceIdentity {
            id: DeviceId::new(
                info.ordinal as u32,
                uuid,
                info.name.clone(),
                (
                    info.compute_capability.0.max(0) as u16,
                    info.compute_capability.1.max(0) as u16,
                ),
                info.total_memory as u64,
            ),
        })
    }

    pub fn id(&self) -> &DeviceId {
        &self.id
    }
}

fn query_device_uuid(context: &Arc<CudaContext>) -> Result<[u8; 16]> {
    context.bind_to_thread()?;
    let mut uuid = MaybeUninit::uninit();
    unsafe {
        cuda_core::sys::cuDeviceGetUuid_v2(uuid.as_mut_ptr(), context.cu_device()).result()?;
        let uuid = uuid.assume_init();
        let mut bytes = [0u8; 16];
        for (index, byte) in uuid.bytes.iter().enumerate() {
            bytes[index] = *byte as u8;
        }
        Ok(bytes)
    }
}

static NEXT_CONTEXT_ID: AtomicU64 = AtomicU64::new(1);

type ContextMap = HashMap<usize, (u64, Weak<ContextEntry>)>;

fn context_registry() -> &'static Mutex<ContextMap> {
    static REGISTRY: OnceLock<Mutex<ContextMap>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

struct ContextEntry {
    id: ArenaContextId,
    raw: usize,
    context: Arc<CudaContext>,
}

impl Drop for ContextEntry {
    fn drop(&mut self) {
        let Ok(registry) = context_registry().lock() else {
            // A poisoned context map is still structurally sound; recovering
            // is safer than panicking during event/arena teardown.
            let mut registry = context_registry()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if registry
                .get(&self.raw)
                .is_some_and(|(id, _)| *id == self.id.0)
            {
                registry.remove(&self.raw);
            }
            return;
        };
        let mut registry = registry;
        if registry
            .get(&self.raw)
            .is_some_and(|(id, _)| *id == self.id.0)
        {
            registry.remove(&self.raw);
        }
    }
}

/// Process-local CUDA context identity token.
///
/// The id is allocated from a monotonic counter and is never reused, even if
/// the driver reuses a raw `CUcontext` handle after destruction. The token
/// holds `Arc<CudaContext>`, so the context cannot be destroyed while an
/// identity exists. The id is process-local and must not be persisted.
#[derive(Clone)]
pub struct ContextIdentity {
    entry: Arc<ContextEntry>,
}

impl ContextIdentity {
    pub fn from_context(context: &Arc<CudaContext>) -> Self {
        let raw = context.cu_ctx() as usize;
        let mut registry = context_registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, weak)) = registry.get(&raw)
            && let Some(entry) = weak.upgrade()
        {
            return ContextIdentity { entry };
        }
        let id = ArenaContextId(NEXT_CONTEXT_ID.fetch_add(1, Ordering::Relaxed));
        let entry = Arc::new(ContextEntry {
            id,
            raw,
            context: Arc::clone(context),
        });
        registry.insert(raw, (id.0, Arc::downgrade(&entry)));
        ContextIdentity { entry }
    }

    pub fn id(&self) -> ArenaContextId {
        self.entry.id
    }

    /// Raw driver handle. Diagnostic only; never hash or persist this value.
    pub fn raw_handle(&self) -> usize {
        self.entry.raw
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.entry.context
    }
}

impl PartialEq for ContextIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.entry.id == other.entry.id
    }
}

impl Eq for ContextIdentity {}

impl fmt::Debug for ContextIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextIdentity")
            .field("id", &self.entry.id)
            .field("raw_handle", &self.entry.raw)
            .finish_non_exhaustive()
    }
}

impl CudaDevice {
    pub fn identity(&self) -> Result<DeviceIdentity> {
        DeviceIdentity::from_device(self)
    }

    pub fn context_identity(&self) -> ContextIdentity {
        ContextIdentity::from_context(self.context())
    }
}

/// Helper for callers that need to reject a context bound to a different
/// device before building an arena key.
pub fn ensure_context_device(context: &Arc<CudaContext>, device: &DeviceIdentity) -> Result<()> {
    let ordinal = context.ordinal() as u32;
    if ordinal != device.id().ordinal {
        return Err(CudaError::Internal(format!(
            "context ordinal {ordinal} does not match device ordinal {}",
            device.id().ordinal
        )));
    }
    Ok(())
}
