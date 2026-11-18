use std::sync::Arc;

use cuda_core::CudaStream;

use crate::error::Result;
use crate::event::CudaEvent;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    Read,
    Write,
    ReadWrite,
}

/// Per-buffer access state. Same-stream order is left to CUDA; different
/// streams receive explicit `wait_event` calls.
#[derive(Default)]
pub struct BufferAccessState {
    last_write: Option<Arc<CudaEvent>>,
    last_write_stream: Option<Arc<CudaStream>>,
    pending_reads: Vec<(Arc<CudaEvent>, Arc<CudaStream>)>,
}

impl BufferAccessState {
    pub fn wait_for(&self, stream: &Arc<CudaStream>, access: BufferAccess) -> Result<usize> {
        let mut waits = 0usize;
        if matches!(
            access,
            BufferAccess::Write | BufferAccess::ReadWrite | BufferAccess::Read
        ) {
            if let (Some(event), Some(previous)) = (&self.last_write, &self.last_write_stream) {
                if !Arc::ptr_eq(previous, stream) {
                    event.wait_on(stream)?;
                    waits += 1;
                }
            }
        }
        if matches!(access, BufferAccess::Write | BufferAccess::ReadWrite) {
            for (event, previous) in &self.pending_reads {
                if !Arc::ptr_eq(previous, stream) {
                    event.wait_on(stream)?;
                    waits += 1;
                }
            }
        }
        Ok(waits)
    }

    pub fn record(&mut self, stream: Arc<CudaStream>, access: BufferAccess, event: Arc<CudaEvent>) {
        match access {
            BufferAccess::Read => self.pending_reads.push((event, stream)),
            BufferAccess::Write | BufferAccess::ReadWrite => {
                self.last_write = Some(event);
                self.last_write_stream = Some(stream);
                self.pending_reads.clear();
            }
        }
    }

    pub fn pending_reads(&self) -> usize {
        self.pending_reads.len()
    }
}
