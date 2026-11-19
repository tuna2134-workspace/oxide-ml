use std::fmt;

/// Physical buffer allocation identity.
///
/// This is intentionally not a device pointer: pool reuse can produce the same
/// pointer for a different allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub u64);

/// Process-local context identity attached to a buffer allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferContextId(pub u64);

/// Identity of one buffer allocation generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferIdentity {
    pub id: BufferId,
    pub generation: u64,
    pub context_id: BufferContextId,
}

impl BufferIdentity {
    pub fn new(id: BufferId, generation: u64, context_id: BufferContextId) -> Self {
        BufferIdentity {
            id,
            generation,
            context_id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

impl AccessMode {
    pub fn is_write(self) -> bool {
        matches!(self, AccessMode::Write | AccessMode::ReadWrite)
    }

    /// Combines duplicate accesses to the same buffer within one launch.
    pub fn merge(self, other: AccessMode) -> AccessMode {
        match (self, other) {
            (AccessMode::Write, _) | (_, AccessMode::Write) => AccessMode::Write,
            (AccessMode::ReadWrite, _) | (_, AccessMode::ReadWrite) => AccessMode::ReadWrite,
            (AccessMode::Read, AccessMode::Read) => AccessMode::Read,
        }
    }
}

impl fmt::Display for AccessMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AccessMode::Read => f.write_str("read"),
            AccessMode::Write => f.write_str("write"),
            AccessMode::ReadWrite => f.write_str("read-write"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PendingAccess<E, S> {
    pub stream: S,
    pub mode: AccessMode,
    pub event: E,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AccessStats {
    pub event_waits: u64,
    pub same_stream_elided_waits: u64,
    pub cross_stream_waits: u64,
    pub read_read_concurrent: u64,
    pub pending_readers: usize,
    pub pending_writer: usize,
}

#[derive(Clone, Debug)]
pub struct AccessPlan<E> {
    /// Events that must be waited on before the launch.
    pub waits: Vec<E>,
    pub same_stream_elided: usize,
    pub cross_stream_waits: usize,
}

/// Per-buffer hazard state.
///
/// `E` is an opaque completion token (CUDA uses an Arc-held event + EventLease)
/// and `S` is a stream identity token. The state never calls CUDA itself; the
/// CUDA layer executes the returned waits and records the completion token.
pub struct BufferAccessState<E, S> {
    identity: BufferIdentity,
    readers: Vec<PendingAccess<E, S>>,
    writer: Option<PendingAccess<E, S>>,
    stats: AccessStats,
}

impl<E, S> BufferAccessState<E, S>
where
    E: Clone,
    S: Clone + PartialEq,
{
    pub fn new(identity: BufferIdentity) -> Self {
        BufferAccessState {
            identity,
            readers: Vec::new(),
            writer: None,
            stats: AccessStats::default(),
        }
    }

    pub fn identity(&self) -> BufferIdentity {
        self.identity
    }

    pub fn reset(&mut self, identity: BufferIdentity) {
        self.identity = identity;
        self.readers.clear();
        self.writer = None;
    }

    /// Plans waits for a new access. Completed tokens are pruned first.
    ///
    /// Same-stream hazards are elided because CUDA stream order already
    /// enforces them. Different-stream dependencies are returned as event
    /// tokens and must be waited on by the caller before launching.
    pub fn plan_access(
        &mut self,
        stream: &S,
        mode: AccessMode,
        mut is_complete: impl FnMut(&E) -> bool,
    ) -> AccessPlan<E> {
        self.readers.retain(|reader| !is_complete(&reader.event));
        if self
            .writer
            .as_ref()
            .is_some_and(|writer| is_complete(&writer.event))
        {
            self.writer = None;
        }

        let mut waits = Vec::new();
        let mut same_stream_elided = 0usize;
        let mut cross_stream_waits = 0usize;

        if let Some(writer) = &self.writer {
            if &writer.stream == stream {
                same_stream_elided += 1;
            } else {
                waits.push(writer.event.clone());
                cross_stream_waits += 1;
            }
        }

        if mode.is_write() {
            for reader in &self.readers {
                if &reader.stream == stream {
                    same_stream_elided += 1;
                } else {
                    waits.push(reader.event.clone());
                    cross_stream_waits += 1;
                }
            }
        }

        self.stats.event_waits = self.stats.event_waits.saturating_add(waits.len() as u64);
        self.stats.same_stream_elided_waits = self
            .stats
            .same_stream_elided_waits
            .saturating_add(same_stream_elided as u64);
        self.stats.cross_stream_waits = self
            .stats
            .cross_stream_waits
            .saturating_add(cross_stream_waits as u64);

        AccessPlan {
            waits,
            same_stream_elided,
            cross_stream_waits,
        }
    }

    /// Records a completed launch's access. The token must remain alive until
    /// the event completes (or is quarantined) so the event cannot be reused.
    pub fn record(&mut self, stream: S, mode: AccessMode, event: E) {
        match mode {
            AccessMode::Read => {
                // The read is ordered after any previous writer (either by
                // same-stream order or by an explicit wait). A later writer
                // only needs to wait for this read event.
                self.writer = None;
                if !self.readers.is_empty() {
                    self.stats.read_read_concurrent =
                        self.stats.read_read_concurrent.saturating_add(1);
                }
                self.readers.push(PendingAccess {
                    stream,
                    mode,
                    event,
                });
            }
            AccessMode::Write | AccessMode::ReadWrite => {
                // All prior readers/writers were either waited on or are
                // covered by same-stream ordering.
                self.readers.clear();
                self.writer = Some(PendingAccess {
                    stream,
                    mode,
                    event,
                });
            }
        }
        self.stats.pending_readers = self.readers.len();
        self.stats.pending_writer = usize::from(self.writer.is_some());
    }

    pub fn stats(&self) -> AccessStats {
        let mut stats = self.stats.clone();
        stats.pending_readers = self.readers.len();
        stats.pending_writer = usize::from(self.writer.is_some());
        stats
    }

    pub fn pending_readers(&self) -> usize {
        self.readers.len()
    }

    pub fn has_writer(&self) -> bool {
        self.writer.is_some()
    }
}
