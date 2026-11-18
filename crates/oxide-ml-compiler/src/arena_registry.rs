use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::arena_key::{ArenaDescriptor, ArenaKey};

/// Default upper bound on retained arenas per registry.
pub const DEFAULT_MAX_ARENAS: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaRegistryStats {
    /// Number of retained arenas (entries, not unique keys).
    pub arena_count: usize,
    pub lookup_count: u64,
    pub hit_count: u64,
    pub miss_count: u64,
    /// Lookups whose key existed but whose exact descriptor differed.
    pub collision_count: u64,
    pub capacity_rejection_count: u64,
    pub creation_failure_count: u64,
    pub poison_recovery_count: u64,
}

#[derive(Debug)]
pub enum ArenaRegistryError<E> {
    Factory(E),
    CapacityExceeded { limit: usize, arena_count: usize },
    InvalidCapacity(usize),
}

impl<E: fmt::Display> fmt::Display for ArenaRegistryError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArenaRegistryError::Factory(error) => write!(f, "arena creation failed: {error}"),
            ArenaRegistryError::CapacityExceeded { limit, arena_count } => write!(
                f,
                "arena registry capacity limit {limit} reached ({arena_count} arenas)"
            ),
            ArenaRegistryError::InvalidCapacity(capacity) => {
                write!(
                    f,
                    "arena slot-set capacity must be positive, got {capacity}"
                )
            }
        }
    }
}

impl<E: fmt::Display + fmt::Debug> std::error::Error for ArenaRegistryError<E> {}

struct RegistryEntry<A> {
    arena: Arc<A>,
    descriptor: Arc<ArenaDescriptor>,
    capacity: usize,
}

struct RegistryState<A> {
    arenas: HashMap<ArenaKey, Vec<Arc<RegistryEntry<A>>>>,
    creating: HashMap<ArenaKey, Arc<Mutex<()>>>,
    stats: ArenaRegistryStats,
}

impl<A> Default for RegistryState<A> {
    fn default() -> Self {
        RegistryState {
            arenas: HashMap::new(),
            creating: HashMap::new(),
            stats: ArenaRegistryStats::default(),
        }
    }
}

enum Lookup<A> {
    Hit(Arc<A>),
    Collision,
    Miss,
}

/// Generic, backend-independent arena registry.
///
/// Lookup uses `ArenaKey` as a fast index but reuses an arena only when the
/// exact `ArenaDescriptor` matches. A hash collision therefore creates a
/// separate arena instead of sharing one.
///
/// Concurrent lookups for the same key serialize on a per-key creation lock.
/// CUDA allocation runs outside the registry mutex, so unrelated keys are not
/// blocked. Creation failure leaves the registry unchanged.
pub struct ArenaRegistry<A> {
    inner: Mutex<RegistryState<A>>,
    max_arenas: usize,
    poison_recoveries: AtomicU64,
}

impl<A> ArenaRegistry<A> {
    pub fn new(max_arenas: usize) -> Self {
        ArenaRegistry {
            inner: Mutex::new(RegistryState::default()),
            max_arenas,
            poison_recoveries: AtomicU64::new(0),
        }
    }

    pub fn with_limit(max_arenas: usize) -> Self {
        Self::new(max_arenas)
    }

    fn lock_state(&self) -> MutexGuard<'_, RegistryState<A>> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                self.poison_recoveries.fetch_add(1, Ordering::Relaxed);
                poisoned.into_inner()
            }
        }
    }

    fn lock_creation<'a>(&self, lock: &'a Arc<Mutex<()>>) -> MutexGuard<'a, ()> {
        match lock.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                self.poison_recoveries.fetch_add(1, Ordering::Relaxed);
                poisoned.into_inner()
            }
        }
    }

    fn find(
        state: &RegistryState<A>,
        key: &ArenaKey,
        descriptor: &ArenaDescriptor,
        capacity: usize,
    ) -> Lookup<A> {
        let Some(entries) = state.arenas.get(key) else {
            return Lookup::Miss;
        };
        let mut collision = false;
        for entry in entries {
            if entry.descriptor.as_ref() == descriptor {
                if entry.capacity >= capacity {
                    return Lookup::Hit(Arc::clone(&entry.arena));
                }
            } else {
                collision = true;
            }
        }
        if collision {
            Lookup::Collision
        } else {
            Lookup::Miss
        }
    }

    /// Returns an existing arena for `key`/`descriptor` or creates one.
    ///
    /// `create` is called outside all registry mutexes. A matching entry that
    /// appears while creation is in flight (for example from another thread)
    /// wins and the newly created arena is dropped.
    pub fn get_or_create<F, E>(
        &self,
        key: ArenaKey,
        descriptor: ArenaDescriptor,
        capacity: usize,
        create: F,
    ) -> Result<Arc<A>, ArenaRegistryError<E>>
    where
        A: Send + Sync + 'static,
        F: FnOnce() -> Result<A, E>,
    {
        if capacity == 0 {
            return Err(ArenaRegistryError::InvalidCapacity(0));
        }

        {
            let mut state = self.lock_state();
            state.stats.lookup_count = state.stats.lookup_count.saturating_add(1);
            let lookup = Self::find(&state, &key, &descriptor, capacity);
            match lookup {
                Lookup::Hit(arena) => {
                    state.stats.hit_count = state.stats.hit_count.saturating_add(1);
                    return Ok(arena);
                }
                // Collision is counted once, after the per-key creation lock
                // has serialized concurrent lookups.
                Lookup::Collision | Lookup::Miss => {}
            }
        }

        let key_lock = {
            let mut state = self.lock_state();
            Arc::clone(
                state
                    .creating
                    .entry(key.clone())
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };
        let _creation_guard = self.lock_creation(&key_lock);

        {
            let mut state = self.lock_state();
            let lookup = Self::find(&state, &key, &descriptor, capacity);
            match lookup {
                Lookup::Hit(arena) => {
                    state.stats.hit_count = state.stats.hit_count.saturating_add(1);
                    return Ok(arena);
                }
                Lookup::Collision => {
                    state.stats.collision_count = state.stats.collision_count.saturating_add(1);
                }
                Lookup::Miss => {}
            }
            if state.stats.arena_count >= self.max_arenas {
                state.stats.capacity_rejection_count =
                    state.stats.capacity_rejection_count.saturating_add(1);
                return Err(ArenaRegistryError::CapacityExceeded {
                    limit: self.max_arenas,
                    arena_count: state.stats.arena_count,
                });
            }
        }

        let arena = match create() {
            Ok(arena) => Arc::new(arena),
            Err(error) => {
                let mut state = self.lock_state();
                state.stats.creation_failure_count =
                    state.stats.creation_failure_count.saturating_add(1);
                return Err(ArenaRegistryError::Factory(error));
            }
        };

        {
            let mut state = self.lock_state();
            if state.stats.arena_count >= self.max_arenas {
                state.stats.capacity_rejection_count =
                    state.stats.capacity_rejection_count.saturating_add(1);
                return Err(ArenaRegistryError::CapacityExceeded {
                    limit: self.max_arenas,
                    arena_count: state.stats.arena_count,
                });
            }
            let entry = Arc::new(RegistryEntry {
                arena: Arc::clone(&arena),
                descriptor: Arc::new(descriptor),
                capacity,
            });
            state.arenas.entry(key).or_default().push(entry);
            state.stats.arena_count = state.stats.arena_count.saturating_add(1);
            state.stats.miss_count = state.stats.miss_count.saturating_add(1);
        }
        Ok(arena)
    }

    pub fn stats(&self) -> ArenaRegistryStats {
        let state = self.lock_state();
        let mut stats = state.stats;
        stats.arena_count = state.stats.arena_count;
        stats.poison_recovery_count = self.poison_recoveries.load(Ordering::Relaxed);
        stats
    }

    pub fn len(&self) -> usize {
        self.lock_state().stats.arena_count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains_key(&self, key: &ArenaKey) -> bool {
        self.lock_state().arenas.contains_key(key)
    }

    pub fn max_arenas(&self) -> usize {
        self.max_arenas
    }
}

impl<A> Default for ArenaRegistry<A> {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ARENAS)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use oxide_ml_core::{DType, Shape};
    use oxide_ml_graph::GraphBuilder;

    use crate::arena_key::ArenaContextId;
    use crate::plan_hash::{CanonicalPlan, PlanHash};
    use crate::{
        ArenaDescriptor, ArenaKey, DeviceId, MEMORY_PLAN_VERSION, MemoryPlanner,
        canonicalize_memory_plan,
    };

    use super::*;

    struct MockArena {
        id: usize,
    }

    fn descriptor() -> ArenaDescriptor {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::from_static([4]), DType::F32);
        let out = builder.relu(x).unwrap();
        let graph = builder.finish(&[out]).unwrap();
        let plan = MemoryPlanner::default().plan(&graph).unwrap();
        let canonical_graph = graph.canonicalize().unwrap();
        let canonical_plan: CanonicalPlan = canonicalize_memory_plan(&graph, &plan).unwrap();
        ArenaDescriptor::new(
            canonical_graph,
            canonical_plan,
            DeviceId::new(0, None, "mock", (7, 5), 1),
            ArenaContextId(1),
            DType::F32,
        )
    }

    fn key(descriptor: &ArenaDescriptor) -> ArenaKey {
        ArenaKey::new(
            descriptor.graph.hash(),
            PlanHash([7; 32]),
            descriptor.device_id.clone(),
            descriptor.context_id,
            descriptor.dtype,
            MEMORY_PLAN_VERSION,
        )
    }

    #[test]
    fn poisoned_registry_mutex_is_recovered() {
        let registry = ArenaRegistry::<MockArena>::default();
        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            let _guard = registry.inner.lock().expect("test lock");
            panic!("poison registry state");
        }));
        assert!(poisoned.is_err());

        let descriptor = descriptor();
        let arena = registry
            .get_or_create(key(&descriptor), descriptor, 1, || {
                Ok::<MockArena, &'static str>(MockArena { id: 1 })
            })
            .expect("registry should recover");
        assert_eq!(arena.id, 1);
        assert!(registry.stats().poison_recovery_count >= 1);
    }

    #[test]
    fn creation_failure_leaves_registry_empty() {
        let registry = ArenaRegistry::<MockArena>::default();
        let descriptor = descriptor();
        let result = registry.get_or_create(key(&descriptor), descriptor, 1, || {
            Err::<MockArena, &'static str>("boom")
        });
        assert!(matches!(result, Err(ArenaRegistryError::Factory("boom"))));
        assert!(registry.is_empty());
        assert_eq!(registry.stats().creation_failure_count, 1);
    }

    #[test]
    fn capacity_limit_is_reported_without_eviction() {
        let registry = ArenaRegistry::<MockArena>::new(1);
        let first = descriptor();
        let created = registry
            .get_or_create(key(&first), first.clone(), 1, || {
                Ok::<MockArena, &'static str>(MockArena { id: 1 })
            })
            .expect("first arena");
        assert_eq!(registry.len(), 1);

        let mut second = descriptor();
        second.context_id = ArenaContextId(99);
        let result = registry.get_or_create(key(&second), second, 1, || {
            Ok::<MockArena, &'static str>(MockArena { id: 2 })
        });
        assert!(matches!(
            result,
            Err(ArenaRegistryError::CapacityExceeded { .. })
        ));
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.stats().capacity_rejection_count, 1);
        // Existing arena is still present and reusable.
        let reused = registry
            .get_or_create(key(&first), first, 1, || {
                Ok::<MockArena, &'static str>(MockArena { id: 3 })
            })
            .expect("existing arena");
        assert_eq!(reused.id, created.id);
    }

    #[test]
    fn concurrent_same_key_creates_one_arena() {
        let registry = Arc::new(ArenaRegistry::<MockArena>::default());
        let descriptor = descriptor();
        let key = key(&descriptor);
        let creates = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..8 {
                let registry = Arc::clone(&registry);
                let descriptor = descriptor.clone();
                let key = key.clone();
                let creates = &creates;
                handles.push(scope.spawn(move || {
                    registry
                        .get_or_create(key, descriptor, 1, || {
                            creates.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(std::time::Duration::from_millis(30));
                            Ok::<MockArena, &'static str>(MockArena { id: 1 })
                        })
                        .unwrap()
                }));
            }
            for handle in handles {
                assert_eq!(handle.join().unwrap().id, 1);
            }
        });
        assert_eq!(creates.load(Ordering::SeqCst), 1);
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn concurrent_different_keys_create_separate_arenas() {
        let registry = Arc::new(ArenaRegistry::<MockArena>::default());
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for id in 0..8u64 {
                let registry = Arc::clone(&registry);
                handles.push(scope.spawn(move || {
                    let mut descriptor = descriptor();
                    descriptor.context_id = ArenaContextId(100 + id);
                    let key = key(&descriptor);
                    registry
                        .get_or_create(key, descriptor, 1, || {
                            Ok::<MockArena, &'static str>(MockArena { id: id as usize })
                        })
                        .unwrap()
                }));
            }
            for handle in handles {
                let _ = handle.join().unwrap();
            }
        });
        assert_eq!(registry.len(), 8);
    }
}
