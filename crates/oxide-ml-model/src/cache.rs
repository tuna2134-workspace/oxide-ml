//! Compile cache for model graphs.
//!
//! The key is the canonical graph hash of the *unoptimized* graph plus every
//! input that can change lowering: input signature, dtype, train/eval mode,
//! compiler options, backend, target architecture, kernel ABI and cache
//! version. Two models with identical structure and inputs share one compiled
//! graph; a different shape or backend cannot collide.
//!
//! Compilation happens outside the cache lock. A per-key in-flight marker plus
//! a condition variable ensures concurrent callers wait for one compilation
//! instead of compiling the same key twice. Entries are evicted in LRU order.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Condvar, LazyLock, Mutex};

use oxide_ml_core::{DType, Error, Result};
use oxide_ml_graph::graph_structure_hash;
use oxide_ml_kernel::KERNEL_ABI_VERSION;

use crate::module::ModuleMode;
use crate::runtime::{BuiltModel, CompiledModel, InputSpec, ModelCompileOptions};

/// Bump when cache-key composition or compiled-artifact semantics change.
pub const COMPILE_CACHE_VERSION: u32 = 1;
/// Bump when the compiler/codegen changes in a way that invalidates artifacts.
pub const COMPILER_VERSION: u32 = 1;
/// Default number of cached compiled models.
pub const DEFAULT_CAPACITY: usize = 32;
/// Backend identifier for the reference CPU executor.
pub const CPU_BACKEND: &str = "cpu";
const CPU_ARCH: &str = "portable";

/// One graph input in a form that implements `Hash`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InputSignature {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: DType,
    pub batch: bool,
}

impl InputSignature {
    pub fn from_spec(spec: &InputSpec) -> Self {
        InputSignature {
            name: spec.name.clone(),
            dims: spec.dims.clone(),
            dtype: spec.dtype,
            batch: spec.batch,
        }
    }
}

/// Cheap structural identity used before the graph is built.
///
/// A hit here returns the compiled model without rebuilding the graph. On a
/// miss the graph is built, hashed canonically, and looked up again under
/// [`CompileKey`]; the structural key then aliases the graph-keyed artifact.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StructuralKey {
    pub cache_version: u32,
    pub compiler_version: u32,
    pub model: String,
    pub structure: String,
    pub inputs: Vec<InputSignature>,
    pub mode: ModuleMode,
    pub optimize: bool,
    pub memory_plan: bool,
    pub backend: String,
    pub architecture: String,
    pub kernel_abi: u32,
    pub backward: bool,
}

/// Either a structural (pre-build) or canonical-graph (post-build) identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CacheKey {
    Structural(Box<StructuralKey>),
    Graph(Box<CompileKey>),
}

/// Stable cache identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompileKey {
    pub cache_version: u32,
    pub compiler_version: u32,
    /// Canonical graph hash (BLAKE3, hex).
    pub graph: String,
    pub inputs: Vec<InputSignature>,
    pub mode: ModuleMode,
    pub optimize: bool,
    pub memory_plan: bool,
    pub backend: String,
    pub architecture: String,
    pub kernel_abi: u32,
    /// Whether a backward graph was requested.
    pub backward: bool,
}

impl CompileKey {
    pub fn new(
        built: &BuiltModel,
        inputs: &[InputSpec],
        options: &ModelCompileOptions,
        backward: bool,
    ) -> Result<Self> {
        let hash = graph_structure_hash(&built.graph).map_err(|error| {
            Error::invalid_graph(format!("canonical graph hash failed: {error}"))
        })?;
        Ok(CompileKey {
            cache_version: COMPILE_CACHE_VERSION,
            compiler_version: COMPILER_VERSION,
            graph: hash.to_hex(),
            inputs: inputs.iter().map(InputSignature::from_spec).collect(),
            mode: options.mode,
            optimize: options.optimize,
            memory_plan: options.memory_plan,
            backend: CPU_BACKEND.to_string(),
            architecture: CPU_ARCH.to_string(),
            kernel_abi: KERNEL_ABI_VERSION,
            backward,
        })
    }

    /// Replace the backend/architecture after construction (used by the CUDA
    /// crate so it can key its own artifacts through the same structure).
    pub fn with_backend(
        mut self,
        backend: impl Into<String>,
        architecture: impl Into<String>,
    ) -> Self {
        self.backend = backend.into();
        self.architecture = architecture.into();
        self
    }
}

/// Cache counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub entries: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub compilations: u64,
    pub evictions: u64,
    pub insert_conflicts: u64,
    pub invalidations: u64,
    pub compile_failures: u64,
}

struct State {
    entries: HashMap<CacheKey, Arc<CompiledModel>>,
    order: VecDeque<CacheKey>,
    in_flight: HashSet<CacheKey>,
    /// Structural key -> canonical graph key.
    aliases: HashMap<CacheKey, CacheKey>,
    capacity: usize,
    hits: u64,
    misses: u64,
    compilations: u64,
    evictions: u64,
    insert_conflicts: u64,
    invalidations: u64,
    compile_failures: u64,
}

/// Thread-safe LRU cache of compiled models.
pub struct CompileCache {
    state: Mutex<State>,
    wake: Condvar,
}

impl CompileCache {
    pub fn new(capacity: usize) -> Self {
        CompileCache {
            state: Mutex::new(State {
                entries: HashMap::new(),
                order: VecDeque::new(),
                in_flight: HashSet::new(),
                aliases: HashMap::new(),
                capacity: capacity.max(1),
                hits: 0,
                misses: 0,
                compilations: 0,
                evictions: 0,
                insert_conflicts: 0,
                invalidations: 0,
                compile_failures: 0,
            }),
            wake: Condvar::new(),
        }
    }

    pub fn stats(&self) -> CacheStats {
        let state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        CacheStats {
            entries: state.entries.len(),
            capacity: state.capacity,
            hits: state.hits,
            misses: state.misses,
            compilations: state.compilations,
            evictions: state.evictions,
            insert_conflicts: state.insert_conflicts,
            invalidations: state.invalidations,
            compile_failures: state.compile_failures,
        }
    }

    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .entries
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Change the capacity and evict down to it.
    pub fn set_capacity(&self, capacity: usize) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.capacity = capacity.max(1);
        Self::evict(&mut state);
    }

    /// Remove every entry. In-flight compilations still complete and insert.
    pub fn clear(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.entries.clear();
        state.order.clear();
        state.aliases.clear();
        state.invalidations += 1;
    }

    pub fn invalidate(&self, key: &CacheKey) -> bool {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let resolved = state.aliases.remove(key).unwrap_or_else(|| key.clone());
        let removed = state.entries.remove(&resolved).is_some();
        state.order.retain(|candidate| candidate != &resolved);
        state.aliases.retain(|_, target| target != &resolved);
        if removed {
            state.invalidations += 1;
        }
        removed
    }

    pub fn contains(&self, key: &CacheKey) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .entries
            .contains_key(key)
    }

    /// Look up a compiled artifact without compiling.
    pub fn get(&self, key: &CacheKey) -> Option<Arc<CompiledModel>> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let resolved = state
            .aliases
            .get(key)
            .cloned()
            .unwrap_or_else(|| key.clone());
        let entry = state.entries.get(&resolved).cloned();
        if entry.is_some() {
            state.hits += 1;
            Self::touch(&mut state, &resolved);
        }
        entry
    }

    /// Alias a structural key to a canonical graph key already in the cache.
    pub fn insert_alias(&self, alias: CacheKey, target: CacheKey) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if state.entries.contains_key(&target) {
            state.aliases.insert(alias, target);
        }
    }

    /// Return a cached model or compile it exactly once for this key.
    pub fn get_or_insert_with<F>(&self, key: CacheKey, compile: F) -> Result<Arc<CompiledModel>>
    where
        F: FnOnce() -> Result<CompiledModel>,
    {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        loop {
            if let Some(entry) = state.entries.get(&key).cloned() {
                state.hits += 1;
                Self::touch(&mut state, &key);
                return Ok(entry);
            }
            if state.in_flight.contains(&key) {
                state = self
                    .wake
                    .wait(state)
                    .unwrap_or_else(|poison| poison.into_inner());
                continue;
            }
            state.in_flight.insert(key.clone());
            state.misses += 1;
            break;
        }
        drop(state);

        let result = compile();

        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.in_flight.remove(&key);
        self.wake.notify_all();
        match result {
            Ok(model) => {
                let entry = Arc::new(model);
                if let Some(existing) = state.entries.get(&key).cloned() {
                    // Another thread completed first despite the in-flight
                    // marker (e.g. a panic recovery path); keep one artifact.
                    state.insert_conflicts += 1;
                    return Ok(existing);
                }
                state.entries.insert(key.clone(), Arc::clone(&entry));
                state.order.push_back(key);
                state.compilations += 1;
                Self::evict(&mut state);
                Ok(entry)
            }
            Err(error) => {
                state.compile_failures += 1;
                Err(error)
            }
        }
    }

    fn touch(state: &mut State, key: &CacheKey) {
        state.order.retain(|candidate| candidate != key);
        state.order.push_back(key.clone());
    }

    fn evict(state: &mut State) {
        while state.entries.len() > state.capacity {
            let Some(oldest) = state.order.pop_front() else {
                break;
            };
            if state.entries.remove(&oldest).is_some() {
                state.evictions += 1;
            }
        }
        state
            .aliases
            .retain(|_, target| state.entries.contains_key(target));
    }
}

impl std::fmt::Debug for CompileCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("CompileCache")
            .field("entries", &stats.entries)
            .field("capacity", &stats.capacity)
            .finish()
    }
}

static GLOBAL_CACHE: LazyLock<CompileCache> = LazyLock::new(|| CompileCache::new(DEFAULT_CAPACITY));

/// Process-wide cache used by generated `predict` methods.
pub fn global() -> &'static CompileCache {
    &GLOBAL_CACHE
}

/// Compile through the process-wide cache.
pub fn compile_cached<M: crate::Module + ?Sized>(
    model: &M,
    inputs: &[InputSpec],
    options: &ModelCompileOptions,
) -> Result<Arc<CompiledModel>> {
    compile_cached_with(&GLOBAL_CACHE, model, inputs, options)
}

/// Compile through an explicit cache (useful for tests and per-device caches).
pub fn compile_cached_with<M: crate::Module + ?Sized>(
    cache: &CompileCache,
    model: &M,
    inputs: &[InputSpec],
    options: &ModelCompileOptions,
) -> Result<Arc<CompiledModel>> {
    let structural = CacheKey::Structural(Box::new(StructuralKey {
        cache_version: COMPILE_CACHE_VERSION,
        compiler_version: COMPILER_VERSION,
        model: std::any::type_name::<M>().to_string(),
        structure: structure_fingerprint(model)?,
        inputs: inputs.iter().map(InputSignature::from_spec).collect(),
        mode: options.mode,
        optimize: options.optimize,
        memory_plan: options.memory_plan,
        backend: CPU_BACKEND.to_string(),
        architecture: CPU_ARCH.to_string(),
        kernel_abi: KERNEL_ABI_VERSION,
        backward: false,
    }));
    if let Some(entry) = cache.get(&structural) {
        return Ok(entry);
    }
    let built = BuiltModel::build(model, inputs, None, options.mode)?;
    let graph_key = CacheKey::Graph(Box::new(CompileKey::new(&built, inputs, options, false)?));
    let entry = cache.get_or_insert_with(graph_key.clone(), || {
        CompiledModel::from_built(built, options, false)
    })?;
    cache.insert_alias(structural, graph_key);
    Ok(entry)
}

/// Stable description of module structure used by the fast pre-build lookup.
fn structure_fingerprint<M: crate::Module + ?Sized>(model: &M) -> Result<String> {
    let mut registry = crate::ParameterRegistry::new();
    model.visit_parameters(&crate::ModulePath::root(), &mut registry)?;
    let mut out = model.descriptor()?.render();
    for spec in registry.specs() {
        out.push_str(&format!(
            "|{}:{:?}:{}:{}",
            spec.name,
            spec.shape,
            spec.role.as_str(),
            spec.dtype
        ));
    }
    for (alias, canonical) in registry.aliases() {
        out.push_str(&format!("|{alias}={canonical}"));
    }
    Ok(out)
}
