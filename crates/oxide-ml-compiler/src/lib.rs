//! Phase 5 Graph Compiler: IR-preserving optimization passes and pass manager.
//!
//! Passes mutate the semantic Graph in place and keep ValueIds stable. Dense
//! NodeIds are restored after removals. Every pass boundary is verified before
//! compilation returns.

mod arena_key;
pub mod arena_registry;
mod compiler;
mod constant_folding;
mod cse;
mod dce;
mod fusion;
mod manager;
mod memory_planner;
mod pass;
pub mod plan_hash;
mod rewrite;

pub use arena_key::{ArenaContextId, ArenaDescriptor, ArenaKey, DeviceId};
pub use arena_registry::{
    ArenaRegistry, ArenaRegistryError, ArenaRegistryStats, DEFAULT_MAX_ARENAS,
};
pub use compiler::{CompileOptions, CompiledGraph, Compiler, PassToggles};
pub use constant_folding::ConstantFolding;
pub use cse::CommonSubexpressionElimination;
pub use dce::DeadCodeElimination;
pub use fusion::OperatorFusion;
pub use manager::{CompileReport, PassManager, PassReport};
pub use memory_planner::MemoryPlanner;
pub use plan_hash::{
    CANONICAL_PLAN_VERSION, CanonicalAssignment, CanonicalLifetime, CanonicalLifetimePoint,
    CanonicalPlan, CanonicalSlot, MEMORY_PLAN_VERSION, PlanHash, PlanHashError,
    canonical_memory_plan_hash, canonicalize_memory_plan, memory_plan_hash,
    validate_memory_plan_graph_independent, validate_memory_plan_with_graph,
};
pub use pass::{GraphPass, PassContext, PassResult};
