use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use oxide_ml_compiler::{
    ArenaContextId, ArenaDescriptor, ArenaKey, ArenaRegistry, ArenaRegistryError,
    MEMORY_PLAN_VERSION, MemoryPlanner, canonicalize_memory_plan,
};
use oxide_ml_core::{DType, Shape};
use oxide_ml_graph::GraphBuilder;

#[derive(Debug)]
struct MockArena {
    id: usize,
}

fn descriptor(context: u64) -> ArenaDescriptor {
    descriptor_with(4, 64, context)
}

fn descriptor_with(elements: usize, alignment: usize, context: u64) -> ArenaDescriptor {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([elements]), DType::F32);
    let out = builder.relu(x).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let plan = MemoryPlanner::new(alignment).unwrap().plan(&graph).unwrap();
    let canonical_graph = graph.canonicalize().unwrap();
    let canonical_plan = canonicalize_memory_plan(&graph, &plan).unwrap();
    ArenaDescriptor::new(
        canonical_graph,
        canonical_plan,
        oxide_ml_compiler::DeviceId::new(0, None, "mock", (7, 5), 1024),
        ArenaContextId(context),
        DType::F32,
    )
}

fn key(descriptor: &ArenaDescriptor) -> ArenaKey {
    ArenaKey::new(
        descriptor.graph.hash(),
        descriptor.plan.hash(),
        descriptor.device_id.clone(),
        descriptor.context_id,
        descriptor.dtype,
        MEMORY_PLAN_VERSION,
    )
}

fn create(
    registry: &ArenaRegistry<MockArena>,
    key: ArenaKey,
    descriptor: ArenaDescriptor,
    id: usize,
) -> usize {
    registry
        .get_or_create(key, descriptor, 1, || {
            Ok::<_, &'static str>(MockArena { id })
        })
        .unwrap()
        .id
}

#[test]
fn same_key_and_descriptor_reuses_arena() {
    let registry = ArenaRegistry::<MockArena>::default();
    let descriptor = descriptor(1);
    let key = key(&descriptor);
    let calls = AtomicUsize::new(0);
    let first = registry
        .get_or_create(key.clone(), descriptor.clone(), 1, || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, &'static str>(MockArena { id: 1 })
        })
        .unwrap();
    let second = registry
        .get_or_create(key, descriptor, 1, || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, &'static str>(MockArena { id: 2 })
        })
        .unwrap();
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(registry.len(), 1);
    assert_eq!(registry.stats().hit_count, 1);
}

#[test]
fn same_hash_different_descriptor_does_not_share() {
    let registry = ArenaRegistry::<MockArena>::default();
    let first_descriptor = descriptor(1);
    let key = key(&first_descriptor);
    let first = registry
        .get_or_create(key.clone(), first_descriptor, 1, || {
            Ok::<_, &'static str>(MockArena { id: 1 })
        })
        .unwrap();
    let second = registry
        .get_or_create(key, descriptor(2), 1, || {
            Ok::<_, &'static str>(MockArena { id: 2 })
        })
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(registry.len(), 2);
    assert_eq!(registry.stats().collision_count, 1);
}

#[test]
fn different_key_is_separate() {
    let registry = ArenaRegistry::<MockArena>::default();
    let first = descriptor(1);
    let second = descriptor(2);
    assert_eq!(create(&registry, key(&first), first, 1), 1);
    assert_eq!(create(&registry, key(&second), second, 2), 2);
    assert_eq!(registry.len(), 2);
}

#[test]
fn different_device_dtype_and_version_are_different_keys() {
    let base = descriptor(1);
    let base_key = key(&base);

    let mut other_device = base.clone();
    other_device.device_id.ordinal = 1;
    assert_ne!(base_key, key(&other_device));

    let mut other_dtype = base.clone();
    other_dtype.dtype = DType::F64;
    assert_ne!(base_key, key(&other_dtype));

    let version_key = ArenaKey::new(
        base.graph.hash(),
        base.plan.hash(),
        base.device_id.clone(),
        base.context_id,
        base.dtype,
        MEMORY_PLAN_VERSION + 1,
    );
    assert_ne!(base_key, version_key);
}

#[test]
fn slot_set_capacity_mismatch_creates_second_arena() {
    let registry = ArenaRegistry::<MockArena>::default();
    let descriptor = descriptor(1);
    let key = key(&descriptor);
    let small = registry
        .get_or_create(key.clone(), descriptor.clone(), 1, || {
            Ok::<_, &'static str>(MockArena { id: 1 })
        })
        .unwrap();
    let large = registry
        .get_or_create(key, descriptor, 4, || {
            Ok::<_, &'static str>(MockArena { id: 2 })
        })
        .unwrap();
    assert!(!Arc::ptr_eq(&small, &large));
    assert_eq!(registry.len(), 2);
}

#[test]
fn capacity_limit_does_not_evict_existing_arena() {
    let registry = ArenaRegistry::<MockArena>::new(1);
    let first = descriptor(1);
    let first_key = key(&first);
    let first_arena = registry
        .get_or_create(first_key.clone(), first.clone(), 1, || {
            Ok::<_, &'static str>(MockArena { id: 1 })
        })
        .unwrap();
    let second = descriptor(2);
    let result = registry.get_or_create(key(&second), second, 1, || {
        Ok::<_, &'static str>(MockArena { id: 2 })
    });
    assert!(matches!(
        result,
        Err(ArenaRegistryError::CapacityExceeded { .. })
    ));
    let reused = registry
        .get_or_create(first_key, first, 1, || {
            Ok::<_, &'static str>(MockArena { id: 3 })
        })
        .unwrap();
    assert!(Arc::ptr_eq(&first_arena, &reused));
}

#[test]
fn same_hash_different_graph_descriptor_does_not_share() {
    let registry = ArenaRegistry::<MockArena>::default();
    let first_descriptor = descriptor(1);
    let key = key(&first_descriptor);
    let first = registry
        .get_or_create(key.clone(), first_descriptor, 1, || {
            Ok::<_, &'static str>(MockArena { id: 1 })
        })
        .unwrap();
    let second = registry
        .get_or_create(key, descriptor_with(8, 64, 1), 1, || {
            Ok::<_, &'static str>(MockArena { id: 2 })
        })
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(registry.len(), 2);
}

#[test]
fn same_hash_different_plan_descriptor_does_not_share() {
    let registry = ArenaRegistry::<MockArena>::default();
    let first_descriptor = descriptor(1);
    let key = key(&first_descriptor);
    let first = registry
        .get_or_create(key.clone(), first_descriptor, 1, || {
            Ok::<_, &'static str>(MockArena { id: 1 })
        })
        .unwrap();
    let second = registry
        .get_or_create(key, descriptor_with(4, 32, 1), 1, || {
            Ok::<_, &'static str>(MockArena { id: 2 })
        })
        .unwrap();
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(registry.len(), 2);
}

#[test]
fn registry_drop_does_not_panic() {
    let registry = ArenaRegistry::<MockArena>::default();
    let descriptor = descriptor(1);
    let _ = create(&registry, key(&descriptor), descriptor, 1);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(registry)));
    assert!(result.is_ok());
}
