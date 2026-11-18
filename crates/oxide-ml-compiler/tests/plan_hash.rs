use oxide_ml_compiler::{
    MemoryPlanner, canonical_memory_plan_hash, canonicalize_memory_plan, memory_plan_hash,
};
use oxide_ml_core::{BufferId, DType, LifetimePoint, MemoryPlan, Shape, ValueId};
use oxide_ml_graph::{Graph, GraphBuilder};

fn branching() -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4]), DType::F32);
    let a = builder.relu(x).unwrap();
    let b = builder.sigmoid(x).unwrap();
    let out = builder.add(a, b).unwrap();
    builder.finish(&[out]).unwrap()
}

fn ordered(reverse: bool) -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4]), DType::F32);
    let y = builder.input("y", Shape::from_static([4]), DType::F32);
    let (a, b) = if reverse {
        let b = builder.tanh(y).unwrap();
        let a = builder.relu(x).unwrap();
        (a, b)
    } else {
        let a = builder.relu(x).unwrap();
        let b = builder.tanh(y).unwrap();
        (a, b)
    };
    let out = builder.add(a, b).unwrap();
    builder.finish(&[out]).unwrap()
}

fn plan_for(graph: &Graph) -> MemoryPlan {
    MemoryPlanner::default().plan(graph).unwrap()
}

fn canonical_hash(graph: &Graph, plan: &MemoryPlan) -> [u8; 32] {
    canonical_memory_plan_hash(graph, plan).unwrap().0
}

#[test]
fn same_plan_same_hash() {
    let graph = branching();
    let plan = plan_for(&graph);
    assert_eq!(canonical_hash(&graph, &plan), canonical_hash(&graph, &plan));
    assert_eq!(
        memory_plan_hash(&plan).unwrap(),
        memory_plan_hash(&plan).unwrap()
    );
}

#[test]
fn canonical_plan_is_independent_of_graph_numbering() {
    let first = ordered(false);
    let second = ordered(true);
    // The greedy planner ties on definition order, so independently generated
    // plans may legitimately differ. Use an explicitly identical plan
    // structure to test that ValueId/NodeId renumbering alone is normalized.
    let first_plan = flat_plan(&first);
    let second_plan = flat_plan(&second);
    assert_eq!(
        canonical_hash(&first, &first_plan),
        canonical_hash(&second, &second_plan)
    );
    let canonical = canonicalize_memory_plan(&first, &first_plan).unwrap();
    assert_eq!(canonical.version, oxide_ml_compiler::MEMORY_PLAN_VERSION);
}

fn flat_plan(graph: &Graph) -> MemoryPlan {
    let slots = vec![oxide_ml_core::BufferSlot {
        id: BufferId(0),
        size_bytes: 64,
        alignment: 64,
    }];
    let mut assignments = std::collections::HashMap::new();
    let mut lifetimes = Vec::new();
    let mut total = 0usize;
    for node in graph.nodes() {
        for &output in &node.outputs {
            let value = graph.value(output).unwrap();
            let bytes = value.shape.numel_static().unwrap() * value.dtype.size_in_bytes();
            assignments.insert(
                output,
                oxide_ml_core::BufferAssignment {
                    slot: BufferId(0),
                    offset: 0,
                    size_bytes: bytes,
                    dtype: value.dtype,
                    shape: value.shape.clone(),
                },
            );
            lifetimes.push(oxide_ml_core::ValueLifetime {
                value: output,
                defined_at: LifetimePoint::Node(node.id.0),
                last_used_at: LifetimePoint::End,
                size_bytes: Some(bytes),
                dtype: value.dtype,
                shape: value.shape.clone(),
            });
            total += bytes;
        }
    }
    assert!(total <= slots[0].size_bytes);
    MemoryPlan {
        slots,
        assignments,
        lifetimes,
        unplanned_values: Vec::new(),
        total_value_bytes: total,
    }
}

#[test]
fn different_slot_assignment_is_different_hash() {
    let graph = branching();
    let plan = plan_for(&graph);
    assert!(plan.assignments.len() >= 2);
    let mut changed = plan.clone();
    let mut values = changed.assignments.keys().copied().collect::<Vec<_>>();
    values.sort();
    let left = values[0];
    let right = values[1];
    let left_slot = changed.assignments[&left].slot;
    let right_slot = changed.assignments[&right].slot;
    if left_slot == right_slot {
        return;
    }
    let left_size = changed.assignments[&left].size_bytes;
    let right_size = changed.assignments[&right].size_bytes;
    if changed.slot(right_slot).unwrap().size_bytes < left_size
        || changed.slot(left_slot).unwrap().size_bytes < right_size
    {
        return;
    }
    changed.assignments.get_mut(&left).unwrap().slot = right_slot;
    changed.assignments.get_mut(&right).unwrap().slot = left_slot;
    assert_ne!(
        canonical_hash(&graph, &plan),
        canonical_hash(&graph, &changed)
    );
}

#[test]
fn different_slot_capacity_is_different_hash() {
    let graph = branching();
    let plan = plan_for(&graph);
    let mut changed = plan.clone();
    changed.slots[0].size_bytes += changed.slots[0].alignment;
    assert_ne!(
        canonical_hash(&graph, &plan),
        canonical_hash(&graph, &changed)
    );
}

#[test]
fn different_alignment_is_different_hash() {
    let graph = branching();
    let plan = MemoryPlanner::new(32).unwrap().plan(&graph).unwrap();
    let other = MemoryPlanner::new(64).unwrap().plan(&graph).unwrap();
    assert_ne!(
        canonical_hash(&graph, &plan),
        canonical_hash(&graph, &other)
    );
}

#[test]
fn different_lifetime_is_different_hash() {
    let graph = branching();
    let plan = plan_for(&graph);
    let mut changed = plan.clone();
    let index = changed
        .lifetimes
        .iter()
        .position(|lifetime| lifetime.last_used_at != LifetimePoint::End)
        .unwrap_or(0);
    changed.lifetimes[index].last_used_at = LifetimePoint::End;
    assert_ne!(
        canonical_hash(&graph, &plan),
        canonical_hash(&graph, &changed)
    );
}

#[test]
fn different_shape_is_different_hash() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4]), DType::F32);
    let out = builder.relu(x).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let plan = plan_for(&graph);

    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([8]), DType::F32);
    let out = builder.relu(x).unwrap();
    let other_graph = builder.finish(&[out]).unwrap();
    let other_plan = plan_for(&other_graph);

    assert_ne!(
        canonical_hash(&graph, &plan),
        canonical_hash(&other_graph, &other_plan)
    );
}

#[test]
fn different_dtype_is_different_hash() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4]), DType::F32);
    let out = builder.relu(x).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let plan = plan_for(&graph);

    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4]), DType::F64);
    let out = builder.relu(x).unwrap();
    let other_graph = builder.finish(&[out]).unwrap();
    let other_plan = plan_for(&other_graph);

    assert_ne!(
        canonical_hash(&graph, &plan),
        canonical_hash(&other_graph, &other_plan)
    );
}

#[test]
fn canonical_plan_version_participates_in_hash() {
    let graph = branching();
    let plan = plan_for(&graph);
    let mut canonical = canonicalize_memory_plan(&graph, &plan).unwrap();
    let first = canonical.hash();
    canonical.version += 1;
    assert_ne!(first, canonical.hash());
}

#[test]
fn invalid_plan_is_rejected() {
    let graph = branching();
    let plan = plan_for(&graph);
    assert!(canonical_memory_plan_hash(&graph, &plan).is_ok());

    let mut duplicate_slot = plan.clone();
    duplicate_slot.slots[1].id = duplicate_slot.slots[0].id;
    assert!(canonical_memory_plan_hash(&graph, &duplicate_slot).is_err());

    let mut bad_alignment = plan.clone();
    bad_alignment.slots[0].alignment = 3;
    assert!(canonical_memory_plan_hash(&graph, &bad_alignment).is_err());

    let mut bad_lifetime = plan.clone();
    bad_lifetime.lifetimes[0].defined_at = LifetimePoint::End;
    bad_lifetime.lifetimes[0].last_used_at = LifetimePoint::Node(0);
    assert!(canonical_memory_plan_hash(&graph, &bad_lifetime).is_err());

    let mut too_small = plan.clone();
    let value = *too_small.assignments.keys().next().unwrap();
    let slot = too_small.assignments[&value].slot;
    too_small.assignments.get_mut(&value).unwrap().offset =
        too_small.slot(slot).unwrap().size_bytes;
    assert!(canonical_memory_plan_hash(&graph, &too_small).is_err());

    let mut unknown_value = plan.clone();
    let template = unknown_value.assignments.values().next().unwrap().clone();
    unknown_value.assignments.insert(
        ValueId(999),
        oxide_ml_core::BufferAssignment {
            slot: template.slot,
            offset: template.offset,
            size_bytes: template.size_bytes,
            dtype: template.dtype,
            shape: template.shape,
        },
    );
    unknown_value.total_value_bytes += template.size_bytes;
    assert!(canonical_memory_plan_hash(&graph, &unknown_value).is_err());

    let mut dtype_mismatch = plan.clone();
    let value = *dtype_mismatch.assignments.keys().next().unwrap();
    let slot = dtype_mismatch.assignments[&value].slot;
    let doubled = dtype_mismatch.assignments[&value].size_bytes * 2;
    let slot_capacity = dtype_mismatch.slot(slot).unwrap().size_bytes;
    dtype_mismatch.total_value_bytes += dtype_mismatch.assignments[&value].size_bytes;
    let assignment = dtype_mismatch.assignments.get_mut(&value).unwrap();
    assignment.dtype = DType::F64;
    assignment.size_bytes = doubled;
    if slot_capacity < doubled {
        return;
    }
    assert!(canonical_memory_plan_hash(&graph, &dtype_mismatch).is_err());
}

#[test]
fn raw_plan_hash_rejects_graph_independent_invalid_plan() {
    let graph = branching();
    let plan = plan_for(&graph);
    let mut duplicate_slot = plan.clone();
    duplicate_slot.slots[1].id = BufferId(0);
    assert!(memory_plan_hash(&duplicate_slot).is_err());
    assert!(memory_plan_hash(&plan).is_ok());
}

#[test]
fn canonical_plan_uses_canonical_value_refs() {
    let graph = branching();
    let plan = plan_for(&graph);
    let canonical = canonicalize_memory_plan(&graph, &plan).unwrap();
    assert_eq!(canonical.assignments.len(), plan.assignments.len());
    assert!(!canonical.assignments.is_empty());
    assert!(canonical.lifetimes.iter().all(|lifetime| {
        matches!(
            lifetime.defined_at,
            oxide_ml_compiler::CanonicalLifetimePoint::Node(_)
        )
    }));
}

#[test]
fn plan_with_node_lifetime_uses_canonical_node_index() {
    let graph = branching();
    let plan = plan_for(&graph);
    let canonical = canonicalize_memory_plan(&graph, &plan).unwrap();
    let node_count = graph.nodes().len() as u32;
    for lifetime in &canonical.lifetimes {
        match lifetime.defined_at {
            oxide_ml_compiler::CanonicalLifetimePoint::Node(node) => {
                assert!(node < node_count);
            }
            oxide_ml_compiler::CanonicalLifetimePoint::End => {}
        }
    }
}

#[test]
fn dynamic_shape_values_are_unplanned_and_hashed() {
    let mut builder = GraphBuilder::new();
    let x = builder.input(
        "x",
        Shape::new(vec![
            oxide_ml_core::Dim::Dynamic,
            oxide_ml_core::Dim::Static(4),
        ]),
        DType::F32,
    );
    let out = builder.relu(x).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let plan = plan_for(&graph);
    assert!(plan.unplanned_values.contains(&out));
    assert!(!plan.assignments.contains_key(&out));
    let canonical = canonicalize_memory_plan(&graph, &plan).unwrap();
    assert!(!canonical.unplanned.is_empty());
    assert!(
        canonical
            .lifetimes
            .iter()
            .any(|lifetime| lifetime.size_bytes.is_none())
    );
    assert!(memory_plan_hash(&plan).is_ok());
}

#[test]
fn out_of_range_slot_is_rejected() {
    let graph = branching();
    let plan = plan_for(&graph);
    let mut invalid = plan.clone();
    let value = *invalid.assignments.keys().next().unwrap();
    invalid.assignments.get_mut(&value).unwrap().slot = BufferId(99);
    assert!(canonical_memory_plan_hash(&graph, &invalid).is_err());
    assert!(memory_plan_hash(&invalid).is_err());
}
