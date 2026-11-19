//! Constant values in a MemoryPlan must never be assigned a slot smaller than
//! their logical byte length. The CUDA planned executor copies only the
//! logical prefix into a slot, so a too-small slot would be a hard error.

use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{DType, Shape};
use oxide_ml_graph::{ConstantTensor, GraphBuilder};

#[test]
fn planned_constant_capacity_covers_logical_bytes() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 4]), DType::F32);
    let bias = builder
        .constant_tensor(
            ConstantTensor::f32(Shape::from_static([4]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
        )
        .unwrap();
    let y = builder.add(x, bias).unwrap();
    let graph = builder.finish(&[y]).unwrap();
    let plan = MemoryPlanner::default().plan(&graph).unwrap();

    let mut checked = 0usize;
    for (value, assignment) in &plan.assignments {
        let info = graph.value(*value).unwrap();
        let logical = info.shape.numel_static().unwrap() * info.dtype.size_in_bytes();
        assert_eq!(
            assignment.size_bytes, logical,
            "assignment size mismatch for {value}"
        );
        let slot = plan.slot(assignment.slot).unwrap();
        assert!(
            slot.size_bytes >= assignment.offset + logical,
            "slot {} is too small for value {value}: slot={} offset={} logical={}",
            assignment.slot.0,
            slot.size_bytes,
            assignment.offset,
            logical
        );
        checked += 1;
    }
    assert!(checked > 0, "planner assigned no values");
}

#[test]
fn planned_scalar_constant_never_exceeds_one_element() {
    let mut builder = GraphBuilder::new();
    let seed = builder.constant_scalar(1.0);
    let graph = builder.finish(&[seed]).unwrap();
    let plan = MemoryPlanner::default().plan(&graph).unwrap();
    if let Some(assignment) = plan.assignment(seed) {
        assert_eq!(assignment.size_bytes, 4);
        let slot = plan.slot(assignment.slot).unwrap();
        assert!(slot.size_bytes >= 4);
    }
}

#[test]
fn scalar_constant_can_share_a_larger_slot() {
    // Regression for the 16-vs-1 planned constant copy: with the default
    // 64-byte alignment, a scalar constant (4 bytes) is assigned to a
    // 64-byte slot (16 f32 elements). The CUDA tensor's logical shape is
    // still scalar, so the copy must write only the logical prefix.
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 4]), DType::F32);
    let constant = builder.constant_scalar(1.5);
    let y = builder.add(x, constant).unwrap();
    let graph = builder.finish(&[y]).unwrap();
    let plan = MemoryPlanner::default().plan(&graph).unwrap();
    let assignment = plan
        .assignment(constant)
        .expect("scalar constant was planned");
    assert_eq!(assignment.size_bytes, 4);
    let slot = plan.slot(assignment.slot).unwrap();
    assert!(
        slot.size_bytes > assignment.size_bytes,
        "expected a slot larger than the scalar: slot={} assignment={}",
        slot.size_bytes,
        assignment.size_bytes
    );
    assert_eq!(slot.size_bytes % 4, 0);
    // 16 f32 elements is the concrete historical shape of the mismatch.
    assert_eq!(slot.size_bytes / 4, 16);
}
