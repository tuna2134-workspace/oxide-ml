//! Fused MoE op: a hand-computed two-expert, top-1 case.

use oxide_ml_core::{DType, DenseTensor, ParameterStore, Shape};
use oxide_ml_cpu::{ExecutionInputs, Executor};
use oxide_ml_graph::GraphBuilder;

fn tensor(dims: &[usize], values: Vec<f32>) -> DenseTensor<f32> {
    DenseTensor::from_vec(dims, values).unwrap()
}

#[test]
fn moe_top1_matches_hand_computation() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([1, 2]), DType::F32);
    let scale = builder.input("scale", Shape::from_static([2]), DType::F32);
    let per_expert = builder.input("per", Shape::from_static([2]), DType::F32);
    let router = builder.input("router", Shape::from_static([2, 2]), DType::F32);
    let gate_up = builder.input("gate_up", Shape::from_static([2, 4, 2]), DType::F32);
    let down = builder.input("down", Shape::from_static([2, 2, 2]), DType::F32);
    let norm = builder.input("norm", Shape::from_static([2]), DType::F32);
    let out = builder
        .moe(
            &[x, scale, per_expert, router, gate_up, down, norm],
            2,
            1,
            2,
            0.0,
        )
        .unwrap();
    let graph = builder.finish(&[out]).unwrap();

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(x, tensor(&[1, 2], vec![1.0, 1.0]));
    inputs.insert(scale, tensor(&[2], vec![1.0, 1.0]));
    inputs.insert(per_expert, tensor(&[2], vec![1.0, 1.0]));
    inputs.insert(router, tensor(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]));
    inputs.insert(
        gate_up,
        tensor(
            &[2, 4, 2],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
    );
    inputs.insert(
        down,
        tensor(&[2, 2, 2], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    );
    inputs.insert(norm, tensor(&[2], vec![1.0, 1.0]));

    let result = Executor::<f32>::new(graph)
        .unwrap()
        .run(&ParameterStore::new(), &inputs)
        .unwrap();
    let output = result.output(0).unwrap();
    // softmax([H^-0.5, H^-0.5]) is uniform; the tie picks expert 0, which
    // yields gelu(1) * 1 through the identity down projection.
    let expected = 0.5f32 * (1.0f32 + (0.797_884_6f32 * (1.0f32 + 0.044_715f32)).tanh());
    assert!(
        (output.data()[0] - expected).abs() < 1e-5,
        "{}",
        output.data()[0]
    );
    assert!(output.data()[1].abs() < 1e-6);
}
