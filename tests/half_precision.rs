//! Low-level F16/BF16 execution: the graph executor is generic over `Scalar`.

use oxide_ml::half::{bf16, f16};
use oxide_ml::prelude::*;

fn build(dtype: DType) -> (Graph, ValueId, ValueId, ValueId) {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), dtype);
    let identity = builder.input("identity", Shape::from_static([3, 2]), dtype);
    let bias = builder.input("bias", Shape::from_static([2, 2]), dtype);
    let product = builder.matmul(x, identity).unwrap();
    let sum = builder.add(product, bias).unwrap();
    let graph = builder.finish(&[sum]).unwrap();
    (graph, x, identity, bias)
}

fn run_f16() -> Vec<f32> {
    let (graph, x, identity, bias) = build(DType::F16);
    let values_x = [1.0f32, 2.0, 3.0, -1.0, 0.5, 2.0];
    let values_i = [1.0f32, 0.0, 0.0, 1.0, 1.0, -1.0];
    let values_b = [0.5f32, -0.5, 0.25, 0.125];

    let mut inputs = ExecutionInputs::<f16>::new();
    inputs.insert(
        x,
        DenseTensor::<f16>::from_vec([2, 3], values_x.iter().map(|v| f16::from_f32(*v)).collect())
            .unwrap(),
    );
    inputs.insert(
        identity,
        DenseTensor::<f16>::from_vec([3, 2], values_i.iter().map(|v| f16::from_f32(*v)).collect())
            .unwrap(),
    );
    inputs.insert(
        bias,
        DenseTensor::<f16>::from_vec([2, 2], values_b.iter().map(|v| f16::from_f32(*v)).collect())
            .unwrap(),
    );
    let result = Executor::<f16>::new(graph)
        .unwrap()
        .run(&ParameterStore::<f16>::new(), &inputs)
        .unwrap();
    result
        .output(0)
        .unwrap()
        .data()
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

#[test]
fn f16_graph_matches_f32_within_half_precision() {
    let half_values = run_f16();
    // Reference: the same integers/exact halves are representable, so the
    // matmul+add result is exact in f16 for this input.
    assert_eq!(half_values, vec![4.5, -1.5, 1.25, -1.375]);
}

#[test]
fn bf16_graph_executes() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::BF16);
    let y = builder.input("y", Shape::from_static([2, 2]), DType::BF16);
    let sum = builder.add(x, y).unwrap();
    let graph = builder.finish(&[sum]).unwrap();

    let mut inputs = ExecutionInputs::<bf16>::new();
    inputs.insert(
        x,
        DenseTensor::<bf16>::from_vec(
            [2, 2],
            vec![
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
            ],
        )
        .unwrap(),
    );
    inputs.insert(
        y,
        DenseTensor::<bf16>::from_vec(
            [2, 2],
            vec![
                bf16::from_f32(0.5),
                bf16::from_f32(0.25),
                bf16::from_f32(-1.0),
                bf16::from_f32(2.0),
            ],
        )
        .unwrap(),
    );
    let result = Executor::<bf16>::new(graph)
        .unwrap()
        .run(&ParameterStore::<bf16>::new(), &inputs)
        .unwrap();
    let values: Vec<f32> = result
        .output(0)
        .unwrap()
        .data()
        .iter()
        .map(|v| v.to_f32())
        .collect();
    assert_eq!(values, vec![1.5, 2.25, 2.0, 6.0]);
}
