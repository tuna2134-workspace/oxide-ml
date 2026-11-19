//! Fusion correctness: patterns are applied only when safe, and fused graphs are
//! numerically identical to their unfused versions.

use oxide_ml_compiler::{GraphPass, OperatorFusion, PassContext};
use oxide_ml_core::{DType, DenseTensor, ParameterStore, Shape, ValueId};
use oxide_ml_cpu::{ExecutionInputs, Executor};
use oxide_ml_graph::{Graph, GraphBuilder, Op};

fn run(graph: &Graph, inputs: &ExecutionInputs<f32>, output: ValueId) -> Vec<f32> {
    Executor::<f32>::new(graph.clone())
        .unwrap()
        .run(&ParameterStore::new(), inputs)
        .unwrap()
        .value(output)
        .unwrap()
        .data()
        .to_vec()
}

fn fuse(graph: &mut Graph) -> usize {
    let backend = oxide_ml_cpu::CpuBackend::<f32>::new();
    let context = PassContext::<f32> { backend: &backend };
    let report = OperatorFusion.run(graph, &context).unwrap();
    assert_eq!(report.changed, !report.notes.is_empty());
    graph.validate().unwrap();
    graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.op, Op::AddRelu | Op::MulAdd | Op::MatMulBias))
        .count()
}

#[test]
fn fuses_mul_add_and_preserves_values() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = b.input("y", Shape::from_static([2, 3]), DType::F32);
    let c = b.input("c", Shape::from_static([3]), DType::F32);
    let product = b.mul(x, y).unwrap();
    let sum = b.add(product, c).unwrap();
    let mut graph = b.finish(&[sum]).unwrap();

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        x,
        DenseTensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    );
    inputs.insert(
        y,
        DenseTensor::from_vec([2, 3], vec![2., 2., 2., 3., 3., 3.]).unwrap(),
    );
    inputs.insert(c, DenseTensor::from_vec([3], vec![1., 10., 100.]).unwrap());
    let before = run(&graph, &inputs, sum);

    assert_eq!(fuse(&mut graph), 1);
    assert!(
        graph
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::MulAdd))
    );
    assert!(!graph.nodes().iter().any(|node| matches!(node.op, Op::Mul)));
    let after = run(&graph, &inputs, sum);
    assert_eq!(before, after);
}

#[test]
fn fuses_matmul_bias_and_preserves_values() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let w = b.input("w", Shape::from_static([4, 3]), DType::F32);
    let bias = b.input("bias", Shape::from_static([4]), DType::F32);
    let product = b.matmul_with(x, w, false, true).unwrap();
    let sum = b.add(product, bias).unwrap();
    let mut graph = b.finish(&[sum]).unwrap();

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        x,
        DenseTensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    );
    inputs.insert(
        w,
        DenseTensor::from_vec([4, 3], vec![1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.])
            .unwrap(),
    );
    inputs.insert(
        bias,
        DenseTensor::from_vec([4], vec![0.5, 1.5, 2.5, 3.5]).unwrap(),
    );
    let before = run(&graph, &inputs, sum);

    assert_eq!(fuse(&mut graph), 1);
    assert!(
        graph
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::MatMulBias))
    );
    assert!(
        !graph
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::MatMul))
    );
    let after = run(&graph, &inputs, sum);
    assert_eq!(before, after);
}

#[test]
fn does_not_fuse_multi_use_producer() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 2]), DType::F32);
    let y = b.input("y", Shape::from_static([2, 2]), DType::F32);
    let product = b.mul(x, y).unwrap();
    let sum = b.add(product, x).unwrap();
    let activated = b.relu(product).unwrap();
    let mut graph = b.finish(&[sum, activated]).unwrap();

    assert_eq!(fuse(&mut graph), 0);
    assert!(graph.nodes().iter().any(|node| matches!(node.op, Op::Mul)));
    assert!(graph.nodes().iter().any(|node| matches!(node.op, Op::Add)));
}

#[test]
fn does_not_fuse_graph_output_producer() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 2]), DType::F32);
    let y = b.input("y", Shape::from_static([2, 2]), DType::F32);
    let product = b.mul(x, y).unwrap();
    let sum = b.add(product, x).unwrap();
    let mut graph = b.finish(&[product, sum]).unwrap();

    assert_eq!(fuse(&mut graph), 0);
}

#[test]
fn fuses_add_relu_after_other_patterns() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 2]), DType::F32);
    let y = b.input("y", Shape::from_static([2, 2]), DType::F32);
    let sum = b.add(x, y).unwrap();
    let activated = b.relu(sum).unwrap();
    let mut graph = b.finish(&[activated]).unwrap();

    assert_eq!(fuse(&mut graph), 1);
    assert!(
        graph
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::AddRelu))
    );
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        x,
        DenseTensor::from_vec([2, 2], vec![1., -2., 3., -4.]).unwrap(),
    );
    inputs.insert(
        y,
        DenseTensor::from_vec([2, 2], vec![-2., -2., 2., 2.]).unwrap(),
    );
    let output = run(&graph, &inputs, activated);
    assert_eq!(output, vec![0., 0., 5., 0.]);
}
