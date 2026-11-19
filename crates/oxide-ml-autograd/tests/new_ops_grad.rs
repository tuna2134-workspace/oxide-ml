//! Numerical gradient checks for the additive operations' backward rules.
//!
//! Each test builds a forward graph, materializes the backward graph with the
//! built-in registry, executes it on the CPU backend, and compares the analytic
//! gradient against central finite differences (eps `1e-2`, tolerance `2e-2`).

use oxide_ml_autograd::{TrainingGraph, build_training_graph};
use oxide_ml_core::{DType, DenseTensor, ParameterId, ParameterStore, Shape, ValueId};
use oxide_ml_cpu::{ExecutionInputs, Executor};
use oxide_ml_graph::{Graph, GraphBuilder};

const EPS: f32 = 1e-2;
const TOLERANCE: f32 = 2e-2;

fn run_value(
    graph: &Graph,
    params: &ParameterStore<f32>,
    inputs: &ExecutionInputs<f32>,
    value: ValueId,
) -> f32 {
    Executor::<f32>::new(graph.clone())
        .unwrap()
        .run(params, inputs)
        .unwrap()
        .value(value)
        .unwrap()
        .data()[0]
}

fn assert_close(analytic: &[f32], numeric: &[f32], label: &str) {
    assert_eq!(analytic.len(), numeric.len(), "{label}: length mismatch");
    for (index, (&analytic, &numeric)) in analytic.iter().zip(numeric).enumerate() {
        let scale = analytic.abs().max(numeric.abs()).max(1.0);
        let error = (analytic - numeric).abs() / scale;
        assert!(
            error < TOLERANCE,
            "{label}[{index}] analytic={analytic} numeric={numeric} rel_error={error}"
        );
    }
}

/// Compare the analytic gradient of a graph input with central differences of
/// the scalar loss.
fn check_value_gradient(
    forward: &Graph,
    training: &TrainingGraph,
    inputs: &ExecutionInputs<f32>,
    value: ValueId,
) {
    let params = ParameterStore::<f32>::new();
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(&params, inputs)
        .unwrap();
    let gradient = training
        .gradient_for_value(value)
        .expect("input value participates in the backward graph");
    let analytic = result.value(gradient).unwrap().data().to_vec();

    let dims = inputs.get(value).unwrap().dims();
    let data = inputs.get(value).unwrap().data().to_vec();
    let mut numeric = Vec::with_capacity(data.len());
    for index in 0..data.len() {
        let mut plus = inputs.clone();
        let mut minus = inputs.clone();
        let mut plus_data = data.clone();
        let mut minus_data = data.clone();
        plus_data[index] += EPS;
        minus_data[index] -= EPS;
        plus.insert(
            value,
            DenseTensor::from_vec(&dims, plus_data).expect("perturbed input"),
        );
        minus.insert(
            value,
            DenseTensor::from_vec(&dims, minus_data).expect("perturbed input"),
        );
        let forward_plus = run_value(forward, &params, &plus, training.loss);
        let forward_minus = run_value(forward, &params, &minus, training.loss);
        numeric.push((forward_plus - forward_minus) / (2.0 * EPS));
    }
    assert_close(&analytic, &numeric, "input gradient");
}

/// Compare the analytic gradient of a parameter with central differences.
fn check_parameter_gradient(
    forward: &Graph,
    training: &TrainingGraph,
    params: &ParameterStore<f32>,
    inputs: &ExecutionInputs<f32>,
    id: ParameterId,
) {
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(params, inputs)
        .unwrap();
    let gradient = training
        .gradient_for_parameter(id)
        .expect("parameter receives a gradient");
    let analytic = result.value(gradient).unwrap().data().to_vec();

    let data = params.get(id).unwrap().data.data().to_vec();
    let mut numeric = Vec::with_capacity(data.len());
    for index in 0..data.len() {
        let mut plus = params.clone();
        let mut minus = params.clone();
        let plus_data = plus.get_mut(id).unwrap();
        plus_data.data.data_mut()[index] += EPS;
        let minus_data = minus.get_mut(id).unwrap();
        minus_data.data.data_mut()[index] -= EPS;
        let forward_plus = run_value(forward, &plus, inputs, training.loss);
        let forward_minus = run_value(forward, &minus, inputs, training.loss);
        numeric.push((forward_plus - forward_minus) / (2.0 * EPS));
    }
    assert_close(&analytic, &numeric, "parameter gradient");
}

fn unary_case(op: UnaryOp, values: Vec<f32>) {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = match op {
        UnaryOp::Sqrt => builder.sqrt(x),
        UnaryOp::Exp => builder.exp(x),
        UnaryOp::Log => builder.log(x),
        UnaryOp::Abs => builder.abs(x),
    }
    .unwrap();
    let loss = builder.reduce_sum(y, vec![0, 1], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(x, DenseTensor::from_vec([2, 3], values).unwrap());
    check_value_gradient(&forward, &training, &inputs, x);
}

#[derive(Clone, Copy)]
enum UnaryOp {
    Sqrt,
    Exp,
    Log,
    Abs,
}

#[test]
fn sqrt_gradient_matches_finite_differences() {
    unary_case(UnaryOp::Sqrt, vec![0.25, 0.5, 1.0, 2.0, 4.0, 9.0]);
}

#[test]
fn exp_gradient_matches_finite_differences() {
    unary_case(UnaryOp::Exp, vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]);
}

#[test]
fn log_gradient_matches_finite_differences() {
    unary_case(UnaryOp::Log, vec![0.25, 0.5, 1.0, 2.0, 4.0, 9.0]);
}

#[test]
fn abs_gradient_matches_finite_differences() {
    unary_case(UnaryOp::Abs, vec![-1.0, -0.5, 0.25, 0.5, 1.0, 2.0]);
}

#[test]
fn log_softmax_gradient_matches_finite_differences() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = builder.log_softmax(x, -1).unwrap();
    let loss = builder.reduce_sum(y, vec![0, 1], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        x,
        DenseTensor::from_vec([2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.5, 0.0]).unwrap(),
    );
    check_value_gradient(&forward, &training, &inputs, x);
}

#[test]
fn slice_gradient_is_zero_outside_the_slice() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 4]), DType::F32);
    let sliced = builder.slice(x, 1, 1, 2).unwrap();
    let loss = builder.reduce_sum(sliced, vec![0, 1], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        x,
        DenseTensor::from_vec([2, 4], vec![0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]).unwrap(),
    );
    check_value_gradient(&forward, &training, &inputs, x);

    let params = ParameterStore::<f32>::new();
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    let gradient = training.gradient_for_value(x).unwrap();
    assert_eq!(
        result.value(gradient).unwrap().data(),
        &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    );
}

#[test]
fn embedding_gradient_accumulates_repeated_indices() {
    let mut builder = GraphBuilder::new();
    let weight = builder
        .try_parameter("weight", Shape::from_static([4, 3]), DType::F32, true)
        .unwrap();
    let indices = builder.input("indices", Shape::from_static([4]), DType::F32);
    let embedded = builder.embedding(weight, indices).unwrap();
    let loss = builder.reduce_sum(embedded, vec![0, 1], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();

    let mut params = ParameterStore::<f32>::new();
    params
        .add(
            ParameterId(0),
            "weight",
            DenseTensor::from_vec(
                [4, 3],
                vec![
                    0.1, 0.2, 0.3, -0.4, 0.5, 0.6, 0.7, -0.8, 0.9, 1.0, 0.2, -0.3,
                ],
            )
            .unwrap(),
            true,
        )
        .unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        indices,
        DenseTensor::from_vec([4], vec![0.0, 0.0, 2.0, 3.0]).unwrap(),
    );

    check_parameter_gradient(&forward, &training, &params, &inputs, ParameterId(0));

    // Rows 0 appears twice, rows 2 and 3 once, row 1 never: the gradient of the
    // sum of all embeddings is exactly the index multiplicity.
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    let gradient = training.gradient_for_parameter(ParameterId(0)).unwrap();
    assert_eq!(
        result.value(gradient).unwrap().data(),
        &[2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn embedding_backward_is_used_by_the_registered_rule() {
    // The built-in registry must expose rules for the new forward ops only; the
    // fused backward primitives stay unregistered.
    use oxide_ml_autograd::GradientRegistry;
    use oxide_ml_graph::Op;

    let registry = GradientRegistry::builtin();
    for op in [
        Op::Sqrt,
        Op::Exp,
        Op::Log,
        Op::Abs,
        Op::LogSoftmax,
        Op::Slice,
        Op::Embedding,
    ] {
        assert!(registry.has_rule(&op), "missing rule for {}", op.name());
    }
    for op in [Op::AbsBackward, Op::SliceBackward, Op::EmbeddingBackward] {
        assert!(op.is_backward());
        assert!(
            !registry.has_rule(&op),
            "{} must stay unregistered",
            op.name()
        );
    }
}
