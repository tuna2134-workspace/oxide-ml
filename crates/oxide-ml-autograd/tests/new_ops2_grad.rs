//! Numerical gradient and value checks for Cast/Concat/Split/MulAdd/MatMulBias.

use oxide_ml_autograd::{TrainingGraph, build_training_graph};
use oxide_ml_core::{DType, DenseTensor, ParameterStore, Shape, ValueId};
use oxide_ml_cpu::{ExecutionInputs, Executor};
use oxide_ml_graph::{Graph, GraphBuilder};

const EPS: f32 = 1e-2;
const TOLERANCE: f32 = 3e-2;

fn run_value(graph: &Graph, inputs: &ExecutionInputs<f32>, value: ValueId) -> f32 {
    Executor::<f32>::new(graph.clone())
        .unwrap()
        .run(&ParameterStore::new(), inputs)
        .unwrap()
        .value(value)
        .unwrap()
        .data()[0]
}

fn check_input_gradient(
    forward: &Graph,
    training: &TrainingGraph,
    inputs: &ExecutionInputs<f32>,
    value: ValueId,
) {
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(&ParameterStore::new(), inputs)
        .unwrap();
    let gradient = training
        .gradient_for_value(value)
        .expect("value participates in backward");
    let analytic = result.value(gradient).unwrap().data().to_vec();
    let dims = inputs.get(value).unwrap().dims();
    let data = inputs.get(value).unwrap().data().to_vec();
    let mut numeric = vec![0.0f32; data.len()];
    for index in 0..data.len() {
        let mut plus = inputs.clone();
        let mut minus = inputs.clone();
        let mut plus_data = data.clone();
        let mut minus_data = data.clone();
        plus_data[index] += EPS;
        minus_data[index] -= EPS;
        plus.insert(value, DenseTensor::from_vec(&dims, plus_data).unwrap());
        minus.insert(value, DenseTensor::from_vec(&dims, minus_data).unwrap());
        numeric[index] = (run_value(forward, &plus, training.loss)
            - run_value(forward, &minus, training.loss))
            / (2.0 * EPS);
    }
    assert_eq!(analytic.len(), numeric.len());
    for (index, (analytic, numeric)) in analytic.iter().zip(&numeric).enumerate() {
        let scale = analytic.abs().max(numeric.abs()).max(1.0);
        assert!(
            (analytic - numeric).abs() / scale < TOLERANCE,
            "gradient[{index}] analytic={analytic} numeric={numeric}"
        );
    }
}

#[test]
fn mul_add_values_and_gradients() {
    let mut b = GraphBuilder::new();
    let a = b.input("a", Shape::from_static([2, 2]), DType::F32);
    let c = b.input("c", Shape::from_static([2, 2]), DType::F32);
    let d = b.input("d", Shape::from_static([2, 2]), DType::F32);
    let y = b.mul_add(a, c, d).unwrap();
    let loss = b.reduce_mean(y, vec![0, 1], false).unwrap();
    let forward = b.finish(&[loss]).unwrap();

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        a,
        DenseTensor::from_vec([2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    inputs.insert(
        c,
        DenseTensor::from_vec([2, 2], vec![2., 3., 4., 5.]).unwrap(),
    );
    inputs.insert(
        d,
        DenseTensor::from_vec([2, 2], vec![0.5, 0.5, 0.5, 0.5]).unwrap(),
    );
    let expected = (1.0 * 2.0 + 0.5 + 2.0 * 3.0 + 0.5 + 3.0 * 4.0 + 0.5 + 4.0 * 5.0 + 0.5) / 4.0;
    let actual = run_value(&forward, &inputs, loss);
    assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");

    let training = build_training_graph(&forward, loss).unwrap();
    check_input_gradient(&forward, &training, &inputs, a);
    check_input_gradient(&forward, &training, &inputs, c);
    check_input_gradient(&forward, &training, &inputs, d);
}

#[test]
fn matmul_bias_values_and_gradients() {
    let mut b = GraphBuilder::new();
    let a = b.input("a", Shape::from_static([2, 3]), DType::F32);
    let w = b.input("w", Shape::from_static([2, 3]), DType::F32);
    let bias = b.input("bias", Shape::from_static([2]), DType::F32);
    let y = b.matmul_bias(a, w, bias, false, true).unwrap();
    let loss = b.reduce_mean(y, vec![0, 1], false).unwrap();
    let forward = b.finish(&[loss]).unwrap();

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        a,
        DenseTensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    );
    inputs.insert(
        w,
        DenseTensor::from_vec([2, 3], vec![1., 0., 0., 0., 1., 0.]).unwrap(),
    );
    inputs.insert(bias, DenseTensor::from_vec([2], vec![10., 20.]).unwrap());
    // a @ w^T = [[1,2],[4,5]]; + bias = [[11,22],[14,25]]; mean of squares?
    let executor = Executor::<f32>::new(forward.clone()).unwrap();
    let result = executor.run(&ParameterStore::new(), &inputs).unwrap();
    let loss_value = result.value(loss).unwrap().data()[0];
    let expected = (11.0 + 22.0 + 14.0 + 25.0) / 4.0;
    assert!(
        (loss_value - expected).abs() < 1e-5,
        "{loss_value} vs {expected}"
    );

    let training = build_training_graph(&forward, loss).unwrap();
    check_input_gradient(&forward, &training, &inputs, a);
    check_input_gradient(&forward, &training, &inputs, w);
    check_input_gradient(&forward, &training, &inputs, bias);
}

#[test]
fn concat_split_values_and_gradients() {
    let mut b = GraphBuilder::new();
    let a = b.input("a", Shape::from_static([2, 2]), DType::F32);
    let c = b.input("c", Shape::from_static([2, 3]), DType::F32);
    let joined = b.concat(&[a, c], 1).unwrap();
    let parts = b.split(joined, 1, vec![2, 3]).unwrap();
    // Loss mixes the parts so each gradient is observable.
    let mean_a = b.reduce_mean(parts[0], vec![0, 1], false).unwrap();
    let mean_c = b.reduce_mean(parts[1], vec![0, 1], false).unwrap();
    let loss = b.add(mean_a, mean_c).unwrap();
    let forward = b.finish(&[parts[0], parts[1], loss]).unwrap();

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        a,
        DenseTensor::from_vec([2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    inputs.insert(
        c,
        DenseTensor::from_vec([2, 3], vec![5., 6., 7., 8., 9., 10.]).unwrap(),
    );
    let result = Executor::<f32>::new(forward.clone())
        .unwrap()
        .run(&ParameterStore::new(), &inputs)
        .unwrap();
    assert_eq!(result.value(parts[0]).unwrap().dims(), vec![2, 2]);
    assert_eq!(result.value(parts[1]).unwrap().dims(), vec![2, 3]);
    assert_eq!(result.value(parts[0]).unwrap().data(), &[1., 2., 3., 4.]);
    assert_eq!(
        result.value(parts[1]).unwrap().data(),
        &[5., 6., 7., 8., 9., 10.]
    );

    let training = build_training_graph(&forward, loss).unwrap();
    check_input_gradient(&forward, &training, &inputs, a);
    check_input_gradient(&forward, &training, &inputs, c);
}

#[test]
fn cast_is_identity_for_the_executor_dtype() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([4]), DType::F32);
    let y = b.cast(x, DType::F32).unwrap();
    let forward = b.finish(&[y]).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(x, DenseTensor::from_vec([4], vec![1., 2., 3., 4.]).unwrap());
    let result = Executor::<f32>::new(forward)
        .unwrap()
        .run(&ParameterStore::new(), &inputs)
        .unwrap();
    assert_eq!(result.value(y).unwrap().data(), &[1., 2., 3., 4.]);

    // A dtype-changing cast is rejected by the single-dtype reference executor.
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([4]), DType::F32);
    let y = b.cast(x, DType::F64).unwrap();
    let forward = b.finish(&[y]).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(x, DenseTensor::from_vec([4], vec![1., 2., 3., 4.]).unwrap());
    let error = Executor::<f32>::new(forward)
        .unwrap()
        .run(&ParameterStore::new(), &inputs)
        .unwrap_err();
    assert!(matches!(error, oxide_ml_core::Error::DTypeMismatch { .. }));
}
