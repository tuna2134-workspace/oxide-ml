use oxide_ml::prelude::*;

fn run_loss(
    graph: &Graph,
    params: &ParameterStore,
    inputs: &ExecutionInputs,
    loss: ValueId,
) -> f32 {
    let executor = Executor::<f32>::new(graph.clone()).unwrap();
    let result = executor.run(params, inputs).unwrap();
    result.value(loss).unwrap().data()[0]
}

fn assert_gradients(
    forward: &Graph,
    training: &TrainingGraph,
    params: &ParameterStore,
    inputs: &ExecutionInputs,
    tolerance: f32,
) {
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(params, inputs)
        .unwrap();
    let grads = collect_graph_gradients(training, &result).unwrap();
    let eps = 1e-2f32;
    for parameter in forward.parameters() {
        let analytic = grads[&parameter.id].data().to_vec();
        let mut numeric = Vec::with_capacity(analytic.len());
        for index in 0..analytic.len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus.get_mut(parameter.id).unwrap().data.data_mut()[index] += eps;
            minus.get_mut(parameter.id).unwrap().data.data_mut()[index] -= eps;
            let plus_loss = run_loss(forward, &plus, inputs, training.loss);
            let minus_loss = run_loss(forward, &minus, inputs, training.loss);
            numeric.push((plus_loss - minus_loss) / (2.0 * eps));
        }
        for (index, (&a, &n)) in analytic.iter().zip(&numeric).enumerate() {
            let scale = a.abs().max(n.abs()).max(1.0);
            let error = (a - n).abs() / scale;
            assert!(
                error < tolerance,
                "parameter {:?}[{index}] analytic={a} numeric={n} rel_error={error}",
                parameter.name
            );
        }
    }
}

#[test]
fn mlp_mse_gradients_match_finite_differences() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 3]), DType::F32);
    let target = builder.input("target", Shape::from_static([4, 1]), DType::F32);
    let w1 = builder
        .try_parameter("w1", Shape::from_static([3, 4]), DType::F32, true)
        .unwrap();
    let b1 = builder
        .try_parameter("b1", Shape::from_static([4]), DType::F32, true)
        .unwrap();
    let w2 = builder
        .try_parameter("w2", Shape::from_static([4, 1]), DType::F32, true)
        .unwrap();
    let b2 = builder
        .try_parameter("b2", Shape::from_static([1]), DType::F32, true)
        .unwrap();
    let h = builder.matmul(x, w1).unwrap();
    let h = builder.add(h, b1).unwrap();
    let h = builder.relu(h).unwrap();
    let y = builder.matmul(h, w2).unwrap();
    let y = builder.add(y, b2).unwrap();
    let loss = builder.mse_loss(y, target, Reduction::Mean).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();

    let mut params = init_parameters::<f32>(&forward, 7).unwrap();
    params.get_mut(ParameterId(0)).unwrap().data = DenseTensor::from_vec(
        [3, 4],
        vec![
            0.5, -0.2, 0.1, 0.3, 0.8, 0.4, -0.6, 0.2, 0.1, 0.7, 0.3, -0.4,
        ],
    )
    .unwrap();
    params.get_mut(ParameterId(1)).unwrap().data =
        DenseTensor::from_vec([4], vec![0.1, -0.2, 0.3, 0.05]).unwrap();
    params.get_mut(ParameterId(2)).unwrap().data =
        DenseTensor::from_vec([4, 1], vec![0.7, -0.3, 0.2, 0.6]).unwrap();
    params.get_mut(ParameterId(3)).unwrap().data = DenseTensor::from_vec([1], vec![0.1]).unwrap();

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [4, 3],
            vec![
                1.0, 2.0, -1.0, 0.5, -1.0, 2.0, -1.0, 0.5, 1.0, 2.0, -2.0, 0.5,
            ],
        )
        .unwrap(),
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([4, 1], vec![1.0, -1.0, 0.5, 2.0]).unwrap(),
    );
    assert_gradients(&forward, &training, &params, &inputs, 2e-2);
}

#[test]
fn softmax_cross_entropy_gradients_match() {
    let mut builder = GraphBuilder::new();
    let logits = builder.input("logits", Shape::from_static([3, 4]), DType::F32);
    let target = builder.input("target", Shape::from_static([3]), DType::F32);
    let loss = builder
        .cross_entropy_loss(logits, target, Reduction::Mean)
        .unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let params = ParameterStore::<f32>::new();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [3, 4],
            vec![
                0.2, 1.0, -0.3, 0.5, -1.0, 0.2, 0.7, 0.1, 0.4, -0.5, 0.0, 1.2,
            ],
        )
        .unwrap(),
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([3], vec![1.0, 0.0, 3.0]).unwrap(),
    );
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    let grad = training.gradient_for_value(forward.inputs()[0]).unwrap();
    let analytic = result.value(grad).unwrap().data().to_vec();
    let eps = 1e-2f32;
    let mut numeric = Vec::new();
    for index in 0..12 {
        let mut plus = inputs.clone();
        let mut minus = inputs.clone();
        let mut plus_data = plus.get(forward.inputs()[0]).unwrap().data().to_vec();
        let mut minus_data = minus.get(forward.inputs()[0]).unwrap().data().to_vec();
        plus_data[index] += eps;
        minus_data[index] -= eps;
        plus.insert(
            forward.inputs()[0],
            DenseTensor::from_vec([3, 4], plus_data).unwrap(),
        );
        minus.insert(
            forward.inputs()[0],
            DenseTensor::from_vec([3, 4], minus_data).unwrap(),
        );
        let plus_loss = run_loss(&forward, &params, &plus, loss);
        let minus_loss = run_loss(&forward, &params, &minus, loss);
        numeric.push((plus_loss - minus_loss) / (2.0 * eps));
    }
    for (index, (&a, &n)) in analytic.iter().zip(&numeric).enumerate() {
        assert!(
            (a - n).abs() < 5e-2,
            "index {index}: analytic={a}, numeric={n}"
        );
    }
}

#[test]
fn layer_norm_gradients_match() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 4]), DType::F32);
    let gamma = builder
        .try_parameter("gamma", Shape::from_static([4]), DType::F32, true)
        .unwrap();
    let beta = builder
        .try_parameter("beta", Shape::from_static([4]), DType::F32, true)
        .unwrap();
    let y = builder.layer_norm(x, gamma, beta, vec![4], 1e-5).unwrap();
    let loss = builder.reduce_mean(y, vec![0, 1], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let mut params = init_parameters::<f32>(&forward, 2).unwrap();
    params.get_mut(ParameterId(0)).unwrap().data =
        DenseTensor::from_vec([4], vec![1.0, 0.5, -0.3, 0.8]).unwrap();
    params.get_mut(ParameterId(1)).unwrap().data =
        DenseTensor::from_vec([4], vec![0.1, -0.2, 0.3, 0.0]).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([2, 4], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, 0.0]).unwrap(),
    );
    assert_gradients(&forward, &training, &params, &inputs, 3e-2);
}

#[test]
fn conv2d_gradients_match() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([1, 2, 3, 3]), DType::F32);
    let w = builder
        .try_parameter("w", Shape::from_static([2, 2, 2, 2]), DType::F32, true)
        .unwrap();
    let b = builder
        .try_parameter("b", Shape::from_static([2]), DType::F32, true)
        .unwrap();
    let y = builder
        .conv2d(
            x,
            w,
            Some(b),
            Conv2dConfig {
                stride: [1, 1],
                padding: [1, 1],
                dilation: [1, 1],
                groups: 1,
            },
        )
        .unwrap();
    let loss = builder.reduce_mean(y, vec![0, 1, 2, 3], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let mut params = init_parameters::<f32>(&forward, 11).unwrap();
    params.get_mut(ParameterId(0)).unwrap().data = DenseTensor::from_vec(
        [2, 2, 2, 2],
        vec![
            0.2, -0.1, 0.3, 0.05, 0.1, 0.4, -0.2, 0.3, -0.3, 0.2, 0.1, -0.4, 0.5, 0.05, -0.1, 0.2,
        ],
    )
    .unwrap();
    params.get_mut(ParameterId(1)).unwrap().data =
        DenseTensor::from_vec([2], vec![0.1, -0.2]).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec(
            [1, 2, 3, 3],
            (0..18).map(|i| (i as f32 * 0.1) - 0.5).collect(),
        )
        .unwrap(),
    );
    assert_gradients(&forward, &training, &params, &inputs, 4e-2);
}

#[test]
fn broadcast_add_gradient_reduces_to_bias_shape() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([3, 2]), DType::F32);
    let b = builder
        .try_parameter("b", Shape::from_static([2]), DType::F32, true)
        .unwrap();
    let y = builder.add(x, b).unwrap();
    let loss = builder.reduce_sum(y, vec![0, 1], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let mut params = init_parameters::<f32>(&forward, 1).unwrap();
    params.get_mut(ParameterId(0)).unwrap().data =
        DenseTensor::from_vec([2], vec![0.0, 0.0]).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    let grad = training.parameter_gradients().next().unwrap().1;
    assert_eq!(result.value(grad).unwrap().data(), &[3.0, 3.0]);
}
