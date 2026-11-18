use oxide_ml::prelude::*;

fn run_loss(
    graph: &Graph,
    params: &ParameterStore,
    inputs: &ExecutionInputs,
    loss: ValueId,
) -> f32 {
    let result = Executor::<f32>::new(graph.clone())
        .unwrap()
        .run(params, inputs)
        .unwrap();
    result.value(loss).unwrap().data()[0]
}

fn assert_input_gradients(
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
    for &input_id in forward.inputs() {
        let Some(gradient) = training.gradient_for_value(input_id) else {
            continue;
        };
        let analytic = result.value(gradient).unwrap().data().to_vec();
        let original = inputs.get(input_id).unwrap();
        let eps = 1e-2f32;
        let mut numeric = Vec::with_capacity(analytic.len());
        for index in 0..analytic.len() {
            let mut plus = inputs.clone();
            let mut minus = inputs.clone();
            let mut plus_data = original.data().to_vec();
            let mut minus_data = original.data().to_vec();
            plus_data[index] += eps;
            minus_data[index] -= eps;
            plus.insert(
                input_id,
                DenseTensor::from_vec(original.dims(), plus_data).unwrap(),
            );
            minus.insert(
                input_id,
                DenseTensor::from_vec(original.dims(), minus_data).unwrap(),
            );
            let plus_loss = run_loss(forward, params, &plus, training.loss);
            let minus_loss = run_loss(forward, params, &minus, training.loss);
            numeric.push((plus_loss - minus_loss) / (2.0 * eps));
        }
        for (index, (&a, &n)) in analytic.iter().zip(&numeric).enumerate() {
            let scale = a.abs().max(n.abs()).max(1.0);
            assert!(
                (a - n).abs() / scale < tolerance,
                "input {input_id}[{index}] analytic={a} numeric={n}"
            );
        }
    }
}

#[test]
fn transpose_last_and_transposed_matmul_are_correct() {
    let mut builder = GraphBuilder::new();
    let a = builder.input("a", Shape::from_static([2, 3]), DType::F32);
    let w = builder
        .try_parameter("w", Shape::from_static([4, 3]), DType::F32, true)
        .unwrap();
    let y = builder.matmul_with(a, w, false, true).unwrap();
    let graph = builder.finish(&[y]).unwrap();
    let mut params = init_parameters::<f32>(&graph, 1).unwrap();
    params.get_mut(ParameterId(0)).unwrap().data =
        DenseTensor::from_vec([4, 3], vec![1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.])
            .unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap(),
    );
    let result = Executor::<f32>::new(graph)
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    assert_eq!(
        result.output(0).unwrap().data(),
        &[1., 2., 3., 6., 4., 5., 6., 15.]
    );
}

#[test]
fn activation_gradient_checks() {
    for (name, build) in [
        (
            "sigmoid",
            (|b: &mut GraphBuilder, x| b.sigmoid(x))
                as fn(&mut GraphBuilder, ValueId) -> Result<ValueId>,
        ),
        (
            "tanh",
            (|b: &mut GraphBuilder, x| b.tanh(x))
                as fn(&mut GraphBuilder, ValueId) -> Result<ValueId>,
        ),
        (
            "gelu",
            (|b: &mut GraphBuilder, x| b.gelu(x))
                as fn(&mut GraphBuilder, ValueId) -> Result<ValueId>,
        ),
    ] {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
        let y = build(&mut builder, x).unwrap();
        let loss = builder.reduce_sum(y, vec![0, 1], false).unwrap();
        let forward = builder.finish(&[loss]).unwrap();
        let training = build_training_graph(&forward, loss).unwrap();
        let params = ParameterStore::<f32>::new();
        let mut inputs = ExecutionInputs::<f32>::new();
        inputs.insert(
            forward.inputs()[0],
            DenseTensor::from_vec([2, 3], vec![-1.2, -0.4, 0.3, 0.8, 1.5, -2.0]).unwrap(),
        );
        assert_input_gradients(&forward, &training, &params, &inputs, 2e-2);
        assert!(!name.is_empty());
    }
}

#[test]
fn pooling_forward_and_backward_are_consistent() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([1, 1, 3, 3]), DType::F32);
    let pooled = builder
        .max_pool2d(
            x,
            Pool2dConfig {
                kernel: [2, 2],
                stride: [1, 1],
                padding: [0, 0],
                dilation: [1, 1],
            },
        )
        .unwrap();
    let loss = builder
        .reduce_mean(pooled, vec![0, 1, 2, 3], false)
        .unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let params = ParameterStore::<f32>::new();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([1, 1, 3, 3], vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap(),
    );
    let result = Executor::<f32>::new(forward.clone())
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    // max pool windows are [5, 6, 8, 9]; the scalar loss is their mean.
    assert_eq!(result.output(0).unwrap().data(), &[7.0]);
    assert_input_gradients(&forward, &training, &params, &inputs, 2e-2);
}

#[test]
fn avg_pool_backward_distributes_gradient() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([1, 1, 2, 2]), DType::F32);
    let pooled = builder
        .avg_pool2d(
            x,
            Pool2dConfig {
                kernel: [2, 2],
                stride: [2, 2],
                padding: [0, 0],
                dilation: [1, 1],
            },
        )
        .unwrap();
    let loss = builder.reduce_sum(pooled, vec![0, 1, 2, 3], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let params = ParameterStore::<f32>::new();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([1, 1, 2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    let gradient = training.gradient_for_value(forward.inputs()[0]).unwrap();
    let gradient = result.value(gradient).unwrap();
    for value in gradient.data() {
        assert!((*value - 0.25).abs() < 1e-6);
    }
}

#[test]
fn shared_parameter_gradients_accumulate() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([1, 1]), DType::F32);
    let w = builder
        .try_parameter("w", Shape::from_static([1, 1]), DType::F32, true)
        .unwrap();
    let first = builder.matmul(x, w).unwrap();
    let second = builder.matmul(x, w).unwrap();
    let sum = builder.add(first, second).unwrap();
    let loss = builder.reduce_sum(sum, vec![0, 1], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let params = init_parameters::<f32>(&forward, 1).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([1, 1], vec![3.0]).unwrap(),
    );
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    let grad = training.parameter_gradients().next().unwrap().1;
    assert_eq!(result.value(grad).unwrap().data(), &[6.0]);
}

#[test]
fn invalid_batch_symbol_binding_is_rejected() {
    let mut builder = GraphBuilder::new();
    let batch = builder.symbol("batch");
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(2)]),
        DType::F32,
    );
    let y = builder.input(
        "y",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(2)]),
        DType::F32,
    );
    let z = builder.add(x, y).unwrap();
    let graph = builder.finish(&[z]).unwrap();
    let executor = Executor::<f32>::new(graph.clone()).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(graph.inputs()[0], DenseTensor::zeros([2, 2]));
    inputs.insert(graph.inputs()[1], DenseTensor::zeros([3, 2]));
    assert!(
        executor
            .run(&ParameterStore::<f32>::new(), &inputs)
            .is_err()
    );
}

#[test]
fn non_f32_graph_is_rejected_by_reference_backend() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([1, 2]), DType::F16);
    let graph = builder.finish(&[x]).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(graph.inputs()[0], DenseTensor::zeros([1, 2]));
    let error = Executor::<f32>::new(graph)
        .unwrap()
        .run(&ParameterStore::<f32>::new(), &inputs)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Unsupported(_) | Error::DTypeMismatch { .. }
    ));
}

#[test]
fn cuda_tagged_graph_is_rejected_by_reference_backend() {
    let mut builder = GraphBuilder::new();
    let x = builder.input_on("x", Shape::from_static([1, 2]), DType::F32, Device::Cuda(0));
    let graph = builder.finish(&[x]).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(graph.inputs()[0], DenseTensor::zeros([1, 2]));
    let error = Executor::<f32>::new(graph)
        .unwrap()
        .run(&ParameterStore::<f32>::new(), &inputs)
        .unwrap_err();
    assert!(matches!(
        error,
        Error::Unsupported(_) | Error::DTypeMismatch { .. }
    ));
}

#[test]
fn add_relu_fused_gradient_matches_finite_difference() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2, 2]), DType::F32);
    let out = builder.single(Op::AddRelu, OpAttrs::None, &[x, y]).unwrap();
    let loss = builder.reduce_sum(out, vec![0, 1], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let params = ParameterStore::<f32>::new();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([2, 2], vec![-1.0, 2.0, -3.0, 4.0]).unwrap(),
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([2, 2], vec![2.0, -1.0, 4.0, -3.0]).unwrap(),
    );
    assert_input_gradients(&forward, &training, &params, &inputs, 2e-2);
}

#[test]
fn missing_gradient_rule_is_reported_before_backward_generation() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2]), DType::F32);
    let out = builder.add(x, y).unwrap();
    let forward = builder.finish(&[out]).unwrap();
    let registry = GradientRegistry::empty();
    let error = build_training_graph_with_registry(&forward, out, &registry).unwrap_err();
    assert!(error.to_string().contains("no gradient rule registered"));
}

#[test]
fn custom_gradient_rule_can_be_registered() {
    fn add_rule(
        _builder: &mut GraphBuilder,
        _node: &oxide_ml_graph::Node,
        upstream: &[Option<ValueId>],
    ) -> Result<Vec<Option<ValueId>>> {
        let dy = upstream[0].unwrap();
        Ok(vec![Some(dy), Some(dy)])
    }

    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2]), DType::F32);
    let out = builder.add(x, y).unwrap();
    let loss = builder.reduce_sum(out, vec![0], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let mut registry = GradientRegistry::builtin();
    registry.register(
        "add",
        std::sync::Arc::new(oxide_ml::FunctionRule::new("test.add", add_rule)),
    );
    let training = build_training_graph_with_registry(&forward, loss, &registry).unwrap();
    assert!(training.gradient_for_value(x).is_some());
    assert!(training.gradient_for_value(y).is_some());
}

#[test]
fn add_relu_broadcast_gradient() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let b = builder.input("b", Shape::from_static([2]), DType::F32);
    let out = builder.single(Op::AddRelu, OpAttrs::None, &[x, b]).unwrap();
    let loss = builder.reduce_sum(out, vec![0, 1], false).unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let params = ParameterStore::<f32>::new();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([2, 2], vec![-1.0, 2.0, 3.0, -4.0]).unwrap(),
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([2], vec![1.5, -1.5]).unwrap(),
    );
    assert_input_gradients(&forward, &training, &params, &inputs, 2e-2);
}
