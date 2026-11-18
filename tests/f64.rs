use oxide_ml::prelude::*;

#[test]
fn f64_forward_and_gradient_check() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 2]), DType::F64);
    let target = builder.input("y", Shape::from_static([4, 1]), DType::F64);
    let w = builder
        .try_parameter("w", Shape::from_static([2, 1]), DType::F64, true)
        .unwrap();
    let b = builder
        .try_parameter("b", Shape::from_static([1]), DType::F64, true)
        .unwrap();
    let product = builder.matmul(x, w).unwrap();
    let prediction = builder.add(product, b).unwrap();
    let loss = builder
        .mse_loss(prediction, target, Reduction::Mean)
        .unwrap();
    let forward = builder.finish(&[prediction, loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let mut params = init_parameters::<f64>(&forward, 99).unwrap();
    params.get_mut(ParameterId(0)).unwrap().data =
        DenseTensor::<f64>::from_vec([2, 1], vec![0.5, -0.25]).unwrap();
    params.get_mut(ParameterId(1)).unwrap().data =
        DenseTensor::<f64>::from_vec([1], vec![0.1]).unwrap();
    let mut inputs = ExecutionInputs::<f64>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::<f64>::from_vec([4, 2], vec![1.0, 2.0, -1.0, 0.5, 2.0, -2.0, 0.25, 1.5])
            .unwrap(),
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::<f64>::from_vec([4, 1], vec![1.0, -0.5, 0.25, 2.0]).unwrap(),
    );
    let executor = Executor::<f64>::new(training.graph.clone()).unwrap();
    let result = executor.run(&params, &inputs).unwrap();
    let gradients = collect_gradients(&training, &result).unwrap();
    let eps = 1e-6;
    for parameter in forward.parameters() {
        for index in 0..parameter.shape.numel_static().unwrap() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus.get_mut(parameter.id).unwrap().data.data_mut()[index] += eps;
            minus.get_mut(parameter.id).unwrap().data.data_mut()[index] -= eps;
            let plus_loss = Executor::<f64>::new(forward.clone())
                .unwrap()
                .run(&plus, &inputs)
                .unwrap()
                .value(loss)
                .unwrap()
                .data()[0];
            let minus_loss = Executor::<f64>::new(forward.clone())
                .unwrap()
                .run(&minus, &inputs)
                .unwrap()
                .value(loss)
                .unwrap()
                .data()[0];
            let numeric = (plus_loss - minus_loss) / (2.0 * eps);
            let analytic = gradients[&parameter.id].data()[index];
            assert!(
                (analytic - numeric).abs() < 1e-7,
                "{}[{index}] analytic={analytic} numeric={numeric}",
                parameter.name
            );
        }
    }
}

#[test]
fn f64_adamw_updates_parameters() {
    let mut store = ParameterStore::<f64>::new();
    store
        .add(
            ParameterId(0),
            "w",
            DenseTensor::<f64>::from_vec([1], vec![1.0]).unwrap(),
            true,
        )
        .unwrap();
    let mut gradients = Gradients::<f64>::new();
    gradients.insert(
        ParameterId(0),
        DenseTensor::<f64>::from_vec([1], vec![0.5]).unwrap(),
    );
    let mut optimizer = AdamW::<f64>::with_config(0.1, 0.9, 0.999, 1e-8, 0.0).unwrap();
    optimizer.step(&mut store, &gradients).unwrap();
    assert!(store.get(ParameterId(0)).unwrap().data.data()[0] < 1.0);
}

#[test]
fn f16_is_rejected_without_conversion() {
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
fn f64_graph_compiles_and_runs_planned() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F64);
    let a = builder
        .constant_tensor(
            ConstantTensor::f64(Shape::from_static([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        )
        .unwrap();
    let b = builder
        .constant_tensor(
            ConstantTensor::f64(Shape::from_static([2, 2]), vec![0.5, 0.5, 0.5, 0.5]).unwrap(),
        )
        .unwrap();
    let c = builder.add(a, b).unwrap();
    let out = builder.mul(x, c).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let compiler = Compiler::<f64>::new(
        Box::new(oxide_ml_cpu::CpuBackend::<f64>::new()),
        CompileOptions::default(),
    );
    let compiled = compiler.compile(graph).unwrap();
    assert!(compiled.memory_plan.is_some());
    let mut inputs = ExecutionInputs::<f64>::new();
    inputs.insert(
        compiled.graph.inputs()[0],
        DenseTensor::<f64>::from_vec([2, 2], vec![2.0, 2.0, 2.0, 2.0]).unwrap(),
    );
    let params = ParameterStore::<f64>::new();
    let executor = Executor::<f64>::new(compiled.graph.clone()).unwrap();
    let normal = executor.run(&params, &inputs).unwrap();
    let (planned, _) = executor
        .run_planned(&params, &inputs, compiled.memory_plan.as_ref().unwrap())
        .unwrap();
    assert_eq!(normal.output(0).unwrap().data(), &[3.0, 5.0, 7.0, 9.0]);
    assert_eq!(
        planned.output(0).unwrap().data(),
        normal.output(0).unwrap().data()
    );
}

#[test]
fn f64_mlp_layer_builds_and_executes() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F64);
    let target = builder.input("y", Shape::from_static([2, 1]), DType::F64);
    let mlp = Mlp::with_dtype(
        &mut builder,
        "mlp",
        &[3, 4, 1],
        ActivationKind::Tanh,
        true,
        DType::F64,
    )
    .unwrap();
    let prediction = mlp.forward(&mut builder, x).unwrap();
    let loss = builder
        .mse_loss(prediction, target, Reduction::Mean)
        .unwrap();
    let graph = builder.finish(&[loss]).unwrap();
    let params = init_parameters::<f64>(&graph, 3).unwrap();
    let mut inputs = ExecutionInputs::<f64>::new();
    inputs.insert(graph.inputs()[0], DenseTensor::<f64>::zeros([2, 3]));
    inputs.insert(graph.inputs()[1], DenseTensor::<f64>::zeros([2, 1]));
    let result = Executor::<f64>::new(graph)
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    assert!(result.output(0).unwrap().data()[0].is_finite());
}

#[test]
fn f64_cross_entropy_and_layer_norm_execute() {
    let mut builder = GraphBuilder::new();
    let logits = builder.input("logits", Shape::from_static([2, 3]), DType::F64);
    let target = builder.input("target", Shape::from_static([2]), DType::F64);
    let loss = builder
        .cross_entropy_loss(logits, target, Reduction::Mean)
        .unwrap();
    let graph = builder.finish(&[loss]).unwrap();
    let mut inputs = ExecutionInputs::<f64>::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::<f64>::from_vec([2, 3], vec![0.2, 1.0, -0.3, 0.5, -1.0, 0.2]).unwrap(),
    );
    inputs.insert(
        graph.inputs()[1],
        DenseTensor::<f64>::from_vec([2], vec![1.0, 0.0]).unwrap(),
    );
    let result = Executor::<f64>::new(graph)
        .unwrap()
        .run(&ParameterStore::<f64>::new(), &inputs)
        .unwrap();
    assert!(result.output(0).unwrap().data()[0].is_finite());

    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F64);
    let gamma = builder
        .try_parameter("gamma", Shape::from_static([3]), DType::F64, true)
        .unwrap();
    let beta = builder
        .try_parameter("beta", Shape::from_static([3]), DType::F64, true)
        .unwrap();
    let y = builder.layer_norm(x, gamma, beta, vec![3], 1e-5).unwrap();
    let graph = builder.finish(&[y]).unwrap();
    let params = init_parameters::<f64>(&graph, 1).unwrap();
    let mut inputs = ExecutionInputs::<f64>::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::<f64>::from_vec([2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap(),
    );
    let result = Executor::<f64>::new(graph)
        .unwrap()
        .run(&params, &inputs)
        .unwrap();
    assert!(
        result
            .output(0)
            .unwrap()
            .data()
            .iter()
            .all(|value| value.is_finite())
    );
}
