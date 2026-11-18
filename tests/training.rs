use oxide_ml::prelude::*;

fn inputs(graph: &Graph, x: DenseTensor, y: DenseTensor) -> ExecutionInputs {
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(graph.inputs()[0], x);
    inputs.insert(graph.inputs()[1], y);
    inputs
}

#[test]
fn linear_regression_recovers_coefficients() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([8, 1]), DType::F32);
    let target = builder.input("y", Shape::from_static([8, 1]), DType::F32);
    let weight = builder
        .try_parameter(
            "linear.weight",
            Shape::from_static([1, 1]),
            DType::F32,
            true,
        )
        .unwrap();
    let bias = builder
        .try_parameter("linear.bias", Shape::from_static([1]), DType::F32, true)
        .unwrap();
    let prediction = builder.matmul(x, weight).unwrap();
    let prediction = builder.add(prediction, bias).unwrap();
    let loss = builder
        .mse_loss(prediction, target, Reduction::Mean)
        .unwrap();
    let forward = builder.finish(&[prediction, loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let mut params = init_parameters::<f32>(&forward, 5).unwrap();
    params.get_mut(ParameterId(0)).unwrap().data =
        DenseTensor::from_vec([1, 1], vec![0.0]).unwrap();
    params.get_mut(ParameterId(1)).unwrap().data = DenseTensor::from_vec([1], vec![0.0]).unwrap();
    let optimizer = AdamW::with_config(0.08, 0.9, 0.999, 1e-8, 0.0).unwrap();
    let mut trainer = Trainer::new(training, params, Box::new(optimizer)).unwrap();

    let x_data = DenseTensor::from_vec([8, 1], (0..8).map(|i| i as f32 * 0.5).collect()).unwrap();
    let y_data = DenseTensor::from_vec(
        [8, 1],
        (0..8).map(|i| 2.0 * (i as f32 * 0.5) + 1.0).collect(),
    )
    .unwrap();
    let inputs = inputs(&forward, x_data, y_data);
    let mut final_loss = f32::INFINITY;
    for _ in 0..600 {
        final_loss = trainer.step(&inputs).unwrap();
    }
    assert!(final_loss < 1e-4, "final loss was {final_loss}");
    let learned_w = trainer.parameters.get(ParameterId(0)).unwrap().data.data()[0];
    let learned_b = trainer.parameters.get(ParameterId(1)).unwrap().data.data()[0];
    assert!((learned_w - 2.0).abs() < 5e-2, "weight={learned_w}");
    assert!((learned_b - 1.0).abs() < 5e-2, "bias={learned_b}");
}

#[test]
fn xor_mlp_converges() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 2]), DType::F32);
    let target = builder.input("y", Shape::from_static([4, 1]), DType::F32);
    let mlp = Mlp::new(&mut builder, "mlp", &[2, 8, 1], ActivationKind::Tanh, true).unwrap();
    let logits = mlp.forward(&mut builder, x).unwrap();
    let prediction = builder.sigmoid(logits).unwrap();
    let loss = builder
        .mse_loss(prediction, target, Reduction::Mean)
        .unwrap();
    let forward = builder.finish(&[prediction, loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let params = init_parameters::<f32>(&forward, 42).unwrap();
    let optimizer = AdamW::with_config(0.08, 0.9, 0.999, 1e-8, 0.0).unwrap();
    let mut trainer = Trainer::new(training, params, Box::new(optimizer)).unwrap();

    let x_data = DenseTensor::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();
    let y_data = DenseTensor::from_vec([4, 1], vec![0., 1., 1., 0.]).unwrap();
    let inputs = inputs(&forward, x_data, y_data);
    let mut final_loss = f32::INFINITY;
    for _ in 0..2500 {
        final_loss = trainer.step(&inputs).unwrap();
    }
    assert!(final_loss < 1e-3, "XOR loss did not converge: {final_loss}");
    let result = trainer.run(&inputs).unwrap();
    let prediction = result.output(0).unwrap().data();
    assert!(prediction[0] < 0.1, "0 xor 0 = {}", prediction[0]);
    assert!(prediction[1] > 0.9, "0 xor 1 = {}", prediction[1]);
    assert!(prediction[2] > 0.9, "1 xor 0 = {}", prediction[2]);
    assert!(prediction[3] < 0.1, "1 xor 1 = {}", prediction[3]);
}

#[test]
fn symbolic_batch_can_change_between_runs() {
    let mut builder = GraphBuilder::new();
    let batch = builder.symbol("batch");
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(3)]),
        DType::F32,
    );
    let weight = builder
        .try_parameter("w", Shape::from_static([3, 2]), DType::F32, true)
        .unwrap();
    let y = builder.matmul(x, weight).unwrap();
    let graph = builder.finish(&[y]).unwrap();
    let mut params = init_parameters::<f32>(&graph, 1).unwrap();
    params.get_mut(ParameterId(0)).unwrap().data =
        DenseTensor::from_vec([3, 2], vec![1., 0., 0., 1., 1., 1.]).unwrap();
    let executor = Executor::<f32>::new(graph.clone()).unwrap();
    for batch_size in [2usize, 5, 1] {
        let mut inputs = ExecutionInputs::<f32>::new();
        inputs.insert(graph.inputs()[0], DenseTensor::zeros([batch_size, 3]));
        let result = executor.run(&params, &inputs).unwrap();
        assert_eq!(result.output(0).unwrap().dims(), vec![batch_size, 2]);
    }
}

#[test]
fn inference_executor_excludes_backward_plan() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let target = builder.input("y", Shape::from_static([2, 1]), DType::F32);
    let mlp = Mlp::new(&mut builder, "mlp", &[2, 3, 1], ActivationKind::ReLU, true).unwrap();
    let prediction = mlp.forward(&mut builder, x).unwrap();
    let loss = builder
        .mse_loss(prediction, target, Reduction::Mean)
        .unwrap();
    let forward = builder.finish(&[prediction, loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    assert!(training.graph.nodes().len() > forward.nodes().len());
    let params = init_parameters::<f32>(&forward, 3).unwrap();
    let trainer = Trainer::new(training, params, Box::new(Sgd::new(0.1).unwrap())).unwrap();
    let inference = trainer.inference_executor().unwrap();
    assert_eq!(inference.graph().nodes().len(), forward.nodes().len());
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([2, 1], vec![0., 1.]).unwrap(),
    );
    let result = inference.run(&trainer.parameters, &inputs).unwrap();
    assert_eq!(result.output(0).unwrap().dims(), vec![2, 1]);
}

#[test]
fn trainer_can_step_with_memory_plan() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 2]), DType::F32);
    let target = builder.input("y", Shape::from_static([4, 1]), DType::F32);
    let w = builder
        .try_parameter("w", Shape::from_static([2, 1]), DType::F32, true)
        .unwrap();
    let b = builder
        .try_parameter("b", Shape::from_static([1]), DType::F32, true)
        .unwrap();
    let product = builder.matmul(x, w).unwrap();
    let prediction = builder.add(product, b).unwrap();
    let loss = builder
        .mse_loss(prediction, target, Reduction::Mean)
        .unwrap();
    let forward = builder.finish(&[loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let params = init_parameters::<f32>(&forward, 3).unwrap();
    let plan = MemoryPlanner::default().plan(&training.graph).unwrap();
    let optimizer = AdamW::with_config(0.05, 0.9, 0.999, 1e-8, 0.0).unwrap();
    let mut trainer = Trainer::new(training, params, Box::new(optimizer)).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([4, 2], vec![1., 2., -1., 0.5, 2., -2., 0.25, 1.5]).unwrap(),
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([4, 1], vec![1., -0.5, 0.25, 2.]).unwrap(),
    );
    let before = trainer.loss(&trainer.run(&inputs).unwrap()).unwrap();
    let after = trainer.step_planned(&inputs, &plan).unwrap();
    assert!(after.is_finite());
    assert!(after <= before);
}
