//! Custom kernel registration, execution, specialization and autograd.

use std::sync::Arc;

use oxide_ml::prelude::*;

fn activation_descriptor() -> KernelDescriptor {
    KernelDescriptor::new(
        "custom_activation",
        DType::F32,
        vec![TensorConstraint::any().rank(2)],
        vec![TensorConstraint::any().rank(2)],
    )
    .with_launch(LaunchConfig::one_dimensional(256))
    .with_architecture("sm_75")
    .with_autograd()
}

fn activation_forward(call: &CustomCall<'_, f32>) -> Result<Vec<DenseTensor<f32>>> {
    let input = &call.inputs[0];
    let output = input.map(|value| value / (1.0 + value.abs()));
    Ok(vec![output])
}

fn activation_backward(
    builder: &mut GraphBuilder,
    node: &Node,
    gradients: &[Option<ValueId>],
) -> Result<Vec<Option<ValueId>>> {
    let dy = gradients
        .first()
        .copied()
        .flatten()
        .ok_or_else(|| Error::Autograd("custom activation has no upstream gradient".into()))?;
    let x = node.inputs[0];
    let absolute = builder.abs(x)?;
    let one = builder.constant_scalar(1.0);
    let denominator = builder.add(absolute, one)?;
    let squared = builder.mul(denominator, denominator)?;
    let dx = builder.div(dy, squared)?;
    Ok(vec![Some(dx)])
}

fn registry() -> Arc<CustomKernelRegistry<f32>> {
    let mut registry = CustomKernelRegistry::new();
    let kernel = CustomKernel::new(activation_descriptor(), Arc::new(activation_forward))
        .unwrap()
        .with_backward(Arc::new(activation_backward));
    registry.register(kernel).unwrap();
    Arc::new(registry)
}

#[derive(Module)]
struct CustomModel {
    linear: Linear,
}

impl CustomModel {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let x = self.linear.forward(x);
        apply_kernel(x, "custom_activation")
    }
}

#[test]
fn descriptor_validation_and_operator() {
    let descriptor = activation_descriptor();
    descriptor.validate().unwrap();
    assert_eq!(descriptor.launch.grid_for(1000), [4, 1, 1]);
    let operator = KernelOperator::new(descriptor);
    assert_eq!(operator.name(), "custom_activation");

    let bad = KernelDescriptor::new("bad", DType::F32, vec![], vec![]);
    assert!(bad.validate().is_err());
}

#[test]
fn custom_kernel_executes_on_the_cpu_reference_backend() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let y = builder
        .custom("custom_activation", &[x], vec![Shape::from_static([2, 2])])
        .unwrap()[0];
    let graph = builder.finish(&[y]).unwrap();
    let executor = Executor::with_custom(graph, registry()).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        ValueId(0),
        DenseTensor::from_vec([2, 2], vec![1.0, -1.0, 2.0, -2.0]).unwrap(),
    );
    let result = executor.run(&ParameterStore::new(), &inputs).unwrap();
    let output = result.output(0).unwrap();
    assert!((output.data()[0] - 0.5).abs() < 1e-6);
    assert!((output.data()[1] + 0.5).abs() < 1e-6);
    assert!((output.data()[2] - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn missing_registry_is_an_explicit_error() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([1, 2]), DType::F32);
    let y = builder
        .custom("custom_activation", &[x], vec![Shape::from_static([1, 2])])
        .unwrap()[0];
    let graph = builder.finish(&[y]).unwrap();
    let executor = Executor::new(graph).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        ValueId(0),
        DenseTensor::from_vec([1, 2], vec![1.0, 2.0]).unwrap(),
    );
    let error = executor.run(&ParameterStore::new(), &inputs).unwrap_err();
    assert!(error.to_string().contains("custom kernel registry"));
}

#[test]
fn specialization_key_tracks_shape_architecture_and_dtype() {
    let descriptor = activation_descriptor();
    let a = SpecializationKey::new(&descriptor, vec![vec![2, 2]], vec![vec![2, 2]]);
    let b = SpecializationKey::new(&descriptor, vec![vec![4, 4]], vec![vec![4, 4]]);
    let c = a.clone().with_architecture("sm_90");
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_eq!(a.architecture, "sm_75");
}

#[test]
fn model_with_custom_kernel_runs_and_trains() {
    let mut model = CustomModel {
        linear: Linear::with_seed(2, 2, true, 3),
    };
    let input = Tensor::<f32>::from_vec([2, 2], vec![0.5, -0.5, 1.5, -1.5]).unwrap();
    let options = ModelCompileOptions {
        custom_kernels: Some(registry()),
        ..Default::default()
    };
    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [2, 2])],
        options.clone(),
    )
    .unwrap();
    let store = compiled.parameter_store(&model).unwrap();
    let outputs = compiled.run(&store, &[("input0", &input)]).unwrap();
    assert_eq!(outputs[0].dims(), vec![2, 2]);

    let dataset = TensorDataset::from_tensors(
        Tensor::<f32>::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap(),
        Tensor::<f32>::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap(),
    )
    .unwrap();
    let mut trainer = Trainer::builder()
        .optimizer(Sgd::new(0.05).unwrap())
        .loss(MSELoss::new())
        .batch_size(2)
        .custom_kernels(registry())
        .build()
        .unwrap();
    let report = trainer.fit(&mut model, &dataset, 20).unwrap();
    assert!(report.final_loss.is_finite());

    // The same graph trains through the low-level autograd API.
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 2]), DType::F32);
    let y = builder
        .custom("custom_activation", &[x], vec![Shape::from_static([4, 2])])
        .unwrap()[0];
    let loss = builder.reduce_mean(y, vec![0, 1], false).unwrap();
    let graph = builder.finish(&[loss]).unwrap();
    let gradient_registry = GradientRegistry::with_custom_kernels(registry().as_ref());
    let training_graph =
        build_training_graph_with_registry(&graph, loss, &gradient_registry).unwrap();
    assert!(training_graph.parameter_grads.is_empty());
}

#[test]
fn missing_backward_is_reported_before_training() {
    let mut registry = CustomKernelRegistry::new();
    registry
        .register(CustomKernel::new(activation_descriptor(), Arc::new(activation_forward)).unwrap())
        .unwrap();
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([1, 2]), DType::F32);
    let y = builder
        .custom("custom_activation", &[x], vec![Shape::from_static([1, 2])])
        .unwrap()[0];
    let loss = builder.reduce_mean(y, vec![0, 1], false).unwrap();
    let graph = builder.finish(&[loss]).unwrap();
    let gradient_registry = GradientRegistry::with_custom_kernels(&registry);
    let error = build_training_graph_with_registry(&graph, loss, &gradient_registry).unwrap_err();
    assert!(error.to_string().contains("no backward rule"), "{error}");
}
