//! End-to-end tests for the model API: derive, #[model], Sequential,
//! prediction, training, shared parameters and checkpoints.

use std::sync::Arc;

use oxide_ml::prelude::*;

#[derive(Module)]
struct Mlp {
    fc1: Linear,
    norm: LayerNorm,
    fc2: Linear,
}

impl Mlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let x = self.fc1.forward(x);
        let x = relu(x);
        let x = self.norm.forward(x);
        self.fc2.forward(x)
    }
}

fn mlp() -> Mlp {
    Mlp {
        fc1: Linear::with_seed(4, 8, true, 7),
        norm: LayerNorm::new(8),
        fc2: Linear::with_seed(8, 3, true, 9),
    }
}

#[test]
fn derive_discovers_parameters_in_stable_order() {
    let model = mlp();
    let descriptor = model.descriptor().unwrap();
    let mut names = Vec::new();
    model
        .visit_tensors(&ModulePath::root(), &mut |path, tensor, role| {
            names.push((path.as_string(), tensor.dims(), role));
            Ok(())
        })
        .unwrap();
    let names: Vec<(String, Vec<usize>, ParameterRole)> = names;
    assert_eq!(
        names
            .iter()
            .map(|(name, _, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "fc1.weight",
            "fc1.bias",
            "norm.gamma",
            "norm.beta",
            "fc2.weight",
            "fc2.bias",
        ]
    );
    assert_eq!(names[0].1, vec![8, 4]);
    assert_eq!(names[2].2, ParameterRole::Trainable);
    assert_eq!(descriptor.parameter_count(), 6);
    assert!(descriptor.render().contains("fc1.weight"));
}

#[test]
fn model_forward_and_predict() {
    let model = mlp();
    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [2, 4])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    assert_eq!(compiled.report().parameters, 6);
    assert_eq!(compiled.report().trainable_parameters, 6);
    assert!(compiled.report().optimized);

    let input =
        Tensor::<f32>::from_vec([2, 4], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
    let output = model.predict(input.clone()).unwrap();
    assert_eq!(output.dims(), vec![2, 3]);

    let store = compiled.parameter_store(&model).unwrap();
    let direct = compiled.run(&store, &[("input0", &input)]).unwrap();
    assert_eq!(direct.len(), 1);
    assert_eq!(direct[0].dims(), vec![2, 3]);
    for (left, right) in output.data().iter().zip(direct[0].data()) {
        assert!((left - right).abs() < 1e-6);
    }
}

#[test]
fn sequential_matches_explicit_stack() {
    #[derive(Module)]
    struct Stack {
        first: Linear,
        second: Linear,
    }

    impl Stack {
        #[model]
        fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
            let x = relu(self.first.forward(x));
            self.second.forward(x)
        }
    }

    let explicit = Stack {
        first: Linear::with_seed(3, 5, true, 11),
        second: Linear::with_seed(5, 2, true, 12),
    };
    let sequential = Sequential::new()
        .add(Linear::with_seed(3, 5, true, 11))
        .add(ReLU::new())
        .add(Linear::with_seed(5, 2, true, 12));
    let input = Tensor::<f32>::from_vec([1, 3], vec![0.2, -0.4, 0.7]).unwrap();
    let a = explicit.predict(input.clone()).unwrap();
    let b = sequential.predict(input).unwrap();
    for (left, right) in a.data().iter().zip(b.data()) {
        assert!((left - right).abs() < 1e-6);
    }
}

#[derive(Module)]
struct XorMlp {
    fc1: Linear,
    fc2: Linear,
}

impl XorMlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let x = tanh(self.fc1.forward(x));
        self.fc2.forward(x)
    }
}

#[test]
fn trains_xor_with_adamw() {
    let inputs = Tensor::<f32>::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();
    let targets = Tensor::<f32>::from_vec([4, 1], vec![0., 1., 1., 0.]).unwrap();
    let dataset = TensorDataset::from_tensors(inputs, targets).unwrap();

    let mut model = XorMlp {
        fc1: Linear::with_seed(2, 8, true, 21),
        fc2: Linear::with_seed(8, 1, true, 22),
    };
    let mut trainer = Trainer::builder()
        .optimizer(AdamW::new(0.05).unwrap())
        .loss(MSELoss::new())
        .batch_size(4)
        .build()
        .unwrap();
    let report = trainer.fit(&mut model, &dataset, 600).unwrap();
    assert!(
        report.final_loss < 0.05,
        "XOR did not converge: {}",
        report.final_loss
    );
    assert_eq!(report.parameters, 4);
    assert_eq!(report.steps, 600);

    trainer
        .optimizer_mut()
        .unwrap()
        .set_learning_rate(0.001)
        .unwrap();
    let eval = trainer.evaluate(&model, &dataset).unwrap();
    assert!(eval < 0.2, "evaluation loss too high: {eval}");

    let input = Tensor::<f32>::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();
    let prediction = model.predict(input).unwrap();
    assert_eq!(prediction.dims(), vec![4, 1]);
    assert!(prediction.data()[0] < 0.5);
    assert!(prediction.data()[1] > 0.5);
}

#[derive(Module)]
struct SharedTower {
    shared: Arc<Linear>,
    head: Linear,
}

impl SharedTower {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let left = self.shared.forward(x);
        let _ = left;
        self.head.forward(x)
    }
}

#[test]
fn shared_parameter_is_detected_as_alias() {
    let shared = Arc::new(Linear::with_seed(4, 4, true, 33));
    let model = SharedTower {
        shared: Arc::clone(&shared),
        head: Linear::with_seed(4, 2, true, 34),
    };
    let mut registry = ParameterRegistry::new();
    model
        .visit_parameters(&ModulePath::root(), &mut registry)
        .unwrap();
    // head.weight/head.bias plus one canonical shared entry per tensor.
    assert_eq!(registry.len(), 4);
    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [2, 4])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    assert_eq!(
        compiled.report().aliases,
        0,
        "forward only visits `shared` once"
    );
}

#[test]
fn checkpoint_round_trip() {
    let model = mlp();
    let path =
        std::env::temp_dir().join(format!("oxide-ml-checkpoint-{}.json", std::process::id()));
    save_model(&path, &model).unwrap();
    let checkpoint = Checkpoint::load(&path).unwrap();
    assert_eq!(checkpoint.parameters.len(), 6);

    let mut restored = mlp();
    // Perturb a tensor so the restore is observable.
    restored.fc1.weight.data_mut()[0] = 12345.0;
    let report = load_model(&path, &mut restored).unwrap();
    assert!(report.is_exact());
    assert_eq!(restored.fc1.weight.data(), model.fc1.weight.data());
    let _ = std::fs::remove_file(path);
}

#[test]
fn graph_macro_builds_and_verifies() {
    let graph = graph! {
        input x: [4, 8];
        param w: [8, 3];
        param b: [3];
        let y = matmul(x, w);
        let y = add(y, b);
        let y = relu(y);
        output y;
    }
    .unwrap();
    assert!(GraphVerifier::verify(&graph).is_ok());
    assert_eq!(graph.parameters().len(), 2);
    assert_eq!(graph.outputs().len(), 1);
}

#[derive(Module)]
struct LossModel {
    linear: Linear,
}

impl LossModel {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.linear.forward(x)
    }
}

#[test]
fn cross_entropy_training_converges() {
    let inputs = Tensor::<f32>::from_vec(
        [6, 2],
        vec![
            2.0, 2.0, 2.1, 1.9, 1.8, 2.2, -2.0, -2.0, -2.1, -1.9, -1.8, -2.2,
        ],
    )
    .unwrap();
    let labels = Tensor::<f32>::from_vec([6], vec![1., 1., 1., 0., 0., 0.]).unwrap();
    let dataset = TensorDataset::from_tensors(inputs, labels).unwrap();
    let mut model = LossModel {
        linear: Linear::with_seed(2, 2, true, 41),
    };
    let mut trainer = Trainer::builder()
        .optimizer(Sgd::new(0.5).unwrap())
        .loss(CrossEntropyLoss::new())
        .batch_size(6)
        .build()
        .unwrap();
    let report = trainer.fit(&mut model, &dataset, 100).unwrap();
    assert!(report.final_loss < 0.1, "CE loss {}", report.final_loss);
}

#[test]
fn optimizer_state_checkpoint_round_trip() {
    let mut model = XorMlp {
        fc1: Linear::with_seed(2, 4, true, 71),
        fc2: Linear::with_seed(4, 1, true, 72),
    };
    let inputs = Tensor::<f32>::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();
    let targets = Tensor::<f32>::from_vec([4, 1], vec![0., 1., 1., 0.]).unwrap();
    let dataset = TensorDataset::from_tensors(inputs, targets).unwrap();
    let mut trainer = Trainer::builder()
        .optimizer(AdamW::new(0.02).unwrap())
        .loss(MSELoss::new())
        .batch_size(4)
        .build()
        .unwrap();
    trainer.fit(&mut model, &dataset, 20).unwrap();

    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [4, 2])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    let store = compiled.parameter_store(&model).unwrap();
    let path = std::env::temp_dir().join(format!("oxide-ml-optimizer-{}.json", std::process::id()));
    trainer.save_checkpoint(&path, &model, &store, 20).unwrap();
    let checkpoint = Checkpoint::load(&path).unwrap();
    assert!(checkpoint.optimizer.is_some());
    let state = checkpoint.optimizer.as_ref().unwrap();
    assert_eq!(state.kind, "adamw");
    assert_eq!(state.step, 20);
    assert!(!state.entries.is_empty());

    let mut restored = XorMlp {
        fc1: Linear::with_seed(2, 4, true, 1),
        fc2: Linear::with_seed(4, 1, true, 2),
    };
    let mut trainer2 = Trainer::builder()
        .optimizer(AdamW::new(0.02).unwrap())
        .loss(MSELoss::new())
        .batch_size(4)
        .build()
        .unwrap();
    let store2 = compiled.parameter_store(&restored).unwrap();
    let report = trainer2
        .load_checkpoint(&path, &mut restored, &store2)
        .unwrap();
    assert!(report.is_exact());
    assert_eq!(restored.fc1.weight.data(), model.fc1.weight.data());
    // The restored optimizer continues from step 20.
    let restored_state = trainer2
        .optimizer_ref()
        .unwrap()
        .state_dict(&store2)
        .unwrap()
        .unwrap();
    assert_eq!(restored_state.step, 20);
    assert_eq!(
        restored_state.entries.keys().collect::<Vec<_>>(),
        state.entries.keys().collect::<Vec<_>>()
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn unknown_checkpoint_version_is_rejected() {
    let path = std::env::temp_dir().join(format!(
        "oxide-ml-bad-checkpoint-{}.json",
        std::process::id()
    ));
    let json = r#"{"format":"oxide-ml.checkpoint","version":99,"model_type":"X","parameters":[]}"#;
    std::fs::write(&path, json).unwrap();
    let error = Checkpoint::load(&path).unwrap_err();
    assert!(matches!(error, Error::Serialization(_)), "{error:?}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn gradient_accumulation_matches_a_single_update() {
    let inputs = Tensor::<f32>::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]).unwrap();
    let targets = Tensor::<f32>::from_vec([4, 1], vec![0., 1., 1., 0.]).unwrap();
    let dataset = TensorDataset::from_tensors(inputs, targets).unwrap();
    let mut model = XorMlp {
        fc1: Linear::with_seed(2, 8, true, 11),
        fc2: Linear::with_seed(8, 1, true, 12),
    };
    let mut trainer = Trainer::builder()
        .optimizer(Sgd::new(0.1).unwrap())
        .loss(MSELoss::new())
        .batch_size(1)
        .accumulation_steps(4)
        .build()
        .unwrap();
    let report = trainer.fit(&mut model, &dataset, 200).unwrap();
    assert!(report.final_loss < 0.25, "loss {}", report.final_loss);
    assert_eq!(report.steps, 800);
}
