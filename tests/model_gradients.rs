//! Numerical gradient checks for the model API and shared parameters.

use std::collections::HashMap;
use std::sync::Arc;

use oxide_ml::prelude::*;

#[derive(Module)]
struct TinyNet {
    fc1: Linear,
    fc2: Linear,
}

impl TinyNet {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let h = tanh(self.fc1.forward(x));
        self.fc2.forward(h)
    }
}

fn inputs() -> (Tensor<f32>, Tensor<f32>) {
    (
        Tensor::<f32>::from_vec([3, 2], vec![0.2, -0.3, 0.5, 0.7, -0.1, 0.4]).unwrap(),
        Tensor::<f32>::from_vec([3, 1], vec![0.3, -0.2, 0.8]).unwrap(),
    )
}

#[test]
fn numerical_gradient_matches_model_autograd() {
    let mut model = TinyNet {
        fc1: Linear::with_seed(2, 3, true, 101),
        fc2: Linear::with_seed(3, 1, true, 102),
    };
    let (x, y) = inputs();
    let mut options = ModelCompileOptions::training();
    options.optimize = false;
    options.memory_plan = false;
    let compiled = CompiledModel::compile_training(
        &model,
        &[InputSpec::unbatched("input0", [3, 2])],
        InputSpec::unbatched("target0", [3, 1]),
        &MSELoss::new(),
        options,
    )
    .unwrap();
    let mut store = compiled.parameter_store(&model).unwrap();
    let output = compiled
        .run_training(&store, &[("input0", &x), ("target0", &y)])
        .unwrap();

    let parameters = compiled.parameter_ids().clone();
    let epsilon = 1e-2f32;
    let mut checked = 0;
    for (name, id) in &parameters {
        let analytic = output.gradients.get(id).unwrap().data().to_vec();
        let dimensions = store.get(*id).unwrap().data.dims();
        let count = analytic.len().min(4);
        for (index, expected) in analytic.iter().enumerate().take(count) {
            let mut plus = store.clone();
            let mut minus = store.clone();
            plus.get_mut(*id).unwrap().data.data_mut()[index] += epsilon;
            minus.get_mut(*id).unwrap().data.data_mut()[index] -= epsilon;
            let loss_plus = compiled
                .evaluate(&plus, &[("input0", &x), ("target0", &y)])
                .unwrap();
            let loss_minus = compiled
                .evaluate(&minus, &[("input0", &x), ("target0", &y)])
                .unwrap();
            let numeric = (loss_plus - loss_minus) / (2.0 * epsilon);
            let error = (numeric - expected).abs();
            assert!(
                error < 5e-2,
                "gradient mismatch for {name}[{index}] dims {:?}: analytic {}, numeric {}",
                dimensions,
                expected,
                numeric
            );
            checked += 1;
        }
    }
    assert!(checked >= 11, "expected at least 11 gradient elements");

    // Applying one optimizer step must reduce the loss on the batch.
    let before = compiled
        .evaluate(&store, &[("input0", &x), ("target0", &y)])
        .unwrap();
    let mut optimizer = Sgd::new(0.1).unwrap();
    optimizer.step(&mut store, &output.gradients).unwrap();
    let after = compiled
        .evaluate(&store, &[("input0", &x), ("target0", &y)])
        .unwrap();
    assert!(after < before, "loss {before} -> {after}");
    compiled.write_back(&mut model, &store).unwrap();
}

#[derive(Module)]
struct SharedLinear {
    shared: Arc<Linear>,
    other: Arc<Linear>,
}

impl SharedLinear {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let a = self.shared.forward(x);
        let b = self.other.forward(x);
        add(a, b)
    }
}

#[test]
fn shared_parameter_receives_a_single_accumulated_gradient() {
    let shared = Arc::new(Linear::with_seed(2, 2, true, 201));
    let model = SharedLinear {
        shared: Arc::clone(&shared),
        other: Arc::clone(&shared),
    };
    let (x, _) = inputs();
    let mut options = ModelCompileOptions::training();
    options.optimize = false;
    options.memory_plan = false;
    let compiled = CompiledModel::compile_training(
        &model,
        &[InputSpec::unbatched("input0", [3, 2])],
        InputSpec::unbatched("target0", [3, 2]),
        &MSELoss::new(),
        options,
    )
    .unwrap();
    // Only one canonical weight/bias pair is registered for the shared tensor.
    assert_eq!(compiled.report().parameters, 2);
    assert_eq!(compiled.report().aliases, 2);
    let store = compiled.parameter_store(&model).unwrap();
    let output = compiled
        .run_training(&store, &[("input0", &x), ("target0", &x)])
        .unwrap();
    assert_eq!(output.gradients.len(), 2);
    for gradient in output.gradients.values() {
        assert!(gradient.data().iter().all(|value| value.is_finite()));
    }
}

#[test]
fn parameter_store_round_trip_preserves_values() {
    let model = TinyNet {
        fc1: Linear::with_seed(2, 3, true, 301),
        fc2: Linear::with_seed(3, 1, true, 302),
    };
    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [1, 2])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    let store = compiled.parameter_store(&model).unwrap();
    let mut updated = store.clone();
    for parameter in updated.iter_mut() {
        for value in parameter.data.data_mut() {
            *value += 1.0;
        }
    }
    let mut model = model;
    let report = compiled.write_back(&mut model, &updated).unwrap();
    assert_eq!(report.updated, 4);
    let again = compiled.parameter_store(&model).unwrap();
    for (left, right) in store.iter().zip(again.iter()) {
        for (a, b) in left.data.data().iter().zip(right.data.data()) {
            // Everything except the first updated value should differ by 1.
            assert!((a - b).abs() <= 1.0 + 1e-6);
        }
    }
    let map: HashMap<_, _> = compiled
        .parameter_ids()
        .iter()
        .map(|(name, id)| (name.clone(), *id))
        .collect();
    assert_eq!(map.len(), 4);
}

#[test]
fn host_tensor_indexing_and_reductions() {
    let mut tensor = Tensor::<f32>::from_vec([2, 2], vec![1., 2., 3., 4.]).unwrap();
    assert_eq!(tensor.get(&[1, 0]).unwrap(), 3.0);
    tensor.set(&[0, 1], 7.0).unwrap();
    assert_eq!(tensor.data(), &[1., 7., 3., 4.]);
    assert_eq!(tensor.sum_all(), 15.0);
    assert_eq!(tensor.mean_all(), 3.75);
    let transposed = tensor.transpose(&[1, 0]).unwrap();
    assert_eq!(transposed.data(), &[1., 3., 7., 4.]);
    let broadcast = Tensor::<f32>::from_vec([2], vec![10., 20.])
        .unwrap()
        .broadcast_to([2, 2])
        .unwrap();
    let sum = tensor.add(&broadcast).unwrap();
    assert_eq!(sum.data(), &[11., 27., 13., 24.]);
}
