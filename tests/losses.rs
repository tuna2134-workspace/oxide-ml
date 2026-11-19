//! Numerical checks for the model-API loss functions.

use oxide_ml::prelude::*;

#[derive(Module)]
struct Identity {
    unit: (),
}

impl Identity {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        x
    }
}

fn evaluate(loss: impl Loss + 'static, input: Tensor<f32>, target: Tensor<f32>) -> f32 {
    let dataset = TensorDataset::from_tensors(input, target).unwrap();
    let model = Identity { unit: () };
    let trainer = Trainer::new().loss(loss);
    trainer.evaluate(&model, &dataset).unwrap()
}

#[test]
fn mse_matches_hand_computation() {
    let input = Tensor::<f32>::from_vec([1, 2], vec![0.0, 0.0]).unwrap();
    let target = Tensor::<f32>::from_vec([1, 2], vec![1.0, 3.0]).unwrap();
    let loss = evaluate(MSELoss::new(), input, target);
    assert!((loss - 5.0).abs() < 1e-5, "{loss}");
}

#[test]
fn cross_entropy_of_uniform_logits_is_log_classes() {
    let input = Tensor::<f32>::from_vec([1, 3], vec![0.0, 0.0, 0.0]).unwrap();
    let target = Tensor::<f32>::from_vec([1], vec![1.0]).unwrap();
    let loss = evaluate(CrossEntropyLoss::new(), input, target);
    assert!((loss - 3.0f32.ln()).abs() < 1e-5, "{loss}");
}

#[test]
fn bce_with_logits_matches_softplus_value() {
    let input = Tensor::<f32>::from_vec([1, 2], vec![0.0, 0.0]).unwrap();
    let target = Tensor::<f32>::from_vec([1, 2], vec![0.0, 1.0]).unwrap();
    let loss = evaluate(BCEWithLogitsLoss::new(), input, target);
    assert!((loss - 2.0f32.ln()).abs() < 1e-5, "{loss}");
}

#[test]
fn huber_matches_hand_computation() {
    let input = Tensor::<f32>::from_vec([1, 2], vec![0.0, 0.0]).unwrap();
    let target = Tensor::<f32>::from_vec([1, 2], vec![2.0, 0.0]).unwrap();
    let loss = evaluate(HuberLoss::new(1.0).unwrap(), input, target);
    assert!((loss - 0.75).abs() < 1e-5, "{loss}");
}

#[test]
fn nll_with_one_hot_targets_matches_log_softmax() {
    let input = Tensor::<f32>::from_vec([1, 2], vec![0.0, 0.0]).unwrap();
    let target = Tensor::<f32>::from_vec([1, 2], vec![0.0, 1.0]).unwrap();
    let loss = evaluate(NLLLoss::new(), input, target);
    assert!((loss - 2.0f32.ln()).abs() < 1e-5, "{loss}");
}

#[test]
fn huber_rejects_non_positive_delta() {
    assert!(HuberLoss::new(0.0).is_err());
    assert!(HuberLoss::new(-1.0).is_err());
}
