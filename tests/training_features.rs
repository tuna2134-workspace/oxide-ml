//! Gradient clipping, classification metrics and early stopping.

use oxide_ml::prelude::*;

#[derive(Module)]
struct Classifier {
    linear: Linear,
}

impl Classifier {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.linear.forward(x)
    }
}

#[derive(Module)]
struct Regressor {
    linear: Linear,
}

impl Regressor {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.linear.forward(x)
    }
}

fn classification_dataset() -> TensorDataset {
    let inputs = Tensor::<f32>::from_vec(
        [6, 2],
        vec![
            2.0, 2.0, 2.1, 1.9, 1.8, 2.2, -2.0, -2.0, -2.1, -1.9, -1.8, -2.2,
        ],
    )
    .unwrap();
    let labels = Tensor::<f32>::from_vec([6], vec![1., 1., 1., 0., 0., 0.]).unwrap();
    TensorDataset::from_tensors(inputs, labels).unwrap()
}

#[test]
fn early_stopping_state_machine() {
    let mut early = EarlyStopping::new(EarlyStoppingConfig {
        patience: 2,
        min_delta: 0.0,
    });
    assert!(!early.update(1.0));
    assert!(!early.update(0.9));
    assert!(!early.update(0.95));
    assert!(early.update(0.96));
    assert!(early.stopped());
    assert_eq!(early.best(), 0.9);
}

#[test]
fn clip_gradients_scales_global_norm() {
    let mut gradients = Gradients::new();
    gradients.insert(
        ParameterId(0),
        DenseTensor::from_vec([2], vec![3.0, 4.0]).unwrap(),
    );
    let norm = clip_gradients(&mut gradients, 1.0).unwrap();
    assert!((norm - 5.0).abs() < 1e-5);
    let clipped = gradients.get(&ParameterId(0)).unwrap();
    let clipped_norm = (clipped.data()[0].powi(2) + clipped.data()[1].powi(2)).sqrt();
    assert!((clipped_norm - 1.0).abs() < 1e-5);
    assert!(clip_gradients(&mut gradients, 0.0).is_err());
}

#[test]
fn classification_accuracy_reads_logits() {
    let logits = Tensor::<f32>::from_vec([2, 3], vec![0.1, 0.9, 0.0, 0.8, 0.1, 0.1]).unwrap();
    let targets = Tensor::<f32>::from_vec([2], vec![1.0, 0.0]).unwrap();
    assert_eq!(classification_accuracy(&logits, &targets), Some(1.0));
    let targets = Tensor::<f32>::from_vec([2], vec![0.0, 0.0]).unwrap();
    assert_eq!(classification_accuracy(&logits, &targets), Some(0.5));
    let wrong_shape = Tensor::<f32>::from_vec([2, 3, 1], vec![0.0; 6]).unwrap();
    assert_eq!(classification_accuracy(&wrong_shape, &targets), None);
}

#[test]
fn trainer_reports_accuracy_and_clips_gradients() {
    let dataset = classification_dataset();
    let mut model = Classifier {
        linear: Linear::with_seed(2, 2, true, 11),
    };
    let mut trainer = Trainer::builder()
        .optimizer(Sgd::new(0.5).unwrap())
        .loss(CrossEntropyLoss::new())
        .batch_size(6)
        .metrics(true)
        .clip_grad_norm(1.0)
        .build()
        .unwrap();
    let report = trainer.fit(&mut model, &dataset, 120).unwrap();
    assert!(report.final_loss < 0.1, "loss {}", report.final_loss);
    let accuracy = report.final_accuracy.expect("metrics enabled");
    assert!(accuracy >= 5.0 / 6.0, "accuracy {accuracy}");
    assert!(report.final_grad_norm.is_some());
    assert!(report.epochs.last().unwrap().accuracy.is_some());
}

#[test]
fn trainer_stops_early() {
    let inputs = Tensor::<f32>::from_vec([4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let targets = Tensor::<f32>::from_vec([4, 1], vec![0.0, 1.0, 2.0, 3.0]).unwrap();
    let dataset = TensorDataset::from_tensors(inputs, targets).unwrap();
    let mut model = Regressor {
        linear: Linear::with_seed(1, 1, true, 3),
    };
    let mut trainer = Trainer::builder()
        .optimizer(Sgd::new(0.001).unwrap())
        .loss(MSELoss::new())
        .batch_size(4)
        .early_stopping(1, 10.0)
        .build()
        .unwrap();
    let report = trainer.fit(&mut model, &dataset, 100).unwrap();
    assert!(report.stopped_early);
    assert_eq!(
        report.epochs.len(),
        2,
        "one epoch to set the baseline, one miss"
    );
}
