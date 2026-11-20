//! BatchNorm running-stat calibration: probes, updates and eval consistency.

use oxide_ml::prelude::*;

#[derive(Module)]
struct NormNet {
    bn: BatchNorm,
}

impl NormNet {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.bn.forward(x)
    }
}

fn model() -> NormNet {
    NormNet {
        bn: BatchNorm::new(3),
    }
}

fn batch(values: &[f32], rows: usize) -> Tensor<f32> {
    Tensor::<f32>::from_vec([rows, 3, 2, 2], values.to_vec()).unwrap()
}

#[test]
fn calibration_writes_batch_statistics_into_running_buffers() {
    let mut model = model();
    let values: Vec<f32> = (0..4 * 3 * 2 * 2)
        .map(|index| {
            let channel = (index / 4) % 3;
            channel as f32 * 2.0 + (index % 4) as f32 * 0.5
        })
        .collect();
    let input = batch(&values, 4);
    let report = calibrate_batch_norm(
        &mut model,
        &[InputSpec::unbatched("input0", [3, 2, 2])],
        &[vec![input.clone()]],
        1.0,
    )
    .unwrap();
    assert_eq!(report.layers, 1, "{report:?}");
    assert_eq!(report.samples, 4);

    // Expected per-channel mean over N*H*W.
    let mut expected_mean = [0.0f32; 3];
    for (index, value) in values.iter().enumerate() {
        let channel = (index / 4) % 3;
        expected_mean[channel] += value / 16.0;
    }
    for (actual, expected) in model
        .bn
        .running_mean
        .as_ref()
        .unwrap()
        .data()
        .iter()
        .zip(expected_mean)
    {
        assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
    }

    // Eval on individual samples must match the batch-normalized training
    // output because momentum=1 copied the exact batch statistics.
    let training = oxide_ml::predict_single(&model, input.clone()).unwrap();
    model.set_training(false);
    for row in 0..4 {
        let sample = Tensor::<f32>::from_vec(
            [1, 3, 2, 2],
            input.data()[row * 12..(row + 1) * 12].to_vec(),
        )
        .unwrap();
        let evaluated = oxide_ml::predict_single(&model, sample).unwrap();
        for (actual, expected) in evaluated
            .data()
            .iter()
            .zip(&training.data()[row * 12..(row + 1) * 12])
        {
            assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }
    }
}

#[test]
fn calibration_rejects_invalid_momentum_and_empty_batches() {
    let mut model = model();
    let input = batch(&[0.0; 48], 4);
    assert!(
        calibrate_batch_norm(
            &mut model,
            &[InputSpec::unbatched("input0", [3, 2, 2])],
            &[vec![input.clone()]],
            1.5
        )
        .is_err()
    );
    assert!(
        calibrate_batch_norm(
            &mut model,
            &[InputSpec::unbatched("input0", [3, 2, 2])],
            &[],
            0.1
        )
        .is_err()
    );
}
