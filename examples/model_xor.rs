//! Define a model with `#[derive(Module)]` / `#[model]`, train it with the
//! high-level `Trainer`, check the predictions, and round-trip a checkpoint.

use oxide_ml::prelude::*;

#[derive(Module)]
struct XorMlp {
    fc1: Linear,
    fc2: Linear,
}

impl XorMlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = tanh(self.fc1.forward(x));
        self.fc2.forward(hidden)
    }
}

fn main() -> Result<()> {
    let inputs = Tensor::<f32>::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.])?;
    let targets = Tensor::<f32>::from_vec([4, 1], vec![0., 1., 1., 0.])?;
    let dataset = TensorDataset::from_tensors(inputs, targets)?;

    let mut model = XorMlp {
        fc1: Linear::new(2, 8),
        fc2: Linear::new(8, 1),
    };
    println!("{}", model.descriptor()?.render());

    let mut trainer = Trainer::builder()
        .optimizer(AdamW::new(0.05)?)
        .loss(MSELoss::new())
        .batch_size(4)
        .scheduler(StepLr::new(400, 0.5)?)
        .build()?;
    let report = trainer.fit(&mut model, &dataset, 1200)?;

    for epoch in report.epochs.iter().step_by(300) {
        println!("epoch {:4}: loss = {:.6}", epoch.epoch, epoch.loss);
    }
    println!("final loss: {:.6}", report.final_loss);

    let input = Tensor::<f32>::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.])?;
    let prediction = model.predict(input)?;
    let expected = [0.0, 1.0, 1.0, 0.0];
    for (index, value) in prediction.data().iter().enumerate() {
        println!("case {index}: {value:.4} (expected {:.1})", expected[index]);
    }

    let path = std::env::temp_dir().join("oxide-ml-xor-checkpoint.json");
    save_model(&path, &model)?;
    let mut restored = XorMlp {
        fc1: Linear::new(2, 8),
        fc2: Linear::new(8, 1),
    };
    let load = load_model(&path, &mut restored)?;
    println!(
        "checkpoint: loaded {} tensors, exact = {}",
        load.loaded,
        load.is_exact()
    );
    let _ = std::fs::remove_file(path);
    Ok(())
}
