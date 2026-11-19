//! Train a small MLP on XOR with reverse-mode autograd.

use oxide_ml::legacy::{Mlp, Trainer};
use oxide_ml::prelude::*;

fn main() -> Result<()> {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 2]), DType::F32);
    let target = builder.input("y", Shape::from_static([4, 1]), DType::F32);
    let mlp = Mlp::new(&mut builder, "mlp", &[2, 8, 1], ActivationKind::Tanh, true)?;
    let logits = mlp.forward(&mut builder, x)?;
    let prediction = builder.sigmoid(logits)?;
    let loss = builder.mse_loss(prediction, target, Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;
    let training = build_training_graph(&forward, loss)?;
    let parameters = init_parameters::<f32>(&forward, 42)?;
    let optimizer = AdamW::with_config(0.08, 0.9, 0.999, 1e-8, 0.0)?;
    let mut trainer = Trainer::new(training, parameters, Box::new(optimizer))?;

    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.])?,
    );
    inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([4, 1], vec![0., 1., 1., 0.])?,
    );

    for step in 0..2500 {
        let loss = trainer.step(&inputs)?;
        if step % 500 == 0 {
            println!("step {step:4}: loss = {loss:.6}");
        }
    }
    let result = trainer.run(&inputs)?;
    println!("\ninput -> prediction (target)");
    let predictions = result.output(0)?.data().to_vec();
    let targets = [0.0, 1.0, 1.0, 0.0];
    let cases = [(0., 0.), (0., 1.), (1., 0.), (1., 1.)];
    for (index, ((a, b), target)) in cases.iter().zip(targets).enumerate() {
        println!("{a} xor {b} -> {:.4} ({target})", predictions[index]);
    }
    Ok(())
}
