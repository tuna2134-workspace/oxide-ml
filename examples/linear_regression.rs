//! Fit y = 2x + 1 with a one-layer graph and AdamW.

use oxide_ml::legacy::Trainer;
use oxide_ml::prelude::*;

fn main() -> Result<()> {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([16, 1]), DType::F32);
    let target = builder.input("y", Shape::from_static([16, 1]), DType::F32);
    let weight = builder.try_parameter(
        "linear.weight",
        Shape::from_static([1, 1]),
        DType::F32,
        true,
    )?;
    let bias = builder.try_parameter("linear.bias", Shape::from_static([1]), DType::F32, true)?;
    let product = builder.matmul(x, weight)?;
    let prediction = builder.add(product, bias)?;
    let loss = builder.mse_loss(prediction, target, Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;
    let training = build_training_graph(&forward, loss)?;

    let mut parameters = init_parameters::<f32>(&forward, 7)?;
    parameters.get_mut(ParameterId(0))?.data = DenseTensor::from_vec([1, 1], vec![0.0])?;
    parameters.get_mut(ParameterId(1))?.data = DenseTensor::from_vec([1], vec![0.0])?;
    let optimizer = AdamW::with_config(0.08, 0.9, 0.999, 1e-8, 0.0)?;
    let mut trainer = Trainer::new(training, parameters, Box::new(optimizer))?;

    let xs: Vec<f32> = (0..16).map(|i| i as f32 * 0.25).collect();
    let ys: Vec<f32> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(forward.inputs()[0], DenseTensor::from_vec([16, 1], xs)?);
    inputs.insert(forward.inputs()[1], DenseTensor::from_vec([16, 1], ys)?);

    for step in 0..500 {
        let loss = trainer.step(&inputs)?;
        if step % 100 == 0 {
            println!("step {step:3}: loss = {loss:.6}");
        }
    }
    let result = trainer.run(&inputs)?;
    let prediction = result.output(0)?;
    println!(
        "learned y(3.0) = {:.4} (expected 7.0)",
        prediction.data()[12]
    );
    Ok(())
}
