//! CPU/CUDA parity for MobileNetV4 forward and one training step.

use oxide_ml_core::DenseTensor;
use oxide_ml_model::{
    CompiledModel, InputSpec, ModelCompileOptions, Module, calibrate_batch_norm, predict_single,
};
use oxide_ml_nn::vision::{MobileNetV4, MobileNetV4Config};
use oxide_ml_nn::CrossEntropyLoss;

use oxide_ml_cuda::{AdamWParams, CudaAdamW, CudaDevice, CudaModel, CudaTrainingModel};

fn input(batch: usize, size: usize, seed: f32) -> DenseTensor<f32> {
    let values = (0..batch * 3 * size * size)
        .map(|index| ((index as f32 * 0.013 + seed) % 1.0) - 0.5)
        .collect();
    DenseTensor::from_vec([batch, 3, size, size], values).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size = 32;
    let config = MobileNetV4Config::nano(2);
    let mut model = MobileNetV4::new(&config)?;
    model.set_training(false);
    let x = input(1, size, 0.1);
    let cpu_logits = predict_single(&model, oxide_ml_model::Tensor::from_dense(x.clone()))?;

    let device = CudaDevice::new(0)?;
    let gpu_model = CudaModel::compile(
        device.clone(),
        &model,
        &[InputSpec::unbatched("input0", [1, 3, size, size])],
        ModelCompileOptions::default(),
    )?;
    let gpu_logits = gpu_model.run(&[("input0", x.clone())])?;
    let mut max_diff = 0.0f32;
    for (a, b) in cpu_logits.data().iter().zip(gpu_logits[0].data()) {
        max_diff = max_diff.max((a - b).abs());
    }
    println!("MobileNetV4 forward max |diff| = {max_diff:.3e}");
    if max_diff > 1e-2 {
        return Err(format!("forward mismatch {max_diff}").into());
    }

    // Training-mode forward: batch-normalization statistics are computed from
    // the batch, which exercises the ReduceMean path used by training.
    let train_options = ModelCompileOptions {
        mode: oxide_ml_model::ModuleMode::Train,
        ..ModelCompileOptions::default()
    };
    let train_batch = 2;
    let train_images = input(train_batch, size, 0.3);
    let cpu_fwd = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched(
            "input0",
            [train_batch, 3, size, size],
        )],
        train_options.clone(),
    )?;
    let cpu_store = cpu_fwd.parameter_store(&model)?;
    let cpu_train_logits = cpu_fwd.run(
        &cpu_store,
        &[("input0", &oxide_ml_model::Tensor::from_dense(train_images.clone()))],
    )?;
    let gpu_fwd = CudaModel::compile(
        device.clone(),
        &model,
        &[InputSpec::unbatched(
            "input0",
            [train_batch, 3, size, size],
        )],
        train_options,
    )?;
    let gpu_train_logits = gpu_fwd.run(&[("input0", train_images.clone())])?;
    let mut train_diff = 0.0f32;
    for (a, b) in cpu_train_logits[0].data().iter().zip(gpu_train_logits[0].data()) {
        train_diff = train_diff.max((a - b).abs());
    }
    println!("training-mode forward max |diff| = {train_diff:.3e}");

    // Isolate BatchNorm training statistics from the convolution stack.
    #[derive(oxide_ml_macros::Module)]
    #[oxide(crate = ::oxide_ml_model)]
    struct BnOnly {
        bn: oxide_ml_nn::BatchNorm,
    }
    impl BnOnly {
        #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
        fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
            self.bn.forward(x)
        }
    }
    let bn_model = BnOnly {
        bn: oxide_ml_nn::BatchNorm::new(3),
    };
    let bn_x = input(2, 4, 0.2);
    let bn_x = DenseTensor::from_vec(
        [2, 3, 4, 4],
        bn_x.data()[..2 * 3 * 4 * 4].to_vec(),
    )?;
    let bn_cpu = CompiledModel::compile(
        &bn_model,
        &[InputSpec::unbatched("input0", [2, 3, 4, 4])],
        ModelCompileOptions {
            mode: oxide_ml_model::ModuleMode::Train,
            ..ModelCompileOptions::default()
        },
    )?;
    let bn_store = bn_cpu.parameter_store(&bn_model)?;
    let bn_cpu_out = bn_cpu.run(
        &bn_store,
        &[("input0", &oxide_ml_model::Tensor::from_dense(bn_x.clone()))],
    )?;
    let bn_gpu = CudaModel::compile(
        device.clone(),
        &bn_model,
        &[InputSpec::unbatched("input0", [2, 3, 4, 4])],
        ModelCompileOptions {
            mode: oxide_ml_model::ModuleMode::Train,
            ..ModelCompileOptions::default()
        },
    )?;
    let bn_gpu_out = bn_gpu.run(&[("input0", bn_x)])?;
    let mut bn_diff = 0.0f32;
    for (a, b) in bn_cpu_out[0].data().iter().zip(bn_gpu_out[0].data()) {
        bn_diff = bn_diff.max((a - b).abs());
    }
    println!("BatchNorm train-mode max |diff| = {bn_diff:.3e}");

    // One training step: compare the loss with the CPU training graph.
    let batch = 2;
    let images = input(batch, size, 0.7);
    let targets = DenseTensor::from_vec([batch], vec![0.0, 1.0])?;
    let mut cpu_model = MobileNetV4::new(&config)?;
    let cpu_compiled = CompiledModel::compile_training(
        &cpu_model,
        &[InputSpec::unbatched("input0", [batch, 3, size, size])],
        InputSpec::unbatched("target0", [batch]),
        &CrossEntropyLoss::new(),
        ModelCompileOptions::training(),
    )?;
    let cpu_store = cpu_compiled.parameter_store(&cpu_model)?;
    let cpu_images = oxide_ml_model::Tensor::from_dense(images.clone());
    let cpu_targets = oxide_ml_model::Tensor::from_dense(targets.clone());
    let cpu_output = cpu_compiled.run_training(
        &cpu_store,
        &[("input0", &cpu_images), ("target0", &cpu_targets)],
    )?;

    let mut gpu_train = CudaTrainingModel::new(
        device,
        &mut cpu_model,
        &[InputSpec::unbatched("input0", [batch, 3, size, size])],
        InputSpec::unbatched("target0", [batch]),
        &CrossEntropyLoss::new(),
        Box::new(CudaAdamW::new(AdamWParams {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        })?),
    )?;
    let gpu_loss = gpu_train.step(&[("input0", images.clone())], &targets)?;
    println!(
        "training loss cpu {:.6} gpu {:.6} diff {:.3e}",
        cpu_output.loss,
        gpu_loss,
        (cpu_output.loss - gpu_loss).abs()
    );
    if (cpu_output.loss - gpu_loss).abs() > 2e-2 {
        return Err("training loss mismatch".into());
    }

    // BatchNorm calibration on the host after CUDA training.
    let report = calibrate_batch_norm(
        &mut cpu_model,
        &[InputSpec::unbatched("input0", [3, size, size])],
        &[vec![oxide_ml_model::Tensor::from_dense(images.clone())]],
        0.3,
    )?;
    println!("calibration layers {} samples {}", report.layers, report.samples);
    println!("VISION_CHECK: PASS");
    Ok(())
}
