//! End-to-end check of the model API on CUDA: the same module definition is
//! compiled for the CPU reference and for the T4, and both outputs are compared.

use oxide_ml_model::{CompiledModel, InputSpec, ModelCompileOptions, Optimizer, Tensor};
use oxide_ml_nn::MSELoss;
use oxide_ml_nn::modules::Linear;
use oxide_ml_optim::Sgd;

use oxide_ml_cuda::{CudaDevice, CudaError, CudaModel, CudaSgd, CudaTrainingModel};
use oxide_ml_macros::{Module as ModuleDerive, model};

#[derive(ModuleDerive)]
#[oxide(crate = ::oxide_ml_model)]
struct TinyNet {
    fc1: Linear,
    fc2: Linear,
}

impl TinyNet {
    #[model(crate = ::oxide_ml_model)]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = tanh(self.fc1.forward(x));
        self.fc2.forward(hidden)
    }
}

fn inputs() -> oxide_ml_model::Result<Tensor<f32>> {
    Tensor::<f32>::from_vec([4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
}

fn targets() -> oxide_ml_model::Result<Tensor<f32>> {
    Tensor::<f32>::from_vec([4, 1], vec![0.0, 1.0, 1.0, 0.0])
}

fn main() -> oxide_ml_cuda::Result<()> {
    let mut failures = 0usize;
    let mut check = |name: &str, ok: bool, detail: String| {
        println!("[{}] {name}: {detail}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            failures += 1;
        }
    };

    let model = TinyNet {
        fc1: Linear::with_seed(2, 8, true, 41),
        fc2: Linear::with_seed(8, 1, true, 42),
    };
    let input = inputs()?;
    let target = targets()?;
    let specs = [InputSpec::unbatched("input0", [4, 2])];

    let cpu = CompiledModel::compile(&model, &specs, ModelCompileOptions::default())?;
    let cpu_store = cpu.parameter_store(&model)?;
    let cpu_out = cpu.run(&cpu_store, &[("input0", &input)])?;

    let gpu = CudaModel::compile(
        CudaDevice::new(0)?,
        &model,
        &specs,
        ModelCompileOptions::default(),
    )?;
    let gpu_input = input.dense().clone();
    let gpu_out = gpu.run(&[("input0", gpu_input)])?;

    let mut max_diff = 0.0f32;
    for (a, b) in cpu_out[0].data().iter().zip(gpu_out[0].data()) {
        max_diff = max_diff.max((a - b).abs());
    }
    check(
        "CPU/CUDA forward parity",
        max_diff < 1e-4,
        format!("max |diff| = {max_diff:.3e}"),
    );

    // Training parity: 20 identical SGD steps from identical weights.
    let mut cpu_model = TinyNet {
        fc1: Linear::with_seed(2, 8, true, 41),
        fc2: Linear::with_seed(8, 1, true, 42),
    };
    let cpu_training = CompiledModel::compile_training(
        &cpu_model,
        &[InputSpec::unbatched("input0", [4, 2])],
        InputSpec::unbatched("target0", [4, 1]),
        &MSELoss::new(),
        ModelCompileOptions::training(),
    )?;
    let mut cpu_store = cpu_training.parameter_store(&cpu_model)?;
    let mut cpu_optimizer = Sgd::new(0.05)?;
    let mut cpu_losses = Vec::new();
    for _ in 0..20 {
        let output =
            cpu_training.run_training(&cpu_store, &[("input0", &input), ("target0", &target)])?;
        cpu_losses.push(output.loss);
        cpu_optimizer.step(&mut cpu_store, &output.gradients)?;
    }
    cpu_training.write_back(&mut cpu_model, &cpu_store)?;

    let mut gpu_model = TinyNet {
        fc1: Linear::with_seed(2, 8, true, 41),
        fc2: Linear::with_seed(8, 1, true, 42),
    };
    let mut gpu_training = CudaTrainingModel::new(
        CudaDevice::new(0)?,
        &gpu_model,
        &[InputSpec::unbatched("input0", [4, 2])],
        InputSpec::unbatched("target0", [4, 1]),
        &MSELoss::new(),
        Box::new(CudaSgd::new(0.05)?),
    )?;
    let mut gpu_losses = Vec::new();
    for _ in 0..20 {
        gpu_losses.push(gpu_training.step(
            &[("input0", input.dense().clone())],
            &target.dense().clone(),
        )?);
    }
    gpu_training.sync_to_model(&mut gpu_model)?;

    let cpu_last = *cpu_losses.last().unwrap();
    let gpu_last = *gpu_losses.last().unwrap();
    let loss_diff = (cpu_last - gpu_last).abs();
    check(
        "CPU/CUDA training loss parity (20 SGD steps)",
        loss_diff < 1e-3,
        format!("cpu {cpu_last:.6}, cuda {gpu_last:.6}, |diff| {loss_diff:.3e}"),
    );

    let cpu_after = cpu_model.predict(inputs()?)?;
    let gpu_after = gpu_model.predict(inputs()?)?;
    let mut step_diff = 0.0f32;
    for (a, b) in cpu_after.data().iter().zip(gpu_after.data()) {
        step_diff = step_diff.max((a - b).abs());
    }
    check(
        "post-training parameter parity",
        step_diff < 5e-3,
        format!("max |diff| = {step_diff:.3e}"),
    );

    if failures == 0 {
        println!("MODEL_CHECK: PASS");
        Ok(())
    } else {
        println!("MODEL_CHECK: FAIL ({failures})");
        Err(CudaError::Execution(format!(
            "{failures} model checks failed"
        )))
    }
}
