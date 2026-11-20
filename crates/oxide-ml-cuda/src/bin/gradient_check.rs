//! CPU/CUDA gradient parity for convolution, BatchNorm and CE.

use oxide_ml_core::DenseTensor;
use oxide_ml_model::{
    CompiledModel, InputSpec, ModelCompileOptions, Module, ModulePath,
};
use oxide_ml_nn::vision::{ConvNormAct, GlobalAvgPool};
use oxide_ml_nn::{BatchNorm, Conv2d, ConvActivation, CrossEntropyLoss, Linear};
use oxide_ml_optim::{AdamW, Optimizer};

use oxide_ml_cuda::{AdamWParams, CudaAdamW, CudaDevice, CudaTrainingModel};

#[derive(Clone, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
struct TinyConv {
    conv: ConvNormAct,
    depthwise: ConvNormAct,
    pool: GlobalAvgPool,
    fc: Linear,
}

impl TinyConv {
    #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
    fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
        let h = self.conv.forward(x);
        let h = self.depthwise.forward(h);
        let h = self.pool.forward(h);
        self.fc.forward(h)
    }
}

#[derive(Clone, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
struct ConvOnly {
    conv: Conv2d,
    pool: GlobalAvgPool,
    fc: Linear,
}

impl ConvOnly {
    #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
    fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
        let h = self.conv.forward(x);
        let h = self.pool.forward(h);
        self.fc.forward(h)
    }
}

#[derive(Clone, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
struct LinearOnly {
    fc: Linear,
}

impl LinearOnly {
    #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
    fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
        self.fc.forward(x)
    }
}

#[derive(Clone, oxide_ml_macros::Module)]
#[oxide(crate = ::oxide_ml_model)]
struct BnOnly {
    bn: BatchNorm,
    pool: GlobalAvgPool,
    fc: Linear,
}

impl BnOnly {
    #[oxide_ml_macros::model(crate = ::oxide_ml_model)]
    fn forward(&self, x: oxide_ml_model::Tensor<f32>) -> oxide_ml_model::Tensor<f32> {
        let h = self.bn.forward(x);
        let h = self.pool.forward(h);
        self.fc.forward(h)
    }
}

fn input(batch: usize) -> DenseTensor<f32> {
    let values = (0..batch * 3 * 8 * 8)
        .map(|index| ((index as f32 * 0.11) % 1.0) - 0.5)
        .collect();
    DenseTensor::from_vec([batch, 3, 8, 8], values).unwrap()
}

fn compare<M: Module + Clone>(name: &str, model: M) -> Result<f32, Box<dyn std::error::Error>> {
    let batch = 2;
    let images = input(batch);
    let targets = DenseTensor::from_vec([batch], vec![0.0, 1.0])?;
    let mut cpu_model = model.clone();
    let compiled = CompiledModel::compile_training(
        &cpu_model,
        &[InputSpec::unbatched("input0", [batch, 3, 8, 8])],
        InputSpec::unbatched("target0", [batch]),
        &CrossEntropyLoss::new(),
        ModelCompileOptions::training(),
    )?;
    let mut store = compiled.parameter_store(&cpu_model)?;
    let cpu_images = oxide_ml_model::Tensor::from_dense(images.clone());
    let cpu_targets = oxide_ml_model::Tensor::from_dense(targets.clone());
    let out = compiled.run_training(
        &store,
        &[("input0", &cpu_images), ("target0", &cpu_targets)],
    )?;
    let mut optimizer = AdamW::new(1e-3)?;
    optimizer.step(&mut store, &out.gradients)?;
    compiled.write_back(&mut cpu_model, &store)?;

    let mut gpu_model = model;
    let mut training = CudaTrainingModel::new(
        CudaDevice::new(0)?,
        &gpu_model,
        &[InputSpec::unbatched("input0", [batch, 3, 8, 8])],
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
    let gpu_loss = training.step(&[("input0", images.clone())], &targets)?;
    let host_after = training.host_parameters()?;
    let mut initial_tensors = std::collections::HashMap::new();
    gpu_model.visit_tensors(&ModulePath::root(), &mut |path, tensor, _role| {
        initial_tensors.insert(path.as_string(), tensor.clone());
        Ok(())
    })?;
    let mut pre_sync = 0.0f32;
    let mut post_sync = 0.0f32;
    training.sync_to_model(&mut gpu_model)?;
    let mut synced_tensors = std::collections::HashMap::new();
    gpu_model.visit_tensors(&ModulePath::root(), &mut |path, tensor, _role| {
        synced_tensors.insert(path.as_string(), tensor.clone());
        Ok(())
    })?;
    for (name, id) in &training.parameter_ids {
        let Some(initial) = initial_tensors.get(name) else { continue };
        let Some(synced) = synced_tensors.get(name) else { continue };
        let Some(stored) = host_after.get(*id).ok() else { continue };
        for ((a, b), c) in initial.data().iter().zip(synced.data()).zip(stored.data.data()) {
            pre_sync = pre_sync.max((a - c).abs());
            post_sync = post_sync.max((b - c).abs());
        }
    }
    println!("sync diagnostics: host-vs-initial {pre_sync:.3e}, host-vs-synced {post_sync:.3e}");
    let mut cpu_tensors = std::collections::HashMap::new();
    cpu_model.visit_tensors(&ModulePath::root(), &mut |path, tensor, _role| {
        cpu_tensors.insert(path.as_string(), tensor.clone());
        Ok(())
    })?;
    let mut gpu_tensors = std::collections::HashMap::new();
    gpu_model.visit_tensors(&ModulePath::root(), &mut |path, tensor, _role| {
        gpu_tensors.insert(path.as_string(), tensor.clone());
        Ok(())
    })?;
    let mut max_diff = 0.0f32;
    let mut worst = String::new();
    for (tensor_name, cpu) in &cpu_tensors {
        let Some(gpu) = gpu_tensors.get(tensor_name) else { continue };
        for (a, b) in cpu.data().iter().zip(gpu.data()) {
            let diff = (a - b).abs();
            if diff > max_diff {
                max_diff = diff;
                worst = tensor_name.clone();
            }
        }
    }
    println!(
        "{name}: loss diff {:.3e}, one-step parameter max |diff| {:.3e} ({worst})",
        (out.loss - gpu_loss).abs(),
        max_diff
    );
    Ok(max_diff)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let conv = TinyConv {
        conv: ConvNormAct::new(3, 4, [3, 3], [1, 1], 1, ConvActivation::Relu),
        depthwise: ConvNormAct::new(4, 4, [3, 3], [1, 1], 4, ConvActivation::Relu),
        pool: GlobalAvgPool,
        fc: Linear::with_seed(4, 2, true, 7),
    };
    let conv_only = ConvOnly {
        conv: Conv2d::new(3, 4, [3, 3], [1, 1], [1, 1]).without_bias(),
        pool: GlobalAvgPool,
        fc: Linear::with_seed(4, 2, true, 7),
    };
    let bn_only = BnOnly {
        bn: BatchNorm::new(3),
        pool: GlobalAvgPool,
        fc: Linear::with_seed(3, 2, true, 7),
    };
    let linear = LinearOnly {
        fc: Linear::with_seed(3, 2, true, 7),
    };
    let d0 = compare("linear-only", linear)?;
    let d1 = compare("conv+bn+dw", conv)?;
    let d2 = compare("conv-only", conv_only)?;
    let d3 = compare("bn-only", bn_only)?;
    let worst = d0.max(d1).max(d2).max(d3);
    println!("GRADIENT_CHECK: {}", if worst < 5e-3 { "PASS" } else { "FAIL" });
    if worst >= 5e-3 {
        return Err("gradient mismatch".into());
    }
    Ok(())
}
