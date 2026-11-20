//! Tiny Gemma 4 CPU/CUDA parity check for the new CUDA ops.

use std::error::Error;

use oxide_ml_core::DenseTensor;
use oxide_ml_cuda::{CudaDevice, CudaModel};
use oxide_ml_gemma::safetensors::SafeTensorsFile;
use oxide_ml_gemma::{Gemma4Config, Gemma4Text, Gemma4Weights};
use oxide_ml_model::{CompiledModel, InputSpec, ModelCompileOptions, Tensor};

fn main() -> Result<(), Box<dyn Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| ".refs/gemma4-e2b-tiny".to_string());
    let config = Gemma4Config::from_path(format!("{dir}/config.json"))?;
    let file = SafeTensorsFile::open(format!("{dir}/model.safetensors"))?;
    let weights = Gemma4Weights::new(&file)?;
    let model = Gemma4Text::from_weights(config.clone(), &weights)?;
    let ids = vec![5.0f32, 17.0, 42.0];
    let specs = [InputSpec::unbatched("input0", [ids.len()])];

    let cpu = CompiledModel::compile(&model, &specs, ModelCompileOptions::default())?;
    let store = cpu.parameter_store(&model)?;
    let input = Tensor::<f32>::from_vec([ids.len()], ids.clone())?;
    let cpu_out = cpu.run(&store, &[("input0", &input)])?;
    let cpu_logits = cpu_out[0].data().to_vec();

    let device = CudaDevice::new(0)?;
    println!("device: 0");
    let gpu = CudaModel::compile(device, &model, &specs, ModelCompileOptions::default())?;
    let gpu_out = gpu.run(&[(
        "input0",
        DenseTensor::from_vec([ids.len()], ids.clone())?,
    )])?;
    let gpu_logits = gpu_out[0].data();

    let vocab = config.vocab_size;
    let start = (ids.len() - 1) * vocab;
    let mut max_diff = 0.0f32;
    for (a, b) in cpu_logits[start..start + vocab]
        .iter()
        .zip(&gpu_logits[start..start + vocab])
    {
        max_diff = max_diff.max((a - b).abs());
    }
    println!("CPU/CUDA max |diff| = {max_diff:.3e}");
    if max_diff > 1e-3 {
        return Err(format!("parity failed: {max_diff}").into());
    }
    println!("GEMMA_CUDA_CHECK: PASS");
    Ok(())
}
