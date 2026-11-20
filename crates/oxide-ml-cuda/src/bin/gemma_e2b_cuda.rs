//! Full `google/gemma-4-E2B-it` text forward on CUDA (A100).

use std::error::Error;

use oxide_ml_core::DenseTensor;
use oxide_ml_cuda::{CudaDevice, CudaModel};
use oxide_ml_gemma::safetensors::SafeTensorsFile;
use oxide_ml_gemma::{Gemma4Config, Gemma4Text, Gemma4Weights, GemmaTokenizer};
use oxide_ml_model::{InputSpec, ModelCompileOptions};

fn main() -> Result<(), Box<dyn Error>> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/content/models/gemma4-e2b".to_string());
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Hello".to_string());
    let config = Gemma4Config::from_path(format!("{dir}/config.json"))?;
    let tokenizer = GemmaTokenizer::from_file(format!("{dir}/tokenizer.json"))?;
    let ids = tokenizer.encode(&prompt)?;
    println!("prompt: {prompt:?} -> {} tokens", ids.len());

    let file = SafeTensorsFile::open(format!("{dir}/model.safetensors"))?;
    let weights = Gemma4Weights::new(&file)?;
    println!("loading text weights as f32...");
    let model = Gemma4Text::from_weights(config.clone(), &weights)?;
    println!("parameters: {}", model.num_parameters());
    let values: Vec<f32> = ids.iter().map(|id| *id as f32).collect();
    let specs = [InputSpec::unbatched("input0", [values.len()])];
    println!("compiling CUDA model...");
    let gpu = CudaModel::compile(
        CudaDevice::new(0)?,
        &model,
        &specs,
        ModelCompileOptions::default(),
    )?;
    println!("running...");
    let outputs = gpu.run(&[(
        "input0",
        DenseTensor::from_vec([values.len()], values)?,
    )])?;
    let logits = outputs[0].data();
    let vocab = config.vocab_size;
    let start = (ids.len() - 1) * vocab;
    let mut top: Vec<(usize, f32)> = logits[start..start + vocab]
        .iter()
        .copied()
        .enumerate()
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top.truncate(5);
    println!("top-5 next tokens:");
    for (id, value) in &top {
        println!("  {id:6}  {value:10.6}  {:?}", tokenizer.decode(&[*id as u32])?);
    }
    Ok(())
}
