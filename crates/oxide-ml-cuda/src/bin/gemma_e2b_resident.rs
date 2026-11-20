//! Fixed-length, resident-parameter E2B decode (no per-token re-upload).
//!
//! This is a latency experiment: compile once for `prompt + new_tokens`, pad the
//! sequence with token 0 after the current end, and read the logits at the last
//! real position. Causal masking makes trailing padding irrelevant. It is not a
//! KV cache, but it removes the per-token graph compile and 18.6 GB upload.

use std::error::Error;
use std::time::Instant;

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
    let generate: usize = std::env::args()
        .nth(3)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let config = Gemma4Config::from_path(format!("{dir}/config.json"))?;
    let tokenizer = GemmaTokenizer::from_file(format!("{dir}/tokenizer.json"))?;
    let mut ids = tokenizer.encode(&prompt)?;
    let max_len = ids.len() + generate;
    println!("prompt: {prompt:?} -> {} tokens; max_len={max_len}", ids.len());

    let load_start = Instant::now();
    let file = SafeTensorsFile::open(format!("{dir}/model.safetensors"))?;
    let weights = Gemma4Weights::new(&file)?;
    let model = Gemma4Text::from_weights(config.clone(), &weights)?;
    println!(
        "load+f32 time: {:.3} s, parameters: {}",
        load_start.elapsed().as_secs_f64(),
        model.num_parameters()
    );
    let values: Vec<f32> = ids.iter().map(|id| *id as f32).collect();
    let mut padded = values.clone();
    padded.resize(max_len, 0.0);
    let specs = [InputSpec::unbatched("input0", [max_len])];
    let compile_start = Instant::now();
    let gpu = CudaModel::compile(
        CudaDevice::new(0)?,
        &model,
        &specs,
        ModelCompileOptions::default(),
    )?;
    println!("compile+upload time: {:.3} s", compile_start.elapsed().as_secs_f64());

    let mut generation_time = 0.0f64;
    for step in 0..=generate {
        let run_start = Instant::now();
        let outputs = gpu.run(&[(
            "input0",
            DenseTensor::from_vec([max_len], padded.clone())?,
        )])?;
        let elapsed = run_start.elapsed().as_secs_f64();
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
        println!("step {step}: run {elapsed:.3} s");
        for (id, value) in &top {
            println!("  {id:6}  {value:10.6}  {:?}", tokenizer.decode(&[*id as u32])?);
        }
        if step > 0 {
            generation_time += elapsed;
        }
        if step < generate {
            let next = top[0].0 as u32;
            ids.push(next);
            padded[ids.len() - 1] = next as f32;
            if config.eos_token_ids.contains(&next) {
                break;
            }
        }
    }
    println!(
        "generation run total ({} token(s)): {:.3} s",
        generate, generation_time
    );
    Ok(())
}
