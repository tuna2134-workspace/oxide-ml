//! CUDA decode with per-layer KV caches and resident parameters.

use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;

use oxide_ml_core::DenseTensor;
use oxide_ml_cuda::{CudaDevice, CudaModel};
use oxide_ml_gemma::safetensors::SafeTensorsFile;
use oxide_ml_gemma::{
    Gemma4Config, Gemma4Text, Gemma4Weights, GemmaDecode, GemmaTokenizer, StreamingGemma,
};
use oxide_ml_model::{InputSpec, ModelCompileOptions};

fn spec(name: String, dims: Vec<usize>) -> InputSpec {
    InputSpec::unbatched(name, dims)
}

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
    let ids = tokenizer.encode(&prompt)?;
    println!("prompt: {prompt:?} -> {} tokens", ids.len());

    let load_start = Instant::now();
    let file = SafeTensorsFile::open(format!("{dir}/model.safetensors"))?;
    let weights = Gemma4Weights::new(&file)?;
    let model = Gemma4Text::from_weights(config.clone(), &weights)?;
    println!(
        "load+f32 time: {:.3} s, parameters: {}",
        load_start.elapsed().as_secs_f64(),
        model.num_parameters()
    );

    let prefill_start = Instant::now();
    let mut stream = StreamingGemma::open(&dir)?;
    let mut session = stream.session()?;
    let _ = stream.prefill(&mut session, &ids)?;
    let prefill_time = prefill_start.elapsed().as_secs_f64();
    println!("CPU KV prefill ({} tokens): {prefill_time:.3} s", ids.len());

    let hidden_size = config.hidden_size;
    let layers = config.num_hidden_layers;
    // Per-layer cache in [1, P, kv_heads, head_dim] form for non-shared layers.
    let mut caches: Vec<Option<(Vec<f32>, Vec<usize>)>> = vec![None; layers];
    let state = session.state();
    let mut position = state.sequence_length();
    for index in 0..layers {
        let layer = config.layer(index)?;
        if layer.is_shared {
            continue;
        }
        let keys = state.keys(index)?.to_vec();
        let values = state.values(index)?.to_vec();
        let dims = vec![1, position, layer.num_kv_heads, layer.head_dim];
        caches[index] = Some(([keys, values].concat(), dims));
    }

    let mut ids = ids;
    let mut generation_time = 0.0f64;
    let mut compile_time = 0.0f64;
    let mut model_gpu: Option<CudaModel> = None;
    for step in 0..=generate {
        let past_len = position;
        let mut specs = vec![spec("token".to_string(), vec![1])];
        for index in 0..layers {
            let layer = config.layer(index)?;
            if layer.is_shared {
                continue;
            }
            specs.push(spec(format!("past_k_{index}"), vec![1, past_len, layer.num_kv_heads, layer.head_dim]));
            specs.push(spec(format!("past_v_{index}"), vec![1, past_len, layer.num_kv_heads, layer.head_dim]));
        }
        let decode = GemmaDecode {
            model: &model,
            position,
            past_len,
        };
        let step_start = Instant::now();
        match &mut model_gpu {
            None => {
                model_gpu = Some(CudaModel::compile(
                    CudaDevice::new(0)?,
                    &decode,
                    &specs,
                    ModelCompileOptions::default(),
                )?);
                compile_time += step_start.elapsed().as_secs_f64();
            }
            Some(gpu) => {
                gpu.rebuild(&decode, &specs, ModelCompileOptions::default())?;
            }
        }
        let run_start = Instant::now();
        let token = *ids.last().unwrap() as f32;
        let mut host: HashMap<String, DenseTensor<f32>> = HashMap::new();
        host.insert(
            "token".to_string(),
            DenseTensor::from_vec([1], vec![token])?,
        );
        for index in 0..layers {
            let layer = config.layer(index)?;
            if layer.is_shared {
                continue;
            }
            let (values, dims) = caches[index]
                .as_ref()
                .ok_or("missing cache for non-shared layer")?;
            let k_len = layer.num_kv_heads * layer.head_dim * past_len;
            let k = values[..k_len].to_vec();
            let v = values[k_len..].to_vec();
            host.insert(
                format!("past_k_{index}"),
                DenseTensor::from_vec(dims.clone(), k)?,
            );
            host.insert(
                format!("past_v_{index}"),
                DenseTensor::from_vec(dims.clone(), v)?,
            );
        }
        let inputs: Vec<(&str, DenseTensor<f32>)> = host
            .into_iter()
            .map(|(name, tensor)| (Box::leak(name.into_boxed_str()) as &str, tensor))
            .collect();
        let outputs = model_gpu.as_ref().unwrap().run(&inputs)?;
        let elapsed = run_start.elapsed().as_secs_f64();
        let logits = outputs[0].data();
        let vocab = config.vocab_size;
        let mut top: Vec<(usize, f32)> = logits[..vocab].iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top.truncate(5);
        println!("step {step}: run {elapsed:.3} s (past {past_len})");
        for (id, value) in &top {
            println!("  {id:6}  {value:10.6}  {:?}", tokenizer.decode(&[*id as u32])?);
        }
        if step > 0 {
            generation_time += elapsed;
        }
        if step < generate {
            let next = top[0].0 as u32;
            ids.push(next);
            // Update caches: outputs are logits + (k, v) per non-shared layer,
            // each [1, past_len + 1, kv_heads, head_dim].
            let mut cursor = 1usize;
            for index in 0..layers {
                let layer = config.layer(index)?;
                if layer.is_shared {
                    continue;
                }
                let k = outputs[cursor].data().to_vec();
                let v = outputs[cursor + 1].data().to_vec();
                caches[index] = Some(([k, v].concat(), vec![1, past_len + 1, layer.num_kv_heads, layer.head_dim]));
                cursor += 2;
            }
            position += 1;
            if config.eos_token_ids.contains(&next) {
                break;
            }
        }
    }
    let _ = hidden_size;
    println!(
        "generation run total ({} token(s)): {generation_time:.3} s; compile/upload {compile_time:.3} s",
        generate
    );
    Ok(())
}
