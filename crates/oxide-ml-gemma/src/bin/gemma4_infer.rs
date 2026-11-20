//! Gemma 4 text inference from safetensors weights.
//!
//! ```text
//! gemma4_infer --model /path/to/model --tokens 2,123,456 --reference /path/ref.json
//! ```
//!
//! Token ids are produced by the HuggingFace tokenizer (see
//! `scripts/gemma4_reference.py`); this binary is the Rust model runtime.

use std::path::PathBuf;

use oxide_ml_gemma::{Gemma4Config, Gemma4Text, Gemma4Weights, GemmaTokenizer, StreamingGemma};
use oxide_ml_model::{CompiledModel, InputSpec, ModelCompileOptions, Tensor};
use oxide_ml_safetensors::SafeTensorsFile;

struct Args {
    model: PathBuf,
    tokens: Vec<f32>,
    reference: Option<PathBuf>,
    generate: usize,
    tolerance: f32,
    inspect: bool,
    streaming: bool,
    prompt: Option<String>,
    tokenize: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = std::env::args().skip(1);
    let mut model = None;
    let mut tokens: Vec<f32> = Vec::new();
    let mut reference = None;
    let mut generate = 0usize;
    let mut tolerance = 1e-2f32;
    let mut inspect = false;
    let mut streaming = false;
    let mut prompt = None;
    let mut tokenize = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(args.next().ok_or("--model needs a path")?)),
            "--tokens" => {
                let value = args.next().ok_or("--tokens needs a list")?;
                tokens = value
                    .split(',')
                    .map(|value| {
                        value
                            .trim()
                            .parse::<f32>()
                            .map_err(|error| format!("invalid token {value:?}: {error}"))
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--reference" => {
                reference = Some(PathBuf::from(
                    args.next().ok_or("--reference needs a path")?,
                ))
            }
            "--generate" => {
                generate = args
                    .next()
                    .ok_or("--generate needs a count")?
                    .parse()
                    .map_err(|error| format!("invalid --generate: {error}"))?;
            }
            "--inspect" => inspect = true,
            "--streaming" => streaming = true,
            "--prompt" => prompt = Some(args.next().ok_or("--prompt needs text")?),
            "--tokenize" => tokenize = true,
            "--tolerance" => {
                tolerance = args
                    .next()
                    .ok_or("--tolerance needs a value")?
                    .parse()
                    .map_err(|error| format!("invalid --tolerance: {error}"))?;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let model = model.ok_or("--model is required")?;
    Ok(Args {
        model,
        tokens,
        reference,
        generate,
        tolerance,
        inspect,
        streaming,
        prompt,
        tokenize,
    })
}

type Reference = (Vec<f32>, Vec<f32>, Vec<u32>, f32);

fn load_reference(path: &PathBuf) -> Result<Reference, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read reference {}: {error}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("reference JSON: {error}"))?;
    let ids = json
        .get("input_ids")
        .and_then(|value| value.as_array())
        .ok_or("reference is missing input_ids")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|value| value as f32)
                .ok_or("reference input_ids must be integers")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let logits_path = json
        .get("logits_file")
        .and_then(|value| value.as_str())
        .ok_or("reference is missing logits_file")?;
    let logits_path = path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(logits_path);
    let bytes = std::fs::read(&logits_path)
        .map_err(|error| format!("cannot read {}: {error}", logits_path.display()))?;
    let logits = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
        .collect::<Vec<_>>();
    let generated = json
        .get("generated")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_u64().map(|value| value as u32))
                .collect()
        })
        .unwrap_or_default();
    let tolerance = json
        .get("tolerance")
        .and_then(|value| value.as_f64())
        .map(|value| value as f32)
        .unwrap_or(1e-2);
    Ok((ids, logits, generated, tolerance))
}

fn compile(
    model: &Gemma4Text,
    tokens: &[f32],
) -> Result<(CompiledModel, oxide_ml_model::ParameterStore<f32>), oxide_ml_core::Error> {
    let length = tokens.len();
    let specs = [InputSpec::unbatched("input0", [length])];
    let compiled = CompiledModel::compile(model, &specs, ModelCompileOptions::default())?;
    let store = compiled.parameter_store(model)?;
    Ok((compiled, store))
}

fn forward_last(
    model: &Gemma4Text,
    tokens: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), oxide_ml_core::Error> {
    let (compiled, store) = compile(model, tokens)?;
    let input = Tensor::<f32>::from_vec([tokens.len()], tokens.to_vec())?;
    let outputs = compiled.run(&store, &[("input0", &input)])?;
    let logits = &outputs[0];
    let dims = logits.dims();
    if dims.len() != 2 {
        return Err(oxide_ml_core::Error::invalid_graph(format!(
            "expected logits [T, V], got {dims:?}"
        )));
    }
    let vocab = dims[1];
    let last = logits.data()[(dims[0] - 1) * vocab..].to_vec();
    Ok((outputs[0].data().to_vec(), last))
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gemma4_infer: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = parse_args()?;
    let reference = if let Some(path) = &args.reference {
        Some(load_reference(path)?)
    } else {
        None
    };
    if args.tokens.is_empty() {
        if let Some((ids, _, _, _)) = &reference {
            args.tokens = ids.clone();
        }
    }
    let mut tokenizer = None;
    if let Some(prompt) = &args.prompt {
        let tokenizer_path = args.model.join("tokenizer.json");
        let loaded = GemmaTokenizer::from_file(&tokenizer_path)
            .map_err(|error| format!("tokenizer: {error}"))?;
        let ids = loaded
            .encode(prompt)
            .map_err(|error| format!("tokenizer: {error}"))?;
        println!("prompt token ids: {ids:?}");
        if args.tokens.is_empty() {
            args.tokens = ids.iter().map(|id| *id as f32).collect();
        }
        tokenizer = Some(loaded);
    }
    if args.tokenize {
        let ids: Vec<u32> = args.tokens.iter().map(|value| *value as u32).collect();
        let tokenizer = match tokenizer {
            Some(tokenizer) => tokenizer,
            None => GemmaTokenizer::from_file(args.model.join("tokenizer.json"))
                .map_err(|error| format!("tokenizer: {error}"))?,
        };
        let text = tokenizer
            .decode(&ids)
            .map_err(|error| format!("tokenizer: {error}"))?;
        println!("decoded: {text:?}");
        return Ok(());
    }
    if !args.inspect && args.tokens.is_empty() {
        return Err("--tokens, --prompt (or a reference with input_ids) is required".into());
    }
    if let Some((ids, _, _, _)) = &reference
        && ids.len() != args.tokens.len()
    {
        return Err(format!(
            "token count {} does not match reference input_ids {}",
            args.tokens.len(),
            ids.len()
        ));
    }

    let config = Gemma4Config::from_path(args.model.join("config.json"))
        .map_err(|error| format!("config: {error}"))?;
    if args.inspect {
        let sliding = config
            .layer_types
            .iter()
            .filter(|kind| **kind == oxide_ml_gemma::LayerKind::Sliding)
            .count();
        println!("hidden_size: {}", config.hidden_size);
        println!("num_hidden_layers: {}", config.num_hidden_layers);
        println!(
            "heads: {} q / {} kv",
            config.num_attention_heads, config.num_key_value_heads
        );
        println!(
            "head_dim: {} sliding / {:?} full",
            config.head_dim, config.global_head_dim
        );
        println!(
            "layers: {sliding} sliding, {} full, {} shared-KV",
            config.num_hidden_layers - sliding,
            config.num_kv_shared_layers
        );
        println!("PLE dim: {}", config.hidden_size_per_layer_input);
        println!("double-wide MLP: {}", config.use_double_wide_mlp);
        println!("vocab: {}", config.vocab_size);
        println!("softcap: {:?}", config.final_logit_softcapping);
        return Ok(());
    }
    if args.streaming {
        return run_streaming(&args, &reference);
    }
    let weights_path = args.model.join("model.safetensors");
    let file =
        SafeTensorsFile::open(&weights_path).map_err(|error| format!("safetensors: {error}"))?;
    let weights = Gemma4Weights::new(&file).map_err(|error| error.to_string())?;
    println!("weights prefix: {:?}", weights.prefix());
    println!(
        "model: hidden={} layers={} heads={} kv={} vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size
    );
    let model = Gemma4Text::from_weights(config.clone(), &weights)
        .map_err(|error| format!("model: {error}"))?;
    println!("parameters: {}", model.num_parameters());

    let (all_logits, last) =
        forward_last(&model, &args.tokens).map_err(|error| format!("forward: {error}"))?;
    let mut top: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
    top.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top.truncate(5);
    println!("top-5 next tokens:");
    for (id, value) in &top {
        println!("  {id:6}  {value:10.6}");
    }
    let argmax = top[0].0 as u32;
    println!("greedy next token: {argmax}");

    if let Some((_, expected, generated, tolerance)) = &reference {
        let tolerance = args.tolerance.min(*tolerance);
        if expected.len() == last.len() {
            let max_diff = expected
                .iter()
                .zip(&last)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("reference max |diff|: {max_diff:.3e} (tolerance {tolerance:.1e})");
            if max_diff > tolerance {
                return Err("logits do not match the reference".into());
            }
        } else {
            println!(
                "reference logits length {} does not match {}",
                expected.len(),
                last.len()
            );
        }
        if !generated.is_empty() {
            println!("reference generated: {generated:?}");
        }
    }
    let _ = all_logits;

    if args.generate > 0 {
        let mut tokens = args.tokens.clone();
        let mut generated = Vec::new();
        for _ in 0..args.generate {
            let (_, last) =
                forward_last(&model, &tokens).map_err(|error| format!("generate: {error}"))?;
            let mut best = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for (index, value) in last.iter().enumerate() {
                if *value > best_value {
                    best_value = *value;
                    best = index;
                }
            }
            generated.push(best as u32);
            tokens.push(best as f32);
            if config.eos_token_ids.contains(&(best as u32)) {
                break;
            }
        }
        println!("generated tokens: {generated:?}");
        print_generated_text(&args, &generated);
    }
    Ok(())
}

fn print_generated_text(args: &Args, generated: &[u32]) {
    if generated.is_empty() {
        return;
    }
    let Ok(tokenizer) = GemmaTokenizer::from_file(args.model.join("tokenizer.json")) else {
        return;
    };
    if let Ok(text) = tokenizer.decode(generated) {
        println!("generated text: {text:?}");
    }
}

fn top5(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top.truncate(5);
    top
}

fn run_streaming(args: &Args, reference: &Option<Reference>) -> Result<(), String> {
    let mut model =
        StreamingGemma::open(&args.model).map_err(|error| format!("streaming model: {error}"))?;
    let config = model.config().clone();
    let ids: Vec<u32> = args
        .tokens
        .iter()
        .map(|value| {
            if *value < 0.0 || value.fract() != 0.0 {
                Err(format!("invalid token id {value}"))
            } else {
                Ok(*value as u32)
            }
        })
        .collect::<Result<_, _>>()?;
    println!(
        "streaming: hidden={} layers={} ple={} shared_kv={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.hidden_size_per_layer_input,
        config.num_kv_shared_layers
    );
    let mut last = if args.generate > 0 {
        model
            .prefill(&ids)
            .map_err(|error| format!("streaming prefill: {error}"))?
    } else {
        model
            .forward(&ids)
            .map_err(|error| format!("streaming forward: {error}"))?
    };
    let top = top5(&last);
    println!("top-5 next tokens:");
    for (id, value) in &top {
        println!("  {id:6}  {value:10.6}");
    }
    println!("greedy next token: {}", top[0].0);
    if let Some((_, expected, generated, tolerance)) = reference {
        let tolerance = args.tolerance.min(*tolerance);
        if expected.len() == last.len() {
            let max_diff = expected
                .iter()
                .zip(&last)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("reference max |diff|: {max_diff:.3e} (tolerance {tolerance:.1e})");
            if max_diff > tolerance {
                return Err("streaming logits do not match the reference".into());
            }
        }
        if !generated.is_empty() {
            println!("reference generated: {generated:?}");
        }
    }
    if args.generate > 0 {
        let mut generated = Vec::new();
        for _ in 0..args.generate {
            let mut best = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for (index, value) in last.iter().enumerate() {
                if *value > best_value {
                    best_value = *value;
                    best = index;
                }
            }
            generated.push(best as u32);
            if config.eos_token_ids.contains(&(best as u32)) {
                break;
            }
            last = model
                .decode(best as u32)
                .map_err(|error| format!("streaming decode: {error}"))?;
        }
        println!("generated tokens: {generated:?}");
        print_generated_text(args, &generated);
    }
    Ok(())
}
