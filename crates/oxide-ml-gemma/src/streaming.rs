//! Layer-streamed Gemma 4 inference.
//!
//! The official `google/gemma-4-E2B-it` text weights are ~9.3 GB BF16; a
//! full-precision host parameter store does not fit on a 12–16 GB machine.
//! This module compiles one decoder layer at a time, materializes only that
//! layer's weights as f32, runs it, and drops them. Token/PLE embedding rows
//! are read directly from the memory-mapped safetensors file, and the tied LM
//! head is evaluated in vocabulary chunks.

use std::collections::HashMap;
use std::sync::Arc;

use oxide_ml_core::{DenseTensor, Error, ParameterStore, Result};
use oxide_ml_model::{
    CompiledModel, InputSpec, ModelCompileOptions, Module, ModulePath, ParameterRegistry,
    ParameterRole, Tensor, ValueId,
};
use oxide_ml_safetensors::SafeTensorsFile;

use crate::config::{Gemma4Config, LayerKind};
use crate::modules::GemmaLayer;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct RunnerKey {
    sliding: bool,
    head_dim: usize,
    kv_heads: usize,
    shared: bool,
    store: bool,
    ple: usize,
    intermediate: usize,
    length: usize,
    decode: bool,
    position: usize,
}

type KvEntry = (Vec<f32>, Vec<f32>, Vec<usize>);
type SharedKv = Option<KvEntry>;

struct RunnerTemplate {
    compiled: Arc<CompiledModel>,
    template: ParameterStore<f32>,
}

struct LayerRunner {
    layer: GemmaLayer,
    has_ple: bool,
    shared: bool,
    store: bool,
    decode: bool,
    position: usize,
}

impl Module for LayerRunner {
    fn module_type(&self) -> &'static str {
        "Gemma4StreamingLayer"
    }

    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.layer.visit_parameters(path, registry)
    }

    fn forward_multi(
        &self,
        cx: &mut oxide_ml_model::ModelContext,
        inputs: &[ValueId],
    ) -> Result<Vec<ValueId>> {
        let required = 1 + usize::from(self.has_ple) + 2 * usize::from(self.shared || self.decode);
        if inputs.len() < required {
            return Err(Error::invalid_graph(format!(
                "streaming layer needs {required} inputs, got {}",
                inputs.len()
            )));
        }
        let per_layer = self.has_ple.then(|| inputs[1]);
        let k_index = 1 + usize::from(self.has_ple);
        let past = (self.shared || self.decode).then(|| (inputs[k_index], inputs[k_index + 1]));
        let (hidden, stored) = if self.decode {
            let (past_k, past_v) =
                past.ok_or_else(|| Error::invalid_graph("decode layer needs cached K/V"))?;
            self.layer.decode_with(
                cx,
                inputs[0],
                per_layer,
                Some((past_k, past_v)),
                self.position,
            )?
        } else {
            self.layer.apply_with(cx, inputs[0], per_layer, past)?
        };
        let mut outputs = vec![hidden];
        if self.store {
            let (k, v) = stored.ok_or_else(|| {
                Error::invalid_graph("streaming layer did not produce K/V outputs")
            })?;
            outputs.push(k);
            outputs.push(v);
        }
        Ok(outputs)
    }

    fn forward(
        &self,
        cx: &mut oxide_ml_model::ModelContext,
        input: ValueId,
    ) -> Result<Vec<ValueId>> {
        self.forward_multi(cx, &[input])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.layer.visit_tensors(path, visitor)
    }
}

/// Streaming Gemma 4 text runtime.
pub struct StreamingGemma {
    config: Gemma4Config,
    file: SafeTensorsFile,
    prefix: String,
    runners: HashMap<RunnerKey, RunnerTemplate>,
    store_flags: Vec<bool>,
    cache: Vec<Option<KvEntry>>,
    source_cache: [Option<KvEntry>; 2],
    position: usize,
}

impl std::fmt::Debug for StreamingGemma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingGemma")
            .field("layers", &self.config.num_hidden_layers)
            .field("runners", &self.runners.len())
            .finish()
    }
}

impl StreamingGemma {
    pub fn open(model_dir: impl AsRef<std::path::Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = Gemma4Config::from_path(model_dir.join("config.json"))?;
        let file = SafeTensorsFile::open(model_dir.join("model.safetensors"))?;
        let prefix = [
            "model.language_model.",
            "language_model.model.",
            "model.",
            "",
        ]
        .into_iter()
        .find(|prefix| file.contains(&format!("{prefix}embed_tokens.weight")))
        .ok_or_else(|| Error::invalid_graph("safetensors does not contain embed_tokens.weight"))?
        .to_string();
        let first_shared = config
            .num_hidden_layers
            .saturating_sub(config.num_kv_shared_layers);
        let mut store_flags = vec![false; config.num_hidden_layers];
        if config.num_kv_shared_layers > 0 {
            for kind in [LayerKind::Sliding, LayerKind::Full] {
                if let Some(index) = (0..first_shared)
                    .filter(|index| config.layer(*index).map(|layer| layer.kind) == Ok(kind))
                    .max()
                {
                    store_flags[index] = true;
                }
            }
        }
        let num_layers = config.num_hidden_layers;
        Ok(StreamingGemma {
            config,
            file,
            prefix,
            runners: HashMap::new(),
            store_flags,
            cache: vec![None; num_layers],
            source_cache: [None, None],
            position: 0,
        })
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    fn ensure_runner(&mut self, key: &RunnerKey, index: usize) -> Result<()> {
        if self.runners.contains_key(key) {
            return Ok(());
        }
        let dummy = GemmaLayer::dummy(&self.config, index, key.intermediate, key.store)?;
        let runner = LayerRunner {
            layer: dummy,
            has_ple: key.ple > 0,
            shared: key.shared,
            store: key.store,
            decode: key.decode,
            position: key.position,
        };
        let hidden = self.config.hidden_size;
        let mut specs = vec![InputSpec::unbatched("hidden", [1, key.length, hidden])];
        if key.ple > 0 {
            specs.push(InputSpec::unbatched("ple", [1, key.length, key.ple]));
        }
        if key.shared || key.decode {
            let past_len = if key.decode {
                if key.shared {
                    key.position + 1
                } else {
                    key.position
                }
            } else {
                key.length
            };
            specs.push(InputSpec::unbatched(
                "k",
                [1, past_len, key.kv_heads, key.head_dim],
            ));
            specs.push(InputSpec::unbatched(
                "v",
                [1, past_len, key.kv_heads, key.head_dim],
            ));
        }
        let compiled = CompiledModel::compile(&runner, &specs, ModelCompileOptions::default())?;
        let template = compiled.parameter_store(&runner)?;
        self.runners.insert(
            key.clone(),
            RunnerTemplate {
                compiled: Arc::new(compiled),
                template,
            },
        );
        Ok(())
    }

    fn layer_store(
        &self,
        template: &ParameterStore<f32>,
        index: usize,
    ) -> Result<ParameterStore<f32>> {
        let mut store = template.clone();
        for parameter in store.iter_mut() {
            let name = format!("{}layers.{}.{}", self.prefix, index, parameter.name);
            let loaded = self.file.tensor(&name)?;
            if loaded.shape != parameter.data.dims() {
                return Err(Error::shape_mismatch(
                    format!("streaming weight {name:?}"),
                    parameter.data.shape().clone(),
                    oxide_ml_core::Shape::from_static(&loaded.shape),
                ));
            }
            parameter.data = DenseTensor::from_vec(&loaded.shape, loaded.data)?;
        }
        Ok(store)
    }

    fn embeddings(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let hidden = self.config.hidden_size;
        let scale = (hidden as f32).sqrt();
        let embed_name = format!("{}embed_tokens.weight", self.prefix);
        let mut values = Vec::with_capacity(ids.len() * hidden);
        for id in ids {
            let row = self.file.row(&embed_name, *id as usize)?;
            values.extend(row.into_iter().map(|value| value * scale));
        }
        Ok(values)
    }

    fn per_layer_inputs(&self, ids: &[u32], embeddings: &[f32]) -> Result<Option<Vec<Vec<f32>>>> {
        let ple_dim = self.config.hidden_size_per_layer_input;
        if ple_dim == 0 {
            return Ok(None);
        }
        let layers = self.config.num_hidden_layers;
        let hidden = self.config.hidden_size;
        let length = ids.len();
        let packed = layers * ple_dim;
        let ple_scale = (ple_dim as f32).sqrt();
        let token_name = format!("{}embed_tokens_per_layer.weight", self.prefix);
        let mut tokens = Vec::with_capacity(length * packed);
        for id in ids {
            let row = self.file.row(&token_name, *id as usize)?;
            tokens.extend(row.into_iter().map(|value| value * ple_scale));
        }
        let projection = self
            .file
            .tensor(&format!("{}per_layer_model_projection.weight", self.prefix))?;
        if projection.shape != [packed, hidden] {
            return Err(Error::shape_mismatch(
                "per_layer_model_projection",
                oxide_ml_core::Shape::from_static([packed, hidden]),
                oxide_ml_core::Shape::from_static(&projection.shape),
            ));
        }
        let projection = projection.data;
        let norm = self
            .file
            .tensor(&format!("{}per_layer_projection_norm.weight", self.prefix))?;
        let norm = norm.data;
        let context_scale = 1.0 / (hidden as f32).sqrt();
        let combine_scale = std::f32::consts::FRAC_1_SQRT_2;
        let mut per_layer = vec![vec![0.0f32; length * ple_dim]; layers];
        for t in 0..length {
            for (layer, packed_layer) in per_layer.iter_mut().enumerate() {
                let base = layer * ple_dim;
                let mut context = vec![0.0f32; ple_dim];
                for j in 0..ple_dim {
                    let row = &projection[(base + j) * hidden..(base + j + 1) * hidden];
                    let mut sum = 0.0f32;
                    for h in 0..hidden {
                        sum += embeddings[t * hidden + h] * row[h];
                    }
                    context[j] = sum * context_scale;
                }
                let mut mean = 0.0f32;
                for value in &context {
                    mean += value * value;
                }
                mean /= ple_dim as f32;
                let inv = 1.0 / (mean + self.config.rms_norm_eps).sqrt();
                let token_base = t * packed + base;
                for j in 0..ple_dim {
                    let normalized = context[j] * inv * norm[j];
                    let token = tokens[token_base + j];
                    packed_layer[t * ple_dim + j] = (normalized + token) * combine_scale;
                }
            }
        }
        Ok(Some(per_layer))
    }

    fn finish_logits(&self, hidden: &[f32], length: usize) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let config = &self.config;
        // Final RMSNorm on the last token.
        let norm_weight = self.file.tensor(&format!("{}norm.weight", self.prefix))?;
        let norm_weight = norm_weight.data;
        let last = &hidden[(length - 1) * hidden_size..];
        let mut mean = 0.0f32;
        for value in last {
            mean += value * value;
        }
        mean /= hidden_size as f32;
        let inv = 1.0 / (mean + config.rms_norm_eps).sqrt();
        let normalized: Vec<f32> = last
            .iter()
            .zip(&norm_weight)
            .map(|(value, weight)| value * inv * weight)
            .collect();

        // Tied LM head in chunks.
        let vocab = config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        let chunk = 4096usize;
        let head_name = format!("{}embed_tokens.weight", self.prefix);
        let mut start = 0;
        while start < vocab {
            let count = chunk.min(vocab - start);
            let rows = self.file.rows(&head_name, start, count)?;
            for row in 0..count {
                let values = &rows[row * hidden_size..(row + 1) * hidden_size];
                let mut sum = 0.0f32;
                let mut index = 0;
                while index + 4 <= hidden_size {
                    sum += values[index] * normalized[index];
                    sum += values[index + 1] * normalized[index + 1];
                    sum += values[index + 2] * normalized[index + 2];
                    sum += values[index + 3] * normalized[index + 3];
                    index += 4;
                }
                while index < hidden_size {
                    sum += values[index] * normalized[index];
                    index += 1;
                }
                logits[start + row] = sum;
            }
            start += count;
        }
        if let Some(cap) = config.final_logit_softcapping {
            for value in &mut logits {
                *value = (*value / cap).tanh() * cap;
            }
        }
        Ok(logits)
    }

    /// Run a full prompt forward and return the last-position logits.
    pub fn forward(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::invalid_shape("Gemma4 streaming", "empty prompt"));
        }
        let config = self.config.clone();
        let length = ids.len();
        let hidden_size = config.hidden_size;
        let ple_dim = config.hidden_size_per_layer_input;
        let mut hidden = self.embeddings(ids)?;
        let per_layer = self.per_layer_inputs(ids, &hidden)?;
        let mut shared: [SharedKv; 2] = [None, None];

        for index in 0..config.num_hidden_layers {
            let layer = config.layer(index)?;
            let intermediate = config.intermediate_size
                * if config.use_double_wide_mlp && layer.is_shared {
                    2
                } else {
                    1
                };
            let key = RunnerKey {
                sliding: layer.kind == LayerKind::Sliding,
                head_dim: layer.head_dim,
                kv_heads: layer.num_kv_heads,
                shared: layer.is_shared,
                store: self.store_flags[index],
                ple: ple_dim,
                intermediate,
                length,
                decode: false,
                position: 0,
            };
            self.ensure_runner(&key, index)?;
            let runner = self.runners.get(&key).expect("runner inserted");
            let store = self.layer_store(&runner.template, index)?;
            let hidden_tensor = Tensor::from_vec([1, length, hidden_size], hidden.clone())?;
            let ple_tensor = per_layer
                .as_ref()
                .map(|values| Tensor::from_vec([1, length, ple_dim], values[index].clone()))
                .transpose()?;
            let shared_tensors = if layer.is_shared {
                let slot = if layer.kind == LayerKind::Sliding {
                    0
                } else {
                    1
                };
                let (k_values, v_values, dims) = shared[slot].as_ref().ok_or_else(|| {
                    Error::invalid_graph(format!(
                        "shared layer {index} has no K/V from a previous layer"
                    ))
                })?;
                Some((
                    Tensor::from_vec(dims.clone(), k_values.clone())?,
                    Tensor::from_vec(dims.clone(), v_values.clone())?,
                ))
            } else {
                None
            };
            let mut inputs: Vec<(&str, &Tensor<f32>)> = vec![("hidden", &hidden_tensor)];
            if let Some(ple) = &ple_tensor {
                inputs.push(("ple", ple));
            }
            if let Some((k, v)) = &shared_tensors {
                inputs.push(("k", k));
                inputs.push(("v", v));
            }
            let outputs = runner.compiled.run(&store, &inputs)?;
            hidden = outputs[0].data().to_vec();
            if self.store_flags[index] {
                let slot = if layer.kind == LayerKind::Sliding {
                    0
                } else {
                    1
                };
                let k = outputs
                    .get(1)
                    .ok_or_else(|| Error::invalid_graph("missing K output"))?;
                let v = outputs
                    .get(2)
                    .ok_or_else(|| Error::invalid_graph("missing V output"))?;
                shared[slot] = Some((k.data().to_vec(), v.data().to_vec(), k.dims().to_vec()));
            }
        }

        self.finish_logits(&hidden, length)
    }

    /// Prefill a prompt and populate the per-layer KV cache.
    pub fn prefill(&mut self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() {
            return Err(Error::invalid_shape("Gemma4 prefill", "empty prompt"));
        }
        let config = self.config.clone();
        let length = ids.len();
        let hidden_size = config.hidden_size;
        let ple_dim = config.hidden_size_per_layer_input;
        let mut hidden = self.embeddings(ids)?;
        let per_layer = self.per_layer_inputs(ids, &hidden)?;
        self.cache = vec![None; config.num_hidden_layers];
        self.source_cache = [None, None];
        let mut shared: [SharedKv; 2] = [None, None];

        for index in 0..config.num_hidden_layers {
            let layer = config.layer(index)?;
            let intermediate = config.intermediate_size
                * if config.use_double_wide_mlp && layer.is_shared {
                    2
                } else {
                    1
                };
            let key = RunnerKey {
                sliding: layer.kind == LayerKind::Sliding,
                head_dim: layer.head_dim,
                kv_heads: layer.num_kv_heads,
                shared: layer.is_shared,
                store: !layer.is_shared,
                ple: ple_dim,
                intermediate,
                length,
                decode: false,
                position: 0,
            };
            self.ensure_runner(&key, index)?;
            let runner = self.runners.get(&key).expect("runner inserted");
            let store = self.layer_store(&runner.template, index)?;
            let hidden_tensor = Tensor::from_vec([1, length, hidden_size], hidden.clone())?;
            let ple_tensor = per_layer
                .as_ref()
                .map(|values| Tensor::from_vec([1, length, ple_dim], values[index].clone()))
                .transpose()?;
            let slot = if layer.kind == LayerKind::Sliding {
                0
            } else {
                1
            };
            let shared_tensors = if layer.is_shared {
                let (k_values, v_values, dims) = shared[slot].as_ref().ok_or_else(|| {
                    Error::invalid_graph(format!("shared layer {index} has no K/V during prefill"))
                })?;
                Some((
                    Tensor::from_vec(dims.clone(), k_values.clone())?,
                    Tensor::from_vec(dims.clone(), v_values.clone())?,
                ))
            } else {
                None
            };
            let mut inputs: Vec<(&str, &Tensor<f32>)> = vec![("hidden", &hidden_tensor)];
            if let Some(ple) = &ple_tensor {
                inputs.push(("ple", ple));
            }
            if let Some((k, v)) = &shared_tensors {
                inputs.push(("k", k));
                inputs.push(("v", v));
            }
            let outputs = runner.compiled.run(&store, &inputs)?;
            hidden = outputs[0].data().to_vec();
            if !layer.is_shared {
                let k = outputs
                    .get(1)
                    .ok_or_else(|| Error::invalid_graph("missing prefill K output"))?;
                let v = outputs
                    .get(2)
                    .ok_or_else(|| Error::invalid_graph("missing prefill V output"))?;
                let entry = (k.data().to_vec(), v.data().to_vec(), k.dims().to_vec());
                self.cache[index] = Some(entry.clone());
                if self.store_flags[index] {
                    shared[slot] = Some(entry.clone());
                    self.source_cache[slot] = Some(entry);
                }
            }
        }
        self.position = length;
        self.finish_logits(&hidden, length)
    }

    /// Decode one token against the cache populated by [`prefill`].
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        let config = self.config.clone();
        let hidden_size = config.hidden_size;
        let ple_dim = config.hidden_size_per_layer_input;
        let position = self.position;
        let ids = [token];
        let mut hidden = self.embeddings(&ids)?;
        let per_layer = self.per_layer_inputs(&ids, &hidden)?;

        for index in 0..config.num_hidden_layers {
            let layer = config.layer(index)?;
            let intermediate = config.intermediate_size
                * if config.use_double_wide_mlp && layer.is_shared {
                    2
                } else {
                    1
                };
            let key = RunnerKey {
                sliding: layer.kind == LayerKind::Sliding,
                head_dim: layer.head_dim,
                kv_heads: layer.num_kv_heads,
                shared: layer.is_shared,
                store: !layer.is_shared,
                ple: ple_dim,
                intermediate,
                length: 1,
                decode: true,
                position,
            };
            self.ensure_runner(&key, index)?;
            let runner = self.runners.get(&key).expect("runner inserted");
            let store = self.layer_store(&runner.template, index)?;
            let hidden_tensor = Tensor::from_vec([1, 1, hidden_size], hidden.clone())?;
            let ple_tensor = per_layer
                .as_ref()
                .map(|values| Tensor::from_vec([1, 1, ple_dim], values[index].clone()))
                .transpose()?;
            let slot = if layer.kind == LayerKind::Sliding {
                0
            } else {
                1
            };
            let past = if layer.is_shared {
                self.source_cache[slot].as_ref()
            } else {
                self.cache[index].as_ref()
            }
            .ok_or_else(|| {
                Error::invalid_graph(format!("layer {index} has no cached K/V for decode"))
            })?;
            let (k_values, v_values, dims) = past;
            let k_tensor = Tensor::from_vec(dims.clone(), k_values.clone())?;
            let v_tensor = Tensor::from_vec(dims.clone(), v_values.clone())?;
            let mut inputs: Vec<(&str, &Tensor<f32>)> = vec![("hidden", &hidden_tensor)];
            if let Some(ple) = &ple_tensor {
                inputs.push(("ple", ple));
            }
            inputs.push(("k", &k_tensor));
            inputs.push(("v", &v_tensor));
            let outputs = runner.compiled.run(&store, &inputs)?;
            hidden = outputs[0].data().to_vec();
            if !layer.is_shared {
                let k = outputs
                    .get(1)
                    .ok_or_else(|| Error::invalid_graph("missing decode K output"))?;
                let v = outputs
                    .get(2)
                    .ok_or_else(|| Error::invalid_graph("missing decode V output"))?;
                let entry = (k.data().to_vec(), v.data().to_vec(), k.dims().to_vec());
                self.cache[index] = Some(entry.clone());
                if self.store_flags[index] {
                    self.source_cache[slot] = Some(entry);
                }
            }
        }
        self.position += 1;
        self.finish_logits(&hidden, 1)
    }
}
