//! A decoder-only transformer written in the declarative model style.
//!
//! Shows reusable components (`RmsNorm`, `MultiHeadAttention`, `GatedMlp`),
//! `ModuleList`, nested module calls, residual connections and a
//! `#[model]` loop over layers. No graph nodes or `ValueId`s appear in the
//! model code.
//!
//! ```bash
//! cargo run -p oxide-ml --example decoder_transformer
//! ```

use oxide_ml::prelude::*;

#[derive(Module)]
pub struct FeedForward {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let gate = self.gate.forward(x).gelu();
        let up = self.up.forward(x);
        self.down.forward(gate * up)
    }
}

#[derive(Module)]
pub struct DecoderLayer {
    attention_norm: RmsNorm,
    attention: MultiHeadAttention,
    mlp_norm: RmsNorm,
    mlp: FeedForward,
}

impl DecoderLayer {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x);
        let x = residual + x;
        let residual = x.clone();
        let x = self.mlp_norm.forward(x);
        let x = self.mlp.forward(x);
        residual + x
    }
}

#[derive(Module)]
pub struct Transformer {
    embedding: Embedding,
    layers: ModuleList<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
}

impl Transformer {
    #[model]
    fn forward(&self, ids: Tensor<f32>) -> Tensor<f32> {
        let mut hidden = self.embedding.forward(ids);
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        self.lm_head.forward(self.norm.forward(hidden))
    }
}

fn main() -> Result<()> {
    let vocab = 64;
    let hidden = 32;
    let layers = 3;
    let model = Transformer {
        embedding: Embedding::with_seed(vocab, hidden, 7),
        layers: (0..layers)
            .map(|index| DecoderLayer {
                attention_norm: RmsNorm::with_dim(hidden, 1e-6),
                attention: MultiHeadAttention::with_seed(hidden, 4, index as u64)
                    .expect("valid attention")
                    .causal(),
                mlp_norm: RmsNorm::with_dim(hidden, 1e-6),
                mlp: FeedForward {
                    gate: Linear::with_seed(hidden, hidden * 2, true, index as u64 + 10),
                    up: Linear::with_seed(hidden, hidden * 2, true, index as u64 + 20),
                    down: Linear::with_seed(hidden * 2, hidden, true, index as u64 + 30),
                },
            })
            .collect(),
        norm: RmsNorm::with_dim(hidden, 1e-6),
        lm_head: Linear::with_seed(hidden, vocab, true, 99),
    };

    println!(
        "parameters: {} ({} layers, hidden {hidden}, vocab {vocab})",
        model.descriptor()?.parameter_count(),
        layers
    );

    let ids = Tensor::<f32>::from_vec([1, 6], vec![1.0, 4.0, 7.0, 2.0, 9.0, 3.0])?;
    let logits = model.predict(ids)?;
    println!("logits: {:?}", logits.dims());

    let last = logits
        .data()
        .chunks(vocab)
        .next_back()
        .expect("at least one position");
    let mut top: Vec<(usize, f32)> = last.iter().copied().enumerate().collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("top-5 next tokens: {:?}", &top[..5]);
    Ok(())
}
