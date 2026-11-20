//! Compile-pass fixture: the declarative transformer style from the mission.

use oxide_ml::prelude::*;

#[derive(Module)]
struct FeedForward {
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
struct DecoderLayer {
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
struct Transformer {
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

fn main() {
    let model = Transformer {
        embedding: Embedding::with_seed(32, 16, 1),
        layers: (0..2)
            .map(|index| DecoderLayer {
                attention_norm: RmsNorm::with_dim(16, 1e-6),
                attention: MultiHeadAttention::with_seed(16, 2, index).unwrap().causal(),
                mlp_norm: RmsNorm::with_dim(16, 1e-6),
                mlp: FeedForward {
                    gate: Linear::with_seed(16, 32, true, index + 1),
                    up: Linear::with_seed(16, 32, true, index + 2),
                    down: Linear::with_seed(32, 16, true, index + 3),
                },
            })
            .collect(),
        norm: RmsNorm::with_dim(16, 1e-6),
        lm_head: Linear::with_seed(16, 32, true, 9),
    };
    let ids = Tensor::<f32>::from_vec([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = model.predict(ids).unwrap();
    assert_eq!(output.dims(), vec![1, 4, 32]);
}
