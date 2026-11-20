//! Module-system walkthrough: `ModuleList`, enum variants, module loops and a
//! tied embedding/head.
//!
//! ```bash
//! cargo run -p oxide-ml --example module_system
//! ```

use std::sync::Arc;

use oxide_ml::prelude::*;

#[derive(Module)]
struct ResidualBlock {
    norm: LayerNorm,
    layers: ModuleList<Linear>,
}

impl ResidualBlock {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let residual = x.clone();
        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        self.norm.forward(hidden + residual)
    }
}

/// Alternative activation stacks; both variants share the same parameter paths.
#[derive(Module)]
#[allow(dead_code)]
enum Activation {
    Relu(ReLU),
    Tanh(Tanh),
}

#[derive(Module)]
struct TinyLm {
    embed: Arc<Embedding>,
    blocks: ModuleList<ResidualBlock>,
    activation: Activation,
    head: TiedLinear<Embedding>,
}

impl TinyLm {
    #[model]
    fn forward(&self, ids: Tensor<f32>) -> Tensor<f32> {
        let mut hidden = self.embed.forward(ids);
        for block in &self.blocks {
            hidden = block.forward(hidden);
        }
        let hidden = self.activation.forward(hidden);
        self.head.forward(hidden)
    }
}

fn main() -> Result<()> {
    let embed = Arc::new(Embedding::with_seed(64, 16, 7));
    let model = TinyLm {
        embed: embed.clone(),
        blocks: (0..2)
            .map(|index| ResidualBlock {
                norm: LayerNorm::new(16),
                layers: ModuleList::from_vec(vec![
                    Linear::with_seed(16, 32, true, 10 + index * 2),
                    Linear::with_seed(32, 16, true, 11 + index * 2),
                ]),
            })
            .collect(),
        activation: Activation::Relu(ReLU),
        head: TiedLinear::new(embed),
    };

    let descriptor = model.descriptor()?;
    println!("{}", descriptor.render());
    println!(
        "parameters: {} (the tied head shares embed.weight)",
        descriptor.parameter_count()
    );

    let mut registry = ParameterRegistry::new();
    model.visit_parameters(&ModulePath::root(), &mut registry)?;
    for (alias, canonical) in registry.aliases() {
        println!("alias: {alias} -> {canonical}");
    }

    let ids = Tensor::<f32>::from_vec([4], vec![1.0, 5.0, 9.0, 12.0])?;
    let logits = model.predict(ids)?;
    println!("logits: {:?}", logits.dims());

    let best = logits
        .data()
        .chunks(logits.dims()[1])
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap()
        })
        .collect::<Vec<_>>();
    println!("greedy next tokens: {best:?}");
    Ok(())
}
