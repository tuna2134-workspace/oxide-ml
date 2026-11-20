//! Module-system benchmarks: construction, parameter traversal, module
//! collections, compiled inference and KV-cache state operations.

use std::time::Instant;

use oxide_ml::prelude::*;

#[derive(Module)]
struct Block {
    norm: LayerNorm,
    layers: ModuleList<Linear>,
}

#[derive(Module)]
struct BenchModel {
    blocks: ModuleList<Block>,
    head: Linear,
}

impl BenchModel {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let mut hidden = x;
        for block in &self.blocks {
            let residual = hidden.clone();
            for layer in &block.layers {
                hidden = layer.forward(hidden);
            }
            hidden = block.norm.forward(hidden + residual);
        }
        self.head.forward(hidden)
    }
}

fn build(blocks: usize, width: usize) -> BenchModel {
    BenchModel {
        blocks: (0..blocks)
            .map(|block| Block {
                norm: LayerNorm::new(width),
                layers: ModuleList::from_vec(vec![
                    Linear::with_seed(width, width, true, block as u64 * 2 + 1),
                    Linear::with_seed(width, width, true, block as u64 * 2 + 2),
                ]),
            })
            .collect(),
        head: Linear::with_seed(width, width, true, 999),
    }
}

fn deterministic(rows: usize, cols: usize, seed: u64) -> Tensor<f32> {
    let mut state = seed | 1;
    let mut values = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(((state >> 40) as f32 / (1u32 << 24) as f32) - 0.5);
    }
    Tensor::from_vec([rows, cols], values).unwrap()
}

fn time<F: FnMut()>(mut f: F, iterations: usize) -> f64 {
    // Warm up so the first allocation does not dominate.
    f();
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed().as_secs_f64() * 1e6 / iterations as f64
}

fn main() {
    let width = 128;
    let blocks = 8;
    let parameters = blocks * 2 * 3 + 2; // weight+bias per Linear, gamma+beta per norm, head

    // ---- Construction -------------------------------------------------------
    let construct = time(
        || {
            let model = build(blocks, width);
            std::hint::black_box(model);
        },
        200,
    );
    println!("module construction: {construct:.1} us/model ({blocks} blocks, width {width})");

    let model = build(blocks, width);
    let construct_list = time(
        || {
            let list: ModuleList<Linear> = (0..16)
                .map(|index| Linear::with_seed(8, 8, true, index))
                .collect();
            std::hint::black_box(list);
        },
        500,
    );
    println!("module list build (16 x 8x8 Linear): {construct_list:.1} us");

    // ---- Parameter traversal ------------------------------------------------
    let traverse = time(
        || {
            let mut registry = ParameterRegistry::new();
            model
                .visit_parameters(&ModulePath::root(), &mut registry)
                .unwrap();
            assert_eq!(registry.specs().len(), parameters);
            std::hint::black_box(registry);
        },
        500,
    );
    println!(
        "parameter discovery: {traverse:.1} us/model ({parameters} parameters, {:.1} M/s)",
        parameters as f64 / traverse
    );

    let mutate = time(
        || {
            let mut model = build(blocks, width);
            model.set_training(false);
            std::hint::black_box(model);
        },
        200,
    );
    println!("build + set_training propagation: {mutate:.1} us/model");

    // ---- Compiled inference over module collections -------------------------
    let batch = 8;
    let input = deterministic(batch, width, 17);
    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [batch, width])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    let store = compiled.parameter_store(&model).unwrap();
    let run = time(
        || {
            let output = compiled.run(&store, &[("input0", &input)]).unwrap();
            std::hint::black_box(output);
        },
        50,
    );
    println!("compiled forward: {run:.1} us/pass (batch {batch}, {blocks} blocks, width {width})");

    let compile = time(
        || {
            let compiled = CompiledModel::compile(
                &model,
                &[InputSpec::unbatched("input0", [batch, width])],
                ModelCompileOptions::default(),
            )
            .unwrap();
            std::hint::black_box(compiled);
        },
        100,
    );
    println!("graph compile (cold): {compile:.1} us");

    // ---- KV cache state -----------------------------------------------------
    let layers = 32;
    let kv_heads = 8;
    let head_dim = 128;
    let tokens = 256;
    let row = kv_heads * head_dim;
    let mut cache =
        KvCache::new(KvCacheConfig::uniform(layers, kv_heads, head_dim, tokens)).unwrap();
    let keys: Vec<f32> = (0..row).map(|value| value as f32 * 1e-3).collect();
    let append = time(
        || {
            cache.reset();
            for _ in 0..tokens {
                for layer in 0..layers {
                    cache.append(layer, &keys, &keys).unwrap();
                }
            }
            std::hint::black_box(&cache);
        },
        20,
    );
    let bytes = (layers * tokens * row * 2 * std::mem::size_of::<f32>()) as f64;
    println!(
        "kv cache append: {append:.1} us ({tokens} tokens x {layers} layers, {:.1} GB/s)",
        bytes / append / 1e3
    );

    let read = time(
        || {
            let mut checksum = 0.0f32;
            for layer in 0..layers {
                checksum += cache.keys(layer).unwrap().len() as f32;
                checksum += cache.values(layer).unwrap()[0];
            }
            std::hint::black_box(checksum);
        },
        2000,
    );
    println!("kv cache slice lookup ({layers} layers): {read:.2} us");
    println!(
        "kv cache footprint: {:.1} MiB ({} layers, {}x{} heads/dim, {} tokens)",
        bytes / (1024.0 * 1024.0),
        layers,
        kv_heads,
        head_dim,
        tokens
    );
}
