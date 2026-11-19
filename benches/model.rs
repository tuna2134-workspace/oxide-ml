//! Model-API throughput: compile-cache behaviour, compiled inference, batch
//! scaling and a training step for a 784-256-10 MLP.

use std::time::Instant;

use oxide_ml::prelude::*;

#[derive(Module)]
struct MnistMlp {
    fc1: Linear,
    fc2: Linear,
}

impl MnistMlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = relu(self.fc1.forward(x));
        self.fc2.forward(hidden)
    }
}

fn time<F: FnMut()>(mut f: F, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed().as_secs_f64() * 1e6 / iterations as f64
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

fn main() {
    let model = MnistMlp {
        fc1: Linear::with_seed(784, 256, true, 1),
        fc2: Linear::with_seed(256, 10, true, 2),
    };

    // ---- Batch scaling for compiled inference --------------------------------
    println!("compiled inference (MLP 784-256-10)");
    println!("{:>6}  {:>12}", "batch", "us/pass");
    for batch in [1usize, 16, 64, 256] {
        let input = deterministic(batch, 784, 7);
        let compiled = CompiledModel::compile(
            &model,
            &[InputSpec::unbatched("input0", [batch, 784])],
            ModelCompileOptions::default(),
        )
        .unwrap();
        let store = compiled.parameter_store(&model).unwrap();
        let iterations = if batch <= 64 { 100 } else { 20 };
        let us = time(
            || {
                compiled.run(&store, &[("input0", &input)]).unwrap();
            },
            iterations,
        );
        println!("{batch:>6}  {us:>12.1}");
    }

    // ---- Compile cache --------------------------------------------------------
    let input = deterministic(64, 784, 11);
    let specs = [InputSpec::unbatched("input0", [64, 784])];
    let cache = CompileCache::new(4);
    let start = Instant::now();
    let _ = compile_cached_with(&cache, &model, &specs, &ModelCompileOptions::default()).unwrap();
    let cold_compile = start.elapsed().as_secs_f64() * 1e6;
    let warm_compile = time(
        || {
            let _ = compile_cached_with(&cache, &model, &specs, &ModelCompileOptions::default())
                .unwrap();
        },
        500,
    );
    let cold = time(
        || {
            let _ = predict_many_with_cache(&model, vec![input.clone()], &cache).unwrap();
        },
        1,
    );
    let warm = time(
        || {
            let _ = predict_many_with_cache(&model, vec![input.clone()], &cache).unwrap();
        },
        100,
    );
    let stats = cache.stats();
    println!("\npredict (batch 64)");
    println!("compile cold    : {cold_compile:9.1} us");
    println!("compile cached  : {warm_compile:9.1} us");
    println!("predict cold    : {cold:9.1} us");
    println!("predict cached  : {warm:9.1} us");
    println!(
        "cache           : entries={} hits={} misses={} compilations={}",
        stats.entries, stats.hits, stats.misses, stats.compilations
    );

    // ---- Compiler report ------------------------------------------------------
    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [64, 784])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    let report = compiled.report();
    println!("\ncompiler report");
    println!(
        "nodes={} values={} kernels={} fusions={} compile={} us peak={} B",
        report.nodes,
        report.values,
        report.kernels,
        report.fusions,
        report.compile_time_micros,
        report.memory_plan_bytes
    );
    let store = compiled.parameter_store(&model).unwrap();
    let compiled_us = time(
        || {
            compiled.run(&store, &[("input0", &input)]).unwrap();
        },
        200,
    );
    println!("compiled run    : {compiled_us:9.1} us/pass");

    // ---- Training -------------------------------------------------------------
    let targets = deterministic(64, 10, 13);
    let dataset = TensorDataset::from_tensors(input.clone(), targets).unwrap();
    let mut trainer = Trainer::builder()
        .optimizer(Sgd::new(0.01).unwrap())
        .loss(MSELoss::new())
        .batch_size(64)
        .build()
        .unwrap();
    let mut train_model = MnistMlp {
        fc1: Linear::with_seed(784, 256, true, 1),
        fc2: Linear::with_seed(256, 10, true, 2),
    };
    let train_us = time(
        || {
            trainer.fit(&mut train_model, &dataset, 1).unwrap();
        },
        5,
    );
    println!("\ntrain step      : {train_us:9.1} us/batch");
}
