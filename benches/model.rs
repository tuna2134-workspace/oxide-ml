//! Model-API throughput: compiled inference, direct `predict`, and a training
//! step for a 784-256-10 MLP.

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

fn main() {
    let batch = 64usize;
    let mut left = Vec::with_capacity(batch * 784);
    let mut right = Vec::with_capacity(batch * 10);
    let mut state = 0x1234_5678u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    };
    for _ in 0..batch * 784 {
        left.push(next());
    }
    for _ in 0..batch * 10 {
        right.push(next());
    }
    let model = MnistMlp {
        fc1: Linear::with_seed(784, 256, true, 1),
        fc2: Linear::with_seed(256, 10, true, 2),
    };
    let input = Tensor::<f32>::from_vec([batch, 784], left).unwrap();

    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [batch, 784])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    let store = compiled.parameter_store(&model).unwrap();
    let compiled_us = time(
        || {
            compiled.run(&store, &[("input0", &input)]).unwrap();
        },
        200,
    );
    let predict_us = time(
        || {
            model.predict(input.clone()).unwrap();
        },
        50,
    );

    let targets = Tensor::<f32>::from_vec([batch, 10], right).unwrap();
    let dataset = TensorDataset::from_tensors(input.clone(), targets).unwrap();
    let mut trainer = Trainer::builder()
        .optimizer(Sgd::new(0.01).unwrap())
        .loss(MSELoss::new())
        .batch_size(batch)
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

    println!("MLP 784-256-10, batch {batch}");
    println!("compiled inference : {compiled_us:9.1} us/pass");
    println!("predict (rebuild)  : {predict_us:9.1} us/pass");
    println!("train step         : {train_us:9.1} us/step (batch)");
    println!(
        "compiled graph     : {} nodes, {} values",
        compiled.report().nodes,
        compiled.report().values
    );
    println!(
        "memory plan        : {} bytes peak",
        compiled.report().memory_plan_bytes
    );
}
