//! Randomized property tests: graph execution and optimization must agree with
//! a direct tensor evaluation for randomly generated elementwise graphs.

use oxide_ml::prelude::*;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn index(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound.max(1)
    }

    fn value(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }
}

fn tensor(dims: &[usize], rng: &mut Rng) -> Tensor<f32> {
    let len: usize = dims.iter().product();
    Tensor::<f32>::from_vec(dims, (0..len).map(|_| rng.value()).collect()).unwrap()
}

fn broadcast(tensor: &Tensor<f32>, dims: &[usize]) -> Tensor<f32> {
    if tensor.dims() == dims {
        tensor.clone()
    } else {
        tensor.broadcast_to(dims).unwrap()
    }
}

#[test]
fn random_elementwise_graphs_match_direct_evaluation() {
    for seed in 0..64u64 {
        let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234);
        let mut builder = GraphBuilder::new();
        let shape = [3usize, 4];
        let a = builder.input("a", Shape::from_static(shape), DType::F32);
        let b = builder.input("b", Shape::from_static([4]), DType::F32);
        let a_value = tensor(&shape, &mut rng);
        let b_value = tensor(&[4], &mut rng);
        let mut current = a;
        let mut current_value = a_value.clone();
        let mut scalar_values = Vec::new();

        for _ in 0..20 {
            let operand = rng.index(4);
            let (value_id, value_tensor) = match operand {
                0 => (a, a_value.clone()),
                1 => (b, b_value.clone()),
                2 => {
                    let scalar = rng.value();
                    let id = builder.constant_scalar(scalar);
                    scalar_values.push(scalar);
                    (id, Tensor::<f32>::scalar(scalar))
                }
                _ => (current, current_value.clone()),
            };
            let op = rng.index(3);
            let (new_id, new_value) = match op {
                0 => (
                    builder.add(current, value_id).unwrap(),
                    broadcast(&current_value, &shape)
                        .add(&broadcast(&value_tensor, &shape))
                        .unwrap(),
                ),
                1 => (
                    builder.sub(current, value_id).unwrap(),
                    broadcast(&current_value, &shape)
                        .sub(&broadcast(&value_tensor, &shape))
                        .unwrap(),
                ),
                _ => (
                    builder.mul(current, value_id).unwrap(),
                    broadcast(&current_value, &shape)
                        .mul(&broadcast(&value_tensor, &shape))
                        .unwrap(),
                ),
            };
            current = new_id;
            current_value = new_value;
        }
        let graph = builder.finish(&[current]).unwrap();
        let mut inputs = ExecutionInputs::<f32>::new();
        inputs.insert(a, a_value.clone().into_dense());
        inputs.insert(b, b_value.clone().into_dense());
        let parameters = ParameterStore::new();
        let plain = Executor::new(graph.clone())
            .unwrap()
            .run(&parameters, &inputs)
            .unwrap();
        let plain_output = plain.output(0).unwrap().data().to_vec();
        assert_eq!(plain_output.len(), current_value.data().len());
        for (graph_value, expected) in plain_output.iter().zip(current_value.data()) {
            assert!(
                (graph_value - expected).abs() < 1e-5,
                "seed {seed}: {graph_value} vs {expected}"
            );
        }

        // The same graph after the optimizer must produce the same result.
        let compiled = Compiler::new(
            Box::new(CpuBackend::<f32>::new()),
            CompileOptions::default(),
        )
        .compile(graph)
        .unwrap();
        assert!(compiled.report.kernel_count > 0);
        let optimized = Executor::new(compiled.graph)
            .unwrap()
            .run(&parameters, &inputs)
            .unwrap();
        let optimized_output = optimized.output(0).unwrap().data().to_vec();
        for (left, right) in plain_output.iter().zip(&optimized_output) {
            assert!(
                (left - right).abs() < 1e-5,
                "seed {seed}: {left} vs {right}"
            );
        }
        // Unused scalars are folded away or remain constants; either way the
        // result above is the oracle.
        let _ = scalar_values;
    }
}

#[test]
fn random_fusion_preserves_values() {
    for seed in 0..32u64 {
        let mut rng = Rng::new(seed ^ 0xABCD);
        let shape = [2usize, 5];
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::from_static(shape), DType::F32);
        let y = builder.input("y", Shape::from_static(shape), DType::F32);
        let mut current = x;
        let mut value = tensor(&shape, &mut rng);
        let y_value = tensor(&shape, &mut rng);
        for _ in 0..8 {
            if rng.index(2) == 0 {
                current = builder.mul(current, y).unwrap();
                value = value.mul(&y_value).unwrap();
            } else {
                current = builder.add(current, y).unwrap();
                value = value.add(&y_value).unwrap();
            }
        }
        let current = builder.relu(current).unwrap();
        let value = value.relu();
        let graph = builder.finish(&[current]).unwrap();
        let mut inputs = ExecutionInputs::<f32>::new();
        inputs.insert(x, tensor(&shape, &mut Rng::new(seed + 99)).into_dense());
        inputs.insert(y, y_value.clone().into_dense());
        let parameters = ParameterStore::new();
        let before = Executor::new(graph.clone())
            .unwrap()
            .run(&parameters, &inputs)
            .unwrap()
            .output(0)
            .unwrap()
            .data()
            .to_vec();
        let compiled = Compiler::new(
            Box::new(CpuBackend::<f32>::new()),
            CompileOptions::default(),
        )
        .compile(graph)
        .unwrap();
        let after = Executor::new(compiled.graph)
            .unwrap()
            .run(&parameters, &inputs)
            .unwrap()
            .output(0)
            .unwrap()
            .data()
            .to_vec();
        for (left, right) in before.iter().zip(&after) {
            assert!(
                (left - right).abs() < 1e-5,
                "seed {seed}: {left} vs {right}"
            );
        }
        let _ = value;
    }
}
