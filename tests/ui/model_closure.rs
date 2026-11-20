use oxide_ml::prelude::*;

#[derive(Module)]
struct Closure {
    fc: Linear,
}

impl Closure {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let apply = |value| value;
        apply(x)
    }
}

fn main() {}
