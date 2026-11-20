use oxide_ml::prelude::*;

#[derive(Module)]
struct Conflicting {
    #[module(skip)]
    #[param]
    weight: DenseTensor<f32>,
}

fn main() {}
