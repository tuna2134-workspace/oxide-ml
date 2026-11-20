use oxide_ml::prelude::*;

#[derive(Module)]
struct Local {
    fc: Linear,
}

impl Local {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let layer = &self.fc;
        layer.forward(x)
    }
}

fn main() {}
