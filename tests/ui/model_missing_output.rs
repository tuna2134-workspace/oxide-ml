use oxide_ml::prelude::*;

#[derive(Module)]
struct NoOutput {
    fc: Linear,
}

impl NoOutput {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = self.fc.forward(x);
        let _ = hidden;
    }
}

fn main() {}
