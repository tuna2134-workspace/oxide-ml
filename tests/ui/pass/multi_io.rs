//! Compile-pass fixture: tuple structs, multiple inputs and multiple outputs.

use oxide_ml::prelude::*;
use oxide_ml::{ModelContext, ValueId};

#[derive(Module)]
struct Pair(Linear, Linear);

impl Pair {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.1.forward(self.0.forward(x).gelu())
    }
}

struct Add;

impl Module for Add {
    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, _: &mut ModelContext, _: ValueId) -> Result<Vec<ValueId>> {
        Err(Error::invalid_graph("Add expects two inputs"))
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        Ok(vec![cx.builder().add(inputs[0], inputs[1])?])
    }
}

struct Split;

impl Module for Split {
    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, _: &mut ModelContext, _: ValueId) -> Result<Vec<ValueId>> {
        Err(Error::invalid_graph("Split is multi-output"))
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        let doubled = cx.builder().constant_scalar(2.0);
        Ok(vec![inputs[0], cx.builder().mul(inputs[0], doubled)?])
    }
}

#[derive(Module)]
struct MultiIo {
    add: Add,
    split: Split,
    pair: Pair,
}

impl MultiIo {
    #[model]
    fn forward(&self, a: Tensor<f32>, b: Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
        let sum = self.add.forward(a, b);
        let (first, second) = self.split.forward(sum);
        let projected = self.pair.forward(first);
        (projected, second)
    }
}

fn main() {
    let model = MultiIo {
        add: Add,
        split: Split,
        pair: Pair(
            Linear::with_seed(4, 4, true, 1),
            Linear::with_seed(4, 4, true, 2),
        ),
    };
    let a = Tensor::<f32>::from_vec([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::<f32>::from_vec([1, 4], vec![4.0, 3.0, 2.0, 1.0]).unwrap();
    let (first, second) = model.predict(a, b).unwrap();
    assert_eq!(first.dims(), vec![1, 4]);
    assert_eq!(second.dims(), vec![1, 4]);
}
