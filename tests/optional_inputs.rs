//! Optional graph inputs: `Option<Tensor<f32>>`, `forward_slots` and module
//! calls that thread optional state (the framework extension used by Gemma).

use oxide_ml::prelude::*;
use oxide_ml::{ModelContext, ValueId, module_forward_slots};

#[derive(Module)]
#[oxide(slots)]
struct Conditional {
    scale: Linear,
}

impl Conditional {
    #[model]
    fn forward(&self, x: Tensor<f32>, extra: Option<Tensor<f32>>) -> Tensor<f32> {
        if let Some(extra) = extra {
            self.scale.forward(x) + extra
        } else {
            self.scale.forward(x)
        }
    }
}

/// Wrapper that supplies `None` for the optional input.
struct WithoutExtra {
    inner: Conditional,
}

impl Module for WithoutExtra {
    fn visit_parameters(&self, path: &ModulePath, registry: &mut ParameterRegistry) -> Result<()> {
        self.inner
            .visit_parameters(&ModulePath::root().child("inner"), registry)?;
        let _ = path;
        Ok(())
    }

    fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
        module_forward_slots(cx, "inner", &self.inner, &[Some(input), None])
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
    ) -> Result<()> {
        self.inner.visit_tensors(&path.child("inner"), visitor)
    }
}

fn model() -> Conditional {
    Conditional {
        scale: Linear::with_seed(2, 2, false, 5),
    }
}

#[test]
fn optional_input_can_be_absent() {
    let model = model();
    let input = Tensor::<f32>::from_vec([1, 2], vec![1.0, 2.0]).unwrap();
    let expected =
        oxide_ml::predict_single(&Linear::with_seed(2, 2, false, 5), input.clone()).unwrap();

    let wrapper = WithoutExtra { inner: model };
    let compiled = CompiledModel::compile(
        &wrapper,
        &[InputSpec::unbatched("input0", [1, 2])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    let store = compiled.parameter_store(&wrapper).unwrap();
    let output = compiled.run(&store, &[("input0", &input)]).unwrap();
    assert_eq!(output[0].data(), expected.data());
}

#[test]
fn optional_input_can_be_present_through_a_declarative_call() {
    #[derive(Module)]
    struct Outer {
        inner: Conditional,
    }

    impl Outer {
        #[model]
        fn forward(&self, x: Tensor<f32>, extra: Option<Tensor<f32>>) -> Tensor<f32> {
            self.inner.forward_slots(Some(x), extra)
        }
    }

    let outer = Outer { inner: model() };
    let input = Tensor::<f32>::from_vec([1, 2], vec![1.0, 2.0]).unwrap();
    let extra = Tensor::<f32>::from_vec([1, 2], vec![0.5, -0.5]).unwrap();
    let compiled = CompiledModel::compile(
        &outer,
        &[
            InputSpec::unbatched("input0", [1, 2]),
            InputSpec::unbatched("input1", [1, 2]),
        ],
        ModelCompileOptions::default(),
    )
    .unwrap();
    let store = compiled.parameter_store(&outer).unwrap();
    let output = compiled
        .run(&store, &[("input0", &input), ("input1", &extra)])
        .unwrap();

    let scaled = oxide_ml::predict_single(&Linear::with_seed(2, 2, false, 5), input).unwrap();
    // Extra is added to the first row element accordingly.
    assert!((output[0].data()[0] - (scaled.data()[0] + 0.5)).abs() < 1e-6);
    assert!((output[0].data()[1] - (scaled.data()[1] - 0.5)).abs() < 1e-6);
}
