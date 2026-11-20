//! MobileNetV4 construction and a small end-to-end forward/training step.

use oxide_ml::prelude::*;

#[test]
fn mobilenetv4_forward_shapes_and_parameter_count() {
    let config = MobileNetV4Config::nano(2);
    let model = MobileNetV4::new(&config).unwrap();
    assert!(model.parameter_count().unwrap() > 10_000);
    let input = Tensor::<f32>::zeros([2, 3, 64, 64]);
    let logits = model.predict(input).unwrap();
    assert_eq!(logits.dims(), vec![2, 2]);
}

#[test]
fn mobilenetv4_accepts_a_training_step() {
    let config = MobileNetV4Config::nano(2);
    let model = MobileNetV4::new(&config).unwrap();
    let compiled = CompiledModel::compile_training(
        &model,
        &[InputSpec::unbatched("input0", [2, 3, 48, 48])],
        InputSpec::unbatched("target0", [2]),
        &CrossEntropyLoss::new(),
        ModelCompileOptions::training(),
    )
    .unwrap();
    let mut store = compiled.parameter_store(&model).unwrap();
    let images = Tensor::<f32>::zeros([2, 3, 48, 48]);
    let targets = Tensor::<f32>::from_vec([2], vec![0.0, 1.0]).unwrap();
    let output = compiled
        .run_training(&store, &[("input0", &images), ("target0", &targets)])
        .unwrap();
    assert!(output.loss.is_finite());
    let mut optimizer = AdamW::new(1e-3).unwrap();
    optimizer.step(&mut store, &output.gradients).unwrap();
    assert!(!store.is_empty());
}
