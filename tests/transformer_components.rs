//! Reusable transformer components: norms, gated MLPs, RoPE, masks and GQA.

use oxide_ml::prelude::*;
use oxide_ml::{ModelContext, ValueId, causal_mask, grouped_query_attention, rotate_half_value};

#[test]
fn rms_norm_matches_the_definition() {
    let model = RmsNorm::with_dim(4, 1e-6);
    let input = Tensor::<f32>::from_vec([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = oxide_ml::predict_single(&model, input.clone()).unwrap();
    let mean_square = input.data().iter().map(|value| value * value).sum::<f32>() / 4.0;
    let denominator = (mean_square + 1e-6).sqrt();
    for (actual, expected) in output.data().iter().zip(input.data()) {
        assert!(
            (actual - expected / denominator).abs() < 1e-6,
            "{actual} vs {}",
            expected / denominator
        );
    }
}

#[test]
fn no_scale_rms_norm_has_no_parameters() {
    let model = RmsNormNoScale::new(1e-6);
    assert_eq!(model.descriptor().unwrap().parameter_count(), 0);
    let input = Tensor::<f32>::from_vec([1, 2], vec![3.0, 4.0]).unwrap();
    let output = oxide_ml::predict_single(&model, input).unwrap();
    // x / sqrt(mean(x^2) + eps)
    let denominator = ((9.0 + 16.0) / 2.0_f32 + 1e-6).sqrt();
    assert!((output.data()[0] - 3.0 / denominator).abs() < 1e-6);
    assert!((output.data()[1] - 4.0 / denominator).abs() < 1e-6);
}

#[derive(Module)]
struct ManualGeglu {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl ManualGeglu {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let gate = self.gate.forward(x).gelu();
        let up = self.up.forward(x);
        self.down.forward(gate * up)
    }
}

#[test]
fn gated_mlp_activations_match_explicit_composition() {
    let gate = Linear::with_seed(4, 8, true, 1);
    let up = Linear::with_seed(4, 8, true, 2);
    let down = Linear::with_seed(8, 4, true, 3);
    let input =
        Tensor::<f32>::from_vec([2, 4], vec![0.5, -0.25, 0.1, 0.7, -0.3, 0.2, 0.4, -0.6]).unwrap();

    let manual = ManualGeglu {
        gate: gate.clone(),
        up: up.clone(),
        down: down.clone(),
    };
    let component = GatedMlp::geglu(gate.clone(), up.clone(), down.clone());
    let expected = oxide_ml::predict_single(&manual, input.clone()).unwrap();
    let actual = oxide_ml::predict_single(&component, input.clone()).unwrap();
    for (a, b) in actual.data().iter().zip(expected.data()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    let swiglu = GatedMlp::new(gate, up, down, GatedActivation::SwiGlu);
    let output = oxide_ml::predict_single(&swiglu, input).unwrap();
    assert_eq!(output.dims(), vec![2, 4]);
    assert_eq!(
        component.descriptor().unwrap().parameter_count(),
        manual.descriptor().unwrap().parameter_count()
    );
}

#[test]
fn rope_tables_are_identity_at_position_zero_and_for_unrotated_tails() {
    let config = RopeConfig::new(8, 10_000.0);
    let (cos, sin) = config.tables(0, 2);
    assert_eq!(cos.len(), 16);
    assert!(cos[..8].iter().all(|value| (value - 1.0).abs() < 1e-6));
    assert!(sin[..8].iter().all(|value| value.abs() < 1e-6));

    let proportional = RopeConfig::proportional(16, 1_000_000.0, 0.25);
    let (cos, sin) = proportional.tables(1, 1);
    for index in 2..8 {
        assert!((cos[index] - 1.0).abs() < 1e-6);
        assert!(sin[index].abs() < 1e-6);
        assert!((cos[8 + index] - 1.0).abs() < 1e-6);
    }
    assert!(cos[..2].iter().any(|value| (*value - 1.0).abs() > 1e-6));
}

#[test]
fn causal_and_sliding_masks_mask_the_right_positions() {
    let causal = causal_mask(4, None);
    assert_eq!(causal[0], 0.0);
    assert!(causal[1] < -1e20);
    assert_eq!(causal[5], 0.0);

    let sliding = causal_mask(4, Some(2));
    assert_eq!(sliding[2 * 4 + 1], 0.0);
    assert_eq!(sliding[4], 0.0);
    assert!(sliding[8] < -1e20, "distance 2 outside a window of 2");
}

#[test]
fn rotate_half_moves_and_negates_the_second_half() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([1, 4]), DType::F32);
    let rotated = rotate_half_value(&mut builder, x, 4, 1).unwrap();
    let graph = builder.finish(&[rotated]).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        x,
        DenseTensor::from_vec([1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    );
    let result = Executor::<f32>::new(graph)
        .unwrap()
        .run(&ParameterStore::new(), &inputs)
        .unwrap();
    assert_eq!(result.output(0).unwrap().data(), &[-3.0, -4.0, 1.0, 2.0]);
}

struct Gqa {
    config: GqaConfig,
}

impl Module for Gqa {
    fn module_type(&self) -> &'static str {
        "GqaTest"
    }

    fn visit_parameters(&self, _: &ModulePath, _: &mut ParameterRegistry) -> Result<()> {
        Ok(())
    }

    fn forward(&self, _: &mut ModelContext, _: ValueId) -> Result<Vec<ValueId>> {
        Err(Error::invalid_graph("Gqa needs three inputs"))
    }

    fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>> {
        Ok(vec![grouped_query_attention(
            cx,
            inputs[0],
            inputs[1],
            inputs[2],
            None,
            None,
            self.config,
        )?])
    }
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values.iter_mut() {
        *value /= sum;
    }
}

#[test]
fn grouped_query_attention_matches_a_direct_computation() {
    // One query head, one KV head, two positions, head_dim two.
    let query = [0.5f32, -0.25, 1.0, 0.3];
    let key = [0.2f32, 0.4, -0.1, 0.6];
    let value = [1.0f32, 2.0, 3.0, 4.0];
    let config = GqaConfig::new(1, 1, 2).unwrap();
    let model = Gqa { config };

    let q = Tensor::<f32>::from_vec([1, 2, 1, 2], query.to_vec()).unwrap();
    let k = Tensor::<f32>::from_vec([1, 2, 1, 2], key.to_vec()).unwrap();
    let v = Tensor::<f32>::from_vec([1, 2, 1, 2], value.to_vec()).unwrap();
    let compiled = CompiledModel::compile(
        &model,
        &[
            InputSpec::unbatched("q", [1, 2, 1, 2]),
            InputSpec::unbatched("k", [1, 2, 1, 2]),
            InputSpec::unbatched("v", [1, 2, 1, 2]),
        ],
        ModelCompileOptions::default(),
    )
    .unwrap();
    let store = compiled.parameter_store(&model).unwrap();
    let outputs = compiled
        .run(&store, &[("q", &q), ("k", &k), ("v", &v)])
        .unwrap();
    let output = &outputs[0];
    assert_eq!(output.dims(), vec![1, 2, 2]);

    let scale = 1.0 / (2.0f32).sqrt();
    let mut expected = vec![0.0f32; 4];
    for position in 0..2 {
        let mut scores = vec![0.0f32; 2];
        for other in 0..2 {
            scores[other] = (query[position * 2] * key[other * 2]
                + query[position * 2 + 1] * key[other * 2 + 1])
                * scale;
        }
        softmax_in_place(&mut scores);
        for dim in 0..2 {
            expected[position * 2 + dim] = scores[0] * value[dim] + scores[1] * value[2 + dim];
        }
    }
    for (actual, expected) in output.data().iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
    }
}

#[test]
fn gqa_config_rejects_unbalanced_head_counts() {
    assert!(GqaConfig::new(4, 3, 8).is_err());
    assert!(GqaConfig::new(0, 1, 8).is_err());
    assert_eq!(GqaConfig::new(8, 2, 8).unwrap().groups(), 4);
}
