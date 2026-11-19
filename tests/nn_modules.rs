//! Numerical and shape tests for the model-API neural-network modules.

use oxide_ml::prelude::*;

#[derive(Module)]
struct One {
    layer: Linear,
}
impl One {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.layer.forward(x)
    }
}

#[derive(Module)]
struct Stack {
    first: Linear,
    second: Linear,
}
impl Stack {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let x = relu(self.first.forward(x));
        self.second.forward(x)
    }
}

#[derive(Module)]
struct Normed {
    norm: LayerNorm,
}
impl Normed {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.norm.forward(x)
    }
}

#[derive(Module)]
struct Batched {
    norm: BatchNorm,
}
impl Batched {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.norm.forward(x)
    }
}

#[derive(Module)]
struct Grouped {
    norm: GroupNorm,
}
impl Grouped {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.norm.forward(x)
    }
}

#[derive(Module)]
struct Instanced {
    norm: InstanceNorm,
}
impl Instanced {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.norm.forward(x)
    }
}

#[derive(Module)]
struct Conv {
    conv: Conv2d,
}
impl Conv {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.conv.forward(x)
    }
}

#[derive(Module)]
struct ConvOne {
    conv: Conv1d,
}
impl ConvOne {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.conv.forward(x)
    }
}

#[derive(Module)]
struct Pools {
    max: MaxPool2d,
    avg: AvgPool2d,
}
impl Pools {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.avg.forward(self.max.forward(x))
    }
}

#[derive(Module)]
struct Embed {
    embedding: Embedding,
}
impl Embed {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.embedding.forward(x)
    }
}

#[derive(Module)]
struct Attn {
    attention: Attention,
}
impl Attn {
    #[model]
    fn forward(&self, q: Tensor<f32>, k: Tensor<f32>, v: Tensor<f32>) -> Tensor<f32> {
        self.attention.forward_multi(q, k, v)
    }
}

#[derive(Module)]
struct Mha {
    mha: MultiHeadAttention,
}
impl Mha {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.mha.forward(x)
    }
}

#[derive(Module)]
struct Sequence {
    rnn: Rnn,
}
impl Sequence {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.rnn.forward(x)
    }
}

#[derive(Module)]
struct SequenceGru {
    gru: Gru,
}
impl SequenceGru {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.gru.forward(x)
    }
}

#[derive(Module)]
struct SequenceLstm {
    lstm: Lstm,
}
impl SequenceLstm {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        self.lstm.forward(x)
    }
}

fn predict_single<M: Module>(model: &M, dims: Vec<usize>, data: Vec<f32>) -> Tensor<f32> {
    let input = Tensor::<f32>::from_vec(&dims, data).unwrap();
    oxide_ml::predict_single(model, input).unwrap()
}

#[test]
fn linear_is_affine() {
    let mut linear = Linear::with_seed(2, 3, true, 1);
    linear.weight = Tensor::<f32>::from_vec([3, 2], vec![1., 0., 0., 1., 1., 1.])
        .unwrap()
        .into_dense();
    linear.bias = Some(
        Tensor::<f32>::from_vec([3], vec![0.5, -0.5, 0.0])
            .unwrap()
            .into_dense(),
    );
    let model = One { layer: linear };
    let output = predict_single(&model, vec![1, 2], vec![2., 3.]);
    assert_eq!(output.dims(), vec![1, 3]);
    assert!((output.data()[0] - 2.5).abs() < 1e-6);
    assert!((output.data()[1] - 2.5).abs() < 1e-6);
    assert!((output.data()[2] - 5.0).abs() < 1e-6);
}

#[test]
fn stack_matches_reference() {
    let mut first = Linear::with_seed(2, 2, true, 2);
    first.weight = Tensor::<f32>::from_vec([2, 2], vec![1., -1., -1., 1.])
        .unwrap()
        .into_dense();
    first.bias = Some(
        Tensor::<f32>::from_vec([2], vec![0., 0.])
            .unwrap()
            .into_dense(),
    );
    let mut second = Linear::with_seed(2, 1, true, 3);
    second.weight = Tensor::<f32>::from_vec([1, 2], vec![1., 2.])
        .unwrap()
        .into_dense();
    second.bias = Some(
        Tensor::<f32>::from_vec([1], vec![0.5])
            .unwrap()
            .into_dense(),
    );
    let model = Stack { first, second };
    let output = predict_single(&model, vec![1, 2], vec![2., 3.]);
    // relu([-1, 1]) = [0,1]; 1*0 + 2*1 + 0.5 = 2.5
    assert!((output.data()[0] - 2.5).abs() < 1e-6);
}

#[test]
fn layer_norm_normalizes_rows() {
    let mut norm = LayerNorm::new(2);
    norm.gamma = Tensor::<f32>::from_vec([2], vec![1., 1.])
        .unwrap()
        .into_dense();
    norm.beta = Tensor::<f32>::from_vec([2], vec![0., 0.])
        .unwrap()
        .into_dense();
    let model = Normed { norm };
    let output = predict_single(&model, vec![1, 2], vec![1., 3.]);
    assert!((output.data()[0] + 1.0).abs() < 1e-4, "{:?}", output.data());
    assert!((output.data()[1] - 1.0).abs() < 1e-4, "{:?}", output.data());
}

#[test]
fn batch_norm_uses_running_stats_in_eval_and_batch_stats_in_training() {
    let model = Batched {
        norm: BatchNorm::new(2),
    };
    // Eval mode uses running mean 0 / var 1, so the output equals the input.
    let eval = predict_single(&model, vec![2, 2], vec![1., 1., 3., 3.]);
    assert_eq!(eval.dims(), vec![2, 2]);
    assert!((eval.data()[0] - 1.0).abs() < 1e-3, "{:?}", eval.data());

    // Training mode uses batch statistics, so each channel is centered.
    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [2, 2])],
        ModelCompileOptions::training(),
    )
    .unwrap();
    let store = compiled.parameter_store(&model).unwrap();
    let input = Tensor::<f32>::from_vec([2, 2], vec![1., 1., 3., 3.]).unwrap();
    let out = compiled.run(&store, &[("input0", &input)]).unwrap();
    let mean: f32 = out[0].data().iter().sum::<f32>() / out[0].data().len() as f32;
    assert!(mean.abs() < 1e-4, "mean {mean}");
}

#[test]
fn group_norm_and_instance_norm_preserve_shape() {
    let grouped = Grouped {
        norm: GroupNorm::new(2, 4).unwrap(),
    };
    let out = predict_single(&grouped, vec![2, 4, 3], (0..24).map(|i| i as f32).collect());
    assert_eq!(out.dims(), vec![2, 4, 3]);
    let instanced = Instanced {
        norm: InstanceNorm::new(4).unwrap(),
    };
    let out = predict_single(
        &instanced,
        vec![2, 4, 3],
        (0..24).map(|i| i as f32).collect(),
    );
    assert_eq!(out.dims(), vec![2, 4, 3]);
}

#[test]
fn conv2d_computes_reference_values() {
    let mut conv = Conv2d::new(1, 1, [2, 2], [1, 1], [0, 0]);
    conv.weight = Tensor::<f32>::from_vec([1, 1, 2, 2], vec![1., 0., 0., 1.])
        .unwrap()
        .into_dense();
    conv.bias = Some(Tensor::<f32>::from_vec([1], vec![0.]).unwrap().into_dense());
    let model = Conv { conv };
    // 1x1x3x3 input: diagonal kernel sums the diagonal of each 2x2 patch.
    let output = predict_single(
        &model,
        vec![1, 1, 3, 3],
        vec![1., 2., 3., 4., 5., 6., 7., 8., 9.],
    );
    assert_eq!(output.dims(), vec![1, 1, 2, 2]);
    assert_eq!(output.data(), &[6., 8., 12., 14.]);
}

#[test]
fn conv1d_matches_manual_convolution() {
    let mut conv = Conv1d::new(1, 1, 3, 1, 0);
    conv.weight = Tensor::<f32>::from_vec([1, 1, 3], vec![1., 2., 3.])
        .unwrap()
        .into_dense();
    conv.bias = Some(Tensor::<f32>::from_vec([1], vec![0.]).unwrap().into_dense());
    let model = ConvOne { conv };
    let output = predict_single(&model, vec![1, 1, 4], vec![1., 2., 3., 4.]);
    assert_eq!(output.dims(), vec![1, 1, 2]);
    assert_eq!(output.data(), &[14., 20.]);
}

#[test]
fn pooling_halves_spatial_size() {
    let model = Pools {
        max: MaxPool2d::new([2, 2], [2, 2]),
        avg: AvgPool2d::new([2, 2], [2, 2]),
    };
    let output = predict_single(
        &model,
        vec![1, 1, 4, 4],
        (1..=16).map(|value| value as f32).collect(),
    );
    assert_eq!(output.dims(), vec![1, 1, 1, 1]);
    // max pool gives [[6, 8], [14, 16]], average = 11.
    assert_eq!(output.data(), &[11.]);
}

#[test]
fn embedding_looks_up_rows() {
    let mut embedding = Embedding::with_seed(3, 2, 7);
    embedding.weight = Tensor::<f32>::from_vec([3, 2], vec![1., 1., 2., 2., 3., 3.])
        .unwrap()
        .into_dense();
    let model = Embed { embedding };
    let output = predict_single(&model, vec![3], vec![2., 0., 1.]);
    assert_eq!(output.dims(), vec![3, 2]);
    assert_eq!(output.data(), &[3., 3., 1., 1., 2., 2.]);
}

#[test]
fn attention_and_mha_preserve_shape() {
    let model = Attn {
        attention: Attention::new(),
    };
    let data: Vec<f32> = (0..6).map(|value| value as f32 * 0.1).collect();
    let q = Tensor::<f32>::from_vec([1, 2, 3], data.clone()).unwrap();
    let k = Tensor::<f32>::from_vec([1, 2, 3], data.clone()).unwrap();
    let v = Tensor::<f32>::from_vec([1, 2, 3], data).unwrap();
    let output = model.predict(q, k, v).unwrap();
    assert_eq!(output.dims(), vec![1, 2, 3]);

    let model = Mha {
        mha: MultiHeadAttention::with_seed(4, 2, 5).unwrap(),
    };
    let output = predict_single(
        &model,
        vec![2, 3, 4],
        (0..24).map(|value| value as f32 * 0.01).collect(),
    );
    assert_eq!(output.dims(), vec![2, 3, 4]);
}

#[test]
fn recurrent_layers_return_last_hidden_state() {
    let model = Sequence {
        rnn: Rnn::with_seed(3, 5, 6).unwrap(),
    };
    let output = predict_single(
        &model,
        vec![2, 4, 3],
        (0..24).map(|value| value as f32 * 0.01).collect(),
    );
    assert_eq!(output.dims(), vec![2, 5]);

    let model = SequenceGru {
        gru: Gru::with_seed(3, 5, 7).unwrap(),
    };
    let output = predict_single(
        &model,
        vec![2, 4, 3],
        (0..24).map(|value| value as f32 * 0.01).collect(),
    );
    assert_eq!(output.dims(), vec![2, 5]);

    let model = SequenceLstm {
        lstm: Lstm::with_seed(3, 5, 8).unwrap(),
    };
    let output = predict_single(
        &model,
        vec![2, 4, 3],
        (0..24).map(|value| value as f32 * 0.01).collect(),
    );
    assert_eq!(output.dims(), vec![2, 5]);

    #[derive(Module)]
    struct SequenceOut {
        rnn: Rnn,
    }
    impl SequenceOut {
        #[model]
        fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
            self.rnn.forward(x)
        }
    }
    let model = SequenceOut {
        rnn: Rnn::with_seed(3, 5, 9).unwrap().with_sequences(),
    };
    let output = predict_single(
        &model,
        vec![2, 4, 3],
        (0..24).map(|value| value as f32 * 0.01).collect(),
    );
    assert_eq!(output.dims(), vec![2, 4, 5]);
}

#[test]
fn dropout_fails_explicitly_in_training_mode() {
    #[derive(Module)]
    struct WithDropout {
        dropout: Dropout,
    }
    impl WithDropout {
        #[model]
        fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
            self.dropout.forward(x)
        }
    }
    let model = WithDropout {
        dropout: Dropout::new(0.5).unwrap(),
    };
    // `predict` compiles in eval mode, so the mask is skipped.
    let input = Tensor::<f32>::from_vec([1, 2], vec![1., 2.]).unwrap();
    assert!(model.predict(input).is_ok());
    // A graph built in training mode must fail explicitly rather than silently
    // dropping the mask.
    let error = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [1, 2])],
        ModelCompileOptions::training(),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
}

#[test]
fn descriptor_hierarchy_matches_module_tree() {
    let model = Stack {
        first: Linear::with_seed(2, 3, true, 1),
        second: Linear::with_seed(3, 1, true, 2),
    };
    let descriptor = model.descriptor().unwrap();
    assert_eq!(descriptor.module_count(), 3);
    assert_eq!(descriptor.parameter_count(), 4);
    let rendered = descriptor.render();
    assert!(rendered.contains("first.weight"));
    assert!(rendered.contains("second.bias"));
}

#[test]
fn model_mode_propagates_to_children() {
    let mut sequential = Sequential::new()
        .add(Linear::with_seed(2, 2, true, 1))
        .add(ReLU::new());
    assert!(sequential.is_training());
    sequential.set_training(false);
    assert!(!sequential.is_training());
    let input = Tensor::<f32>::from_vec([1, 2], vec![1., 2.]).unwrap();
    assert!(sequential.predict(input).is_ok());
}
