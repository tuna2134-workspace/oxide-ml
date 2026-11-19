//! Shape/dtype inference and validation for the additive operation set
//! (`sqrt`, `exp`, `log`, `abs`, `log_softmax`, `slice`, `embedding`) and their
//! fused backward primitives.

use oxide_ml_core::{DType, Device, Dim, Error, Shape, ValueId};
use oxide_ml_graph::{
    GraphBuilder, Op, OpAttrs, ValueInfo, ValueKind, infer_outputs, normalize_axis,
};

fn value(shape: Shape, dtype: DType) -> ValueInfo {
    ValueInfo {
        id: ValueId(0),
        name: None,
        shape,
        dtype,
        device: Device::Cpu,
        kind: ValueKind::Intermediate,
    }
}

fn f32_value(dims: impl AsRef<[usize]>) -> ValueInfo {
    value(Shape::from_static(dims), DType::F32)
}

fn infer(op: Op, attrs: OpAttrs, inputs: &[ValueInfo]) -> Result<Vec<(Vec<usize>, DType)>, Error> {
    let inferred = infer_outputs(&op, &attrs, inputs)?;
    Ok(inferred
        .into_iter()
        .map(|out| (out.shape.static_dims().unwrap_or_default(), out.dtype))
        .collect())
}

#[test]
fn unary_math_ops_pass_shape_and_dtype_through() {
    for op in [Op::Sqrt, Op::Exp, Op::Log, Op::Abs] {
        let out = infer(op, OpAttrs::None, &[f32_value([2, 3])]).unwrap();
        assert_eq!(out, vec![(vec![2, 3], DType::F32)]);
    }
    // F64 inputs keep F64 outputs.
    let out = infer(
        Op::Sqrt,
        OpAttrs::None,
        &[value(Shape::from_static([4]), DType::F64)],
    )
    .unwrap();
    assert_eq!(out, vec![(vec![4], DType::F64)]);
}

#[test]
fn log_softmax_preserves_shape_and_normalizes_axis() {
    let out = infer(
        Op::LogSoftmax,
        OpAttrs::LogSoftmax { axis: -1 },
        &[f32_value([2, 3])],
    )
    .unwrap();
    assert_eq!(out, vec![(vec![2, 3], DType::F32)]);

    let out = infer(
        Op::LogSoftmax,
        OpAttrs::LogSoftmax { axis: 0 },
        &[f32_value([2, 3])],
    )
    .unwrap();
    assert_eq!(out, vec![(vec![2, 3], DType::F32)]);

    let err = infer(
        Op::LogSoftmax,
        OpAttrs::LogSoftmax { axis: 3 },
        &[f32_value([2, 3])],
    )
    .unwrap_err();
    assert!(matches!(err, Error::InvalidAxis { axis: 3, rank: 2 }));
    assert_eq!(normalize_axis(-1, 2).unwrap(), 1);
}

#[test]
fn slice_replaces_the_axis_dimension() {
    let out = infer(
        Op::Slice,
        OpAttrs::Slice {
            axis: 1,
            start: 1,
            len: 2,
        },
        &[f32_value([2, 4])],
    )
    .unwrap();
    assert_eq!(out, vec![(vec![2, 2], DType::F32)]);

    let out = infer(
        Op::Slice,
        OpAttrs::Slice {
            axis: 0,
            start: 1,
            len: 1,
        },
        &[f32_value([3, 4])],
    )
    .unwrap();
    assert_eq!(out, vec![(vec![1, 4], DType::F32)]);

    let empty = infer(
        Op::Slice,
        OpAttrs::Slice {
            axis: 1,
            start: 4,
            len: 0,
        },
        &[f32_value([2, 4])],
    )
    .unwrap();
    assert_eq!(empty, vec![(vec![2, 0], DType::F32)]);
}

#[test]
fn slice_rejects_out_of_range_and_dynamic_dimensions() {
    let out_of_range = infer(
        Op::Slice,
        OpAttrs::Slice {
            axis: 0,
            start: 5,
            len: 2,
        },
        &[f32_value([4])],
    )
    .unwrap_err();
    assert!(matches!(out_of_range, Error::InvalidShape { .. }));

    let bad_axis = infer(
        Op::Slice,
        OpAttrs::Slice {
            axis: 2,
            start: 0,
            len: 1,
        },
        &[f32_value([2, 4])],
    )
    .unwrap_err();
    assert!(matches!(bad_axis, Error::InvalidShape { .. }));

    let dynamic = value(Shape::new(vec![Dim::Dynamic, Dim::Static(4)]), DType::F32);
    let non_static = infer(
        Op::Slice,
        OpAttrs::Slice {
            axis: 0,
            start: 0,
            len: 1,
        },
        &[dynamic],
    )
    .unwrap_err();
    assert!(matches!(non_static, Error::InvalidShape { .. }));
}

#[test]
fn embedding_appends_the_embedding_dimension() {
    let weight = f32_value([5, 3]);
    let indices = f32_value([2, 4]);
    let out = infer(Op::Embedding, OpAttrs::None, &[weight, indices]).unwrap();
    assert_eq!(out, vec![(vec![2, 4, 3], DType::F32)]);

    let scalar_indices = value(Shape::scalar(), DType::F32);
    let out = infer(
        Op::Embedding,
        OpAttrs::None,
        &[f32_value([5, 3]), scalar_indices],
    )
    .unwrap();
    assert_eq!(out, vec![(vec![3], DType::F32)]);

    let rank = infer(
        Op::Embedding,
        OpAttrs::None,
        &[f32_value([5]), f32_value([2])],
    )
    .unwrap_err();
    assert!(matches!(
        rank,
        Error::RankMismatch {
            expected: 2,
            got: 1,
            ..
        }
    ));

    let dtype = infer(
        Op::Embedding,
        OpAttrs::None,
        &[
            f32_value([5, 3]),
            value(Shape::from_static([2]), DType::F64),
        ],
    )
    .unwrap_err();
    assert!(matches!(dtype, Error::DTypeMismatch { .. }));
}

#[test]
fn embedding_backward_produces_vocabulary_gradient() {
    let out = infer(
        Op::EmbeddingBackward,
        OpAttrs::EmbeddingBackward { vocab: 5 },
        &[f32_value([2, 4]), f32_value([2, 4, 3])],
    )
    .unwrap();
    assert_eq!(out, vec![(vec![5, 3], DType::F32)]);

    let bad_rank = infer(
        Op::EmbeddingBackward,
        OpAttrs::EmbeddingBackward { vocab: 5 },
        &[f32_value([2, 4]), f32_value([2, 3])],
    )
    .unwrap_err();
    assert!(matches!(bad_rank, Error::RankMismatch { .. }));
}

#[test]
fn abs_backward_requires_matching_operands() {
    let out = infer(
        Op::AbsBackward,
        OpAttrs::None,
        &[f32_value([2, 3]), f32_value([2, 3])],
    )
    .unwrap();
    assert_eq!(out, vec![(vec![2, 3], DType::F32)]);

    let err = infer(
        Op::AbsBackward,
        OpAttrs::None,
        &[f32_value([2, 3]), f32_value([2, 1])],
    )
    .unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }));
}

#[test]
fn slice_backward_restores_the_input_shape() {
    let attrs = OpAttrs::SliceBackward {
        axis: 1,
        start: 1,
        len: 2,
        input_shape: Shape::from_static([2, 4]),
    };
    let out = infer(Op::SliceBackward, attrs.clone(), &[f32_value([2, 2])]).unwrap();
    assert_eq!(out, vec![(vec![2, 4], DType::F32)]);

    let err = infer(Op::SliceBackward, attrs, &[f32_value([2, 3])]).unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }));
}

#[test]
fn new_attrs_are_serializable() {
    for attrs in [
        OpAttrs::LogSoftmax { axis: -1 },
        OpAttrs::Slice {
            axis: 1,
            start: 2,
            len: 3,
        },
        OpAttrs::SliceBackward {
            axis: 1,
            start: 2,
            len: 3,
            input_shape: Shape::from_static([4, 8]),
        },
        OpAttrs::EmbeddingBackward { vocab: 7 },
    ] {
        let json = serde_json::to_string(&attrs).unwrap();
        assert_eq!(serde_json::from_str::<OpAttrs>(&json).unwrap(), attrs);
    }
}

#[test]
fn builder_helpers_build_and_validate_a_graph() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 4]), DType::F32);
    let weight = builder
        .try_parameter("weight", Shape::from_static([6, 3]), DType::F32, true)
        .unwrap();
    let indices = builder.input("indices", Shape::from_static([2]), DType::F32);

    let positive = builder.abs(x).unwrap();
    let root = builder.sqrt(positive).unwrap();
    let exponent = builder.exp(root).unwrap();
    let logged = builder.log(exponent).unwrap();
    let log_probs = builder.log_softmax(logged, -1).unwrap();
    let sliced = builder.slice(log_probs, 1, 1, 2).unwrap();
    let embedded = builder.embedding(weight, indices).unwrap();
    let loss = builder.reduce_mean(embedded, vec![0, 1], false).unwrap();
    let graph = builder.finish(&[sliced, loss]).unwrap();

    assert_eq!(
        graph.value(sliced).unwrap().shape.static_dims().unwrap(),
        vec![2, 2]
    );
    assert_eq!(
        graph.value(embedded).unwrap().shape.static_dims().unwrap(),
        vec![2, 3]
    );

    // The graph must survive a serialization round trip through validation.
    let json = graph.to_json().unwrap();
    let restored = oxide_ml_graph::Graph::from_json(&json).unwrap();
    assert_eq!(restored, graph);
}
