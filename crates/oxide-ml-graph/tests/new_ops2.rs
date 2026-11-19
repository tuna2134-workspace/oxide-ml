use oxide_ml_core::{DType, Shape};
use oxide_ml_graph::{GraphBuilder, GraphVerifier};

#[test]
fn cast_inference_keeps_shape_and_changes_dtype() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = b.cast(x, DType::F64).unwrap();
    assert_eq!(b.value_dtype(y).unwrap(), DType::F64);
    assert_eq!(b.value_shape(y).unwrap(), Shape::from_static([2, 3]));
    let graph = b.finish(&[y]).unwrap();
    assert!(GraphVerifier::verify(&graph).is_ok());
}

#[test]
fn concat_and_split_round_trip_shapes() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = b.input("y", Shape::from_static([2, 5]), DType::F32);
    let joined = b.concat(&[x, y], 1).unwrap();
    assert_eq!(b.value_shape(joined).unwrap(), Shape::from_static([2, 8]));
    let parts = b.split(joined, 1, vec![3, 5]).unwrap();
    assert_eq!(b.value_shape(parts[0]).unwrap(), Shape::from_static([2, 3]));
    assert_eq!(b.value_shape(parts[1]).unwrap(), Shape::from_static([2, 5]));
    let graph = b.finish(&parts).unwrap();
    assert!(GraphVerifier::verify(&graph).is_ok());
}

#[test]
fn concat_rejects_mismatched_non_axis_dims() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = b.input("y", Shape::from_static([4, 5]), DType::F32);
    assert!(b.concat(&[x, y], 1).is_err());
}

#[test]
fn split_rejects_wrong_total() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 8]), DType::F32);
    assert!(b.split(x, 1, vec![3, 3]).is_err());
}

#[test]
fn mul_add_inference_broadcasts() {
    let mut b = GraphBuilder::new();
    let a = b.input("a", Shape::from_static([2, 3]), DType::F32);
    let b_value = b.input("b", Shape::from_static([3]), DType::F32);
    let c = b.input("c", Shape::from_static([1]), DType::F32);
    let y = b.mul_add(a, b_value, c).unwrap();
    assert_eq!(b.value_shape(y).unwrap(), Shape::from_static([2, 3]));
}

#[test]
fn matmul_bias_inference_and_bias_validation() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([4, 3]), DType::F32);
    let w = b.input("w", Shape::from_static([5, 3]), DType::F32);
    let bias = b.input("bias", Shape::from_static([5]), DType::F32);
    let y = b.matmul_bias(x, w, bias, false, true).unwrap();
    assert_eq!(b.value_shape(y).unwrap(), Shape::from_static([4, 5]));

    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([4, 3]), DType::F32);
    let w = b.input("w", Shape::from_static([5, 3]), DType::F32);
    let bias = b.input("bias", Shape::from_static([7]), DType::F32);
    assert!(b.matmul_bias(x, w, bias, false, true).is_err());
}

#[test]
fn new_ops_serialize_and_verify() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 4]), DType::F32);
    let y = b.input("y", Shape::from_static([2, 4]), DType::F32);
    let sum = b.add(x, y).unwrap();
    let scaled = b.mul_add(sum, sum, x).unwrap();
    let parts = b.split(scaled, 1, vec![2, 2]).unwrap();
    let joined = b.concat(&parts, 1).unwrap();
    let graph = b.finish(&[joined]).unwrap();
    let json = graph.to_json().unwrap();
    let restored = oxide_ml_graph::Graph::from_json(&json).unwrap();
    assert_eq!(restored, graph);
    assert!(GraphVerifier::verify(&restored).is_ok());
}
