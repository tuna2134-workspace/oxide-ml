//! Constant shape / dtype / logical byte length invariants.

use oxide_ml_core::{DType, Shape};
use oxide_ml_graph::{ConstantTensor, GraphBuilder};

#[test]
fn scalar_constant_is_one_f32_element() {
    let mut builder = GraphBuilder::new();
    let value = builder.constant_scalar(2.5);
    let graph = builder.finish(&[value]).unwrap();
    let info = graph.value(value).unwrap();
    assert_eq!(info.dtype, DType::F32);
    assert_eq!(info.shape.numel_static().unwrap(), 1);
    assert_eq!(info.shape.rank(), 0);
    assert_eq!(
        info.shape.numel_static().unwrap() * DType::F32.size_in_bytes(),
        4
    );
}

#[test]
fn tensor_constant_preserves_declared_shape() {
    let mut builder = GraphBuilder::new();
    let value = builder
        .constant_tensor(
            ConstantTensor::f32(Shape::from_static([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        )
        .unwrap();
    let graph = builder.finish(&[value]).unwrap();
    let info = graph.value(value).unwrap();
    assert_eq!(info.shape.static_dims().unwrap(), vec![2, 2]);
    assert_eq!(info.shape.numel_static().unwrap(), 4);
    assert_eq!(
        info.shape.numel_static().unwrap() * DType::F32.size_in_bytes(),
        16
    );
}

#[test]
fn broadcast_does_not_expand_constant_source() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 4]), DType::F32);
    let bias = builder
        .constant_tensor(
            ConstantTensor::f32(Shape::from_static([4]), vec![0.1, 0.2, 0.3, 0.4]).unwrap(),
        )
        .unwrap();
    let y = builder.add(x, bias).unwrap();
    let graph = builder.finish(&[y]).unwrap();
    assert_eq!(
        graph.value(y).unwrap().shape.static_dims().unwrap(),
        vec![2, 4]
    );
    // The constant keeps its 4-element logical shape; broadcast is an
    // operator responsibility, not a host-side expansion.
    assert_eq!(
        graph.value(bias).unwrap().shape.static_dims().unwrap(),
        vec![4]
    );
    assert_eq!(graph.value(bias).unwrap().shape.numel_static().unwrap(), 4);
}
