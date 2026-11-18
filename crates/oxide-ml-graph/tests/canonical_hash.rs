use oxide_ml_core::{DType, Dim, Shape};
use oxide_ml_graph::{Graph, GraphBuilder, graph_structure_hash};

fn h(graph: &Graph) -> [u8; 32] {
    graph_structure_hash(graph).expect("graph hash").0
}

fn independent(order: u8) -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = builder.input("y", Shape::from_static([2, 3]), DType::F32);
    let (a, b) = if order == 0 {
        let a = builder.relu(x).unwrap();
        let b = builder.tanh(y).unwrap();
        (a, b)
    } else {
        let b = builder.tanh(y).unwrap();
        let a = builder.relu(x).unwrap();
        (a, b)
    };
    let out = builder.add(a, b).unwrap();
    builder.finish(&[out]).unwrap()
}

fn parameters(reverse: bool) -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let (w1, w2) = if reverse {
        let w2 = builder
            .try_parameter("w2", Shape::from_static([3, 3]), DType::F32, true)
            .unwrap();
        let w1 = builder
            .try_parameter("w1", Shape::from_static([3, 3]), DType::F32, true)
            .unwrap();
        (w1, w2)
    } else {
        let w1 = builder
            .try_parameter("w1", Shape::from_static([3, 3]), DType::F32, true)
            .unwrap();
        let w2 = builder
            .try_parameter("w2", Shape::from_static([3, 3]), DType::F32, true)
            .unwrap();
        (w1, w2)
    };
    let a = builder.matmul(x, w1).unwrap();
    let b = builder.matmul(x, w2).unwrap();
    let out = builder.add(a, b).unwrap();
    builder.finish(&[out]).unwrap()
}

fn symbols(reverse: bool) -> Graph {
    let mut builder = GraphBuilder::new();
    let (batch, hidden) = if reverse {
        let hidden = builder.symbol("hidden");
        let batch = builder.symbol("batch");
        (batch, hidden)
    } else {
        let batch = builder.symbol("batch");
        let hidden = builder.symbol("hidden");
        (batch, hidden)
    };
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Symbol(batch), Dim::Symbol(hidden)]),
        DType::F32,
    );
    let out = builder.relu(x).unwrap();
    builder.finish(&[out]).unwrap()
}

fn scalar(value: f32) -> Graph {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::scalar(), DType::F32);
    let c = builder.constant_scalar(value);
    let out = builder.add(x, c).unwrap();
    builder.finish(&[out]).unwrap()
}

#[test]
fn same_graph_same_hash() {
    let graph = independent(0);
    assert_eq!(h(&graph), h(&graph));
}

#[test]
fn construction_order_does_not_change_hash() {
    assert_eq!(h(&independent(0)), h(&independent(1)));
}

#[test]
fn parameter_insertion_order_does_not_change_hash() {
    assert_eq!(h(&parameters(false)), h(&parameters(true)));
}

#[test]
fn symbol_creation_order_does_not_change_hash() {
    assert_eq!(h(&symbols(false)), h(&symbols(true)));
}

#[test]
fn json_roundtrip_preserves_hash() {
    let graph = parameters(false);
    let restored = Graph::from_json(&graph.to_json().unwrap()).unwrap();
    assert_eq!(h(&graph), h(&restored));
}

#[test]
fn different_attribute_is_different_hash() {
    let mut a = GraphBuilder::new();
    let x = a.input("x", Shape::from_static([2, 3]), DType::F32);
    let out = a.softmax(x, -1).unwrap();
    let a = a.finish(&[out]).unwrap();

    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let out = b.softmax(x, 0).unwrap();
    let b = b.finish(&[out]).unwrap();

    assert_ne!(h(&a), h(&b));
}

#[test]
fn different_edge_is_different_hash() {
    let mut a = GraphBuilder::new();
    let x = a.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = a.input("y", Shape::from_static([2, 3]), DType::F32);
    let out = a.relu(x).unwrap();
    let a = a.finish(&[out]).unwrap();
    let _ = y;

    let mut b = GraphBuilder::new();
    let _x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = b.input("y", Shape::from_static([2, 3]), DType::F32);
    let out = b.relu(y).unwrap();
    let b = b.finish(&[out]).unwrap();

    assert_ne!(h(&a), h(&b));
}

#[test]
fn different_operator_is_different_hash() {
    let mut a = GraphBuilder::new();
    let x = a.input("x", Shape::from_static([2, 3]), DType::F32);
    let out = a.relu(x).unwrap();
    let a = a.finish(&[out]).unwrap();

    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let out = b.sigmoid(x).unwrap();
    let b = b.finish(&[out]).unwrap();

    assert_ne!(h(&a), h(&b));
}

#[test]
fn different_shape_is_different_hash() {
    let mut a = GraphBuilder::new();
    let x = a.input("x", Shape::from_static([2, 3]), DType::F32);
    let out = a.relu(x).unwrap();
    let a = a.finish(&[out]).unwrap();

    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([3, 3]), DType::F32);
    let out = b.relu(x).unwrap();
    let b = b.finish(&[out]).unwrap();

    assert_ne!(h(&a), h(&b));
}

#[test]
fn different_dtype_is_different_hash() {
    let mut a = GraphBuilder::new();
    let x = a.input("x", Shape::from_static([2, 3]), DType::F32);
    let out = a.relu(x).unwrap();
    let a = a.finish(&[out]).unwrap();

    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F64);
    let out = b.relu(x).unwrap();
    let b = b.finish(&[out]).unwrap();

    assert_ne!(h(&a), h(&b));
}

#[test]
fn different_constant_value_is_different_hash() {
    assert_ne!(h(&scalar(1.0)), h(&scalar(2.0)));
}

#[test]
fn negative_zero_and_positive_zero_are_different() {
    assert_ne!(h(&scalar(0.0)), h(&scalar(-0.0)));
}

#[test]
fn nan_payloads_are_bitwise_identity() {
    let one = f32::from_bits(0x7fc0_0001);
    let two = f32::from_bits(0x7fc0_0002);
    assert_ne!(h(&scalar(one)), h(&scalar(two)));
    assert_eq!(h(&scalar(one)), h(&scalar(one)));
}

#[test]
fn different_graph_output_is_different_hash() {
    let mut a = GraphBuilder::new();
    let x = a.input("x", Shape::from_static([2, 3]), DType::F32);
    let p = a.relu(x).unwrap();
    let _q = a.tanh(x).unwrap();
    let a = a.finish(&[p]).unwrap();

    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let _p = b.relu(x).unwrap();
    let q = b.tanh(x).unwrap();
    let b = b.finish(&[q]).unwrap();

    assert_ne!(h(&a), h(&b));
}

#[test]
fn canonical_version_participates_in_hash() {
    let graph = independent(0);
    let mut canonical = graph.canonicalize().unwrap();
    let first = canonical.hash();
    canonical.version += 1;
    assert_ne!(first, canonical.hash());
}

#[test]
fn canonical_value_and_node_maps_are_available() {
    let graph = independent(0);
    let canonical = graph.canonicalize().unwrap();
    for value in graph
        .values()
        .iter()
        .map(|value| value.id)
        .collect::<Vec<_>>()
    {
        assert!(canonical.value_ref(value).is_some());
    }
    for node in graph.nodes() {
        assert!(canonical.node_ref(node.id).is_some());
    }
}

#[test]
fn raw_identity_tracks_numbering_and_roundtrips() {
    let first = independent(0);
    let second = independent(1);
    assert_eq!(
        first.raw_identity_hash().unwrap(),
        first.raw_identity_hash().unwrap()
    );
    assert_ne!(
        first.raw_identity_hash().unwrap(),
        second.raw_identity_hash().unwrap()
    );
    let restored = Graph::from_json(&first.to_json().unwrap()).unwrap();
    assert_eq!(
        first.raw_identity_hash().unwrap(),
        restored.raw_identity_hash().unwrap()
    );
}

#[test]
fn canonical_ignores_node_ids_but_raw_identity_tracks_them() {
    let first = independent(0);
    let second = independent(1);
    assert_eq!(
        first.canonicalize().unwrap(),
        second.canonicalize().unwrap()
    );
    assert_ne!(
        first.raw_identity_hash().unwrap(),
        second.raw_identity_hash().unwrap()
    );
}

#[test]
fn output_order_is_part_of_graph_identity() {
    let mut a = GraphBuilder::new();
    let x = a.input("x", Shape::from_static([2, 3]), DType::F32);
    let p = a.relu(x).unwrap();
    let q = a.tanh(x).unwrap();
    let a = a.finish(&[p, q]).unwrap();

    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 3]), DType::F32);
    let p = b.relu(x).unwrap();
    let q = b.tanh(x).unwrap();
    let b = b.finish(&[q, p]).unwrap();

    assert_ne!(h(&a), h(&b));
}
