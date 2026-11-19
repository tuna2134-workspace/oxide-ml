use oxide_ml::Error;

use oxide_ml::legacy::Mlp;
use oxide_ml::prelude::*;

fn mlp() -> Graph {
    let mut builder = GraphBuilder::new();
    let batch = builder.symbol("batch");
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(4)]),
        DType::F32,
    );
    let mlp = Mlp::new(&mut builder, "mlp", &[4, 8, 3], ActivationKind::ReLU, true).unwrap();
    let logits = mlp.forward(&mut builder, x).unwrap();
    builder.finish(&[logits]).unwrap()
}

#[test]
fn graph_json_roundtrip_preserves_structure() {
    let graph = mlp();
    let json = graph.to_json().unwrap();
    assert!(json.contains("\"format_version\": 2"));
    let restored = Graph::from_json(&json).unwrap();
    assert_eq!(graph, restored);
    assert_eq!(
        graph.structural_hash().unwrap(),
        restored.structural_hash().unwrap()
    );
    assert_eq!(restored.parameters().len(), 4);
    assert_eq!(restored.symbol_by_name("batch").unwrap().name, "batch");
}

#[test]
fn graph_file_roundtrip() {
    let graph = mlp();
    let path = std::env::temp_dir().join("oxide-ml-test-graph-ir.json");
    graph.save(&path).unwrap();
    let restored = Graph::load(&path).unwrap();
    assert_eq!(graph, restored);
}

#[test]
fn unknown_format_version_is_rejected() {
    let graph = mlp();
    let json = graph
        .to_json()
        .unwrap()
        .replace("\"format_version\": 2", "\"format_version\": 999");
    let error = Graph::from_json(&json).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("unsupported graph format version")
    );
}

#[test]
fn matmul_shape_mismatch_is_rejected_at_build_time() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 3]), DType::F32);
    let w = builder
        .try_parameter("w", Shape::from_static([5, 3]), DType::F32, true)
        .unwrap();
    let error = builder.matmul(x, w).unwrap_err();
    assert!(matches!(error, Error::ShapeMismatch { .. }));
}

#[test]
fn invalid_axis_is_rejected() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let error = builder.softmax(x, 7).unwrap_err();
    assert!(matches!(error, Error::InvalidAxis { .. }));
}

#[test]
fn empty_output_list_is_rejected() {
    let builder = GraphBuilder::new();
    let error = builder.finish(&[]).unwrap_err();
    assert!(matches!(error, Error::InvalidGraph(_)));
}

#[test]
fn training_graph_serializes_and_validates() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let target = builder.input("y", Shape::from_static([2, 1]), DType::F32);
    let mlp = Mlp::new(&mut builder, "mlp", &[2, 3, 1], ActivationKind::ReLU, true).unwrap();
    let prediction = mlp.forward(&mut builder, x).unwrap();
    let loss = builder
        .mse_loss(prediction, target, Reduction::Mean)
        .unwrap();
    let forward = builder.finish(&[prediction, loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    let json = training.graph.to_json().unwrap();
    let restored = Graph::from_json(&json).unwrap();
    assert_eq!(training.graph, restored);
    assert!(restored.nodes().iter().any(|node| node.op.is_backward()));
}

#[test]
fn unbroadcast_op_accepts_scalar_gradient() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let scalar = builder.constant_scalar(3.0);
    let y = builder.add(x, scalar).unwrap();
    let graph = builder.finish(&[y]).unwrap();
    assert_eq!(graph.values().len(), 3);
}

#[test]
fn verifier_rejects_invalid_value_id() {
    let mut graph = mlp();
    graph.values[0].id = ValueId(999);
    let error = GraphVerifier::verify(&graph).unwrap_err();
    assert!(error.to_string().contains("value table is not dense"));
}

#[test]
fn verifier_rejects_invalid_node_id() {
    let mut graph = mlp();
    graph.nodes[0].id = NodeId(999);
    let error = GraphVerifier::verify(&graph).unwrap_err();
    assert!(error.to_string().contains("node table is not dense"));
    assert!(graph.node(NodeId(999)).is_err());
}

#[test]
fn verifier_rejects_use_before_definition() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let a = builder.relu(x).unwrap();
    let b = builder.sigmoid(a).unwrap();
    let mut graph = builder.finish(&[b]).unwrap();
    graph.nodes[0].inputs = vec![b];
    let error = GraphVerifier::verify(&graph).unwrap_err();
    assert!(error.to_string().contains("before") || error.to_string().contains("cycle"));
}

#[test]
fn verifier_rejects_shape_mismatch() {
    let mut graph = mlp();
    let value = graph.outputs()[0];
    graph.value_mut(value).unwrap().shape = Shape::from_static([1, 2, 3, 4]);
    let error = GraphVerifier::verify(&graph).unwrap_err();
    assert!(error.to_string().contains("shape"));
}

#[test]
fn legacy_v1_envelope_is_migrated() {
    let graph = mlp();
    let json = graph
        .to_json()
        .unwrap()
        .replace("\"format_version\": 2", "\"format_version\": 1");
    // Remove the new IR version field to emulate a v1 envelope.
    let json = json
        .lines()
        .filter(|line| !line.trim_start().starts_with("\"ir_version\""))
        .collect::<Vec<_>>()
        .join("\n");
    let restored = Graph::from_json(&json).unwrap();
    assert_eq!(restored.format_version, SERIALIZATION_FORMAT_VERSION);
    assert_eq!(restored.ir_version, IR_VERSION);
}

#[test]
fn unsupported_ir_version_is_rejected() {
    let graph = mlp();
    let json = graph
        .to_json()
        .unwrap()
        .replace("\"ir_version\": 1", "\"ir_version\": 999");
    let error = Graph::from_json(&json).unwrap_err();
    assert!(error.to_string().contains("unsupported graph IR version"));
}

#[test]
fn invalid_operation_arity_and_attrs_are_rejected() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = builder.input("y", Shape::from_static([3, 2]), DType::F32);
    let error = builder
        .single(Op::MatMul, OpAttrs::None, &[x, y])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("MatMul requires OpAttrs::MatMul")
    );
    let error = builder.add(x, y).unwrap_err();
    assert!(matches!(error, Error::InvalidBroadcast { .. }));
}
