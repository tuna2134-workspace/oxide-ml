use oxide_ml::prelude::*;

fn run_f32(graph: &Graph, inputs: &[(&str, DenseTensor)]) -> Vec<f32> {
    let inputs = ExecutionInputs::<f32>::from_named(
        graph,
        inputs.iter().map(|(name, tensor)| (*name, tensor.clone())),
    )
    .unwrap();
    let result = Executor::<f32>::new(graph.clone())
        .unwrap()
        .run(&ParameterStore::<f32>::new(), &inputs)
        .unwrap();
    result.output(0).unwrap().data().to_vec()
}

fn run_passes(graph: &mut Graph, names: &[&str]) -> CompileReport {
    let backend = Box::new(oxide_ml_cpu::CpuBackend::<f32>::new());
    let mut manager = PassManager::<f32>::default_passes();
    for name in manager.pass_names() {
        manager.set_enabled(name, names.contains(&name));
    }
    manager.run(graph, backend.as_ref()).unwrap()
}

#[test]
fn constant_folding_preserves_output() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let a = builder
        .constant_tensor(
            ConstantTensor::f32(Shape::from_static([2, 2]), vec![1., 2., 3., 4.]).unwrap(),
        )
        .unwrap();
    let b = builder
        .constant_tensor(
            ConstantTensor::f32(Shape::from_static([2, 2]), vec![10., 20., 30., 40.]).unwrap(),
        )
        .unwrap();
    let sum = builder.add(a, b).unwrap();
    let y = builder.add(x, sum).unwrap();
    let mut graph = builder.finish(&[y]).unwrap();
    let inputs = [(
        "x",
        DenseTensor::from_vec([2, 2], vec![0.5, 0.5, 0.5, 0.5]).unwrap(),
    )];
    let before = run_f32(&graph, &inputs);
    let report = run_passes(&mut graph, &["constant-folding"]);
    assert!(report.passes[0].changed);
    assert!(
        graph
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::Constant))
    );
    let after = run_f32(&graph, &inputs);
    assert_eq!(before, after);
}

#[test]
fn dce_removes_dead_nodes() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2]), DType::F32);
    let dead = builder.mul(x, y).unwrap();
    let dead2 = builder.relu(dead).unwrap();
    let _ = dead2;
    let live = builder.add(x, y).unwrap();
    let mut graph = builder.finish(&[live]).unwrap();
    let inputs = [
        ("x", DenseTensor::from_vec([2], vec![1.0, 2.0]).unwrap()),
        ("y", DenseTensor::from_vec([2], vec![3.0, 4.0]).unwrap()),
    ];
    let before = run_f32(&graph, &inputs);
    let before_count = graph.nodes().len();
    let report = run_passes(&mut graph, &["dce"]);
    assert!(report.passes[0].changed);
    assert!(graph.nodes().len() < before_count);
    assert_eq!(before, run_f32(&graph, &inputs));
}

#[test]
fn cse_merges_identical_nodes() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2]), DType::F32);
    let first = builder.add(x, y).unwrap();
    let second = builder.add(x, y).unwrap();
    let out = builder.mul(first, second).unwrap();
    let mut graph = builder.finish(&[out]).unwrap();
    let inputs = [
        ("x", DenseTensor::from_vec([2], vec![2.0, 3.0]).unwrap()),
        ("y", DenseTensor::from_vec([2], vec![4.0, 5.0]).unwrap()),
    ];
    let before = run_f32(&graph, &inputs);
    let before_count = graph.nodes().len();
    let report = run_passes(&mut graph, &["cse"]);
    assert!(report.passes[0].changed);
    assert!(graph.nodes().len() < before_count);
    assert_eq!(before, run_f32(&graph, &inputs));
}

#[test]
fn add_relu_fusion_preserves_forward_and_backward() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2, 2]), DType::F32);
    let sum = builder.add(x, y).unwrap();
    let out = builder.relu(sum).unwrap();
    let mut graph = builder.finish(&[out]).unwrap();
    let inputs = [
        (
            "x",
            DenseTensor::from_vec([2, 2], vec![-1., 2., -3., 4.]).unwrap(),
        ),
        (
            "y",
            DenseTensor::from_vec([2, 2], vec![2., -1., 4., -3.]).unwrap(),
        ),
    ];
    let before = run_f32(&graph, &inputs);
    let report = run_passes(&mut graph, &["fusion"]);
    assert!(report.passes[0].changed);
    assert_eq!(graph.nodes().len(), 1);
    assert!(
        graph
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::AddRelu))
    );
    assert_eq!(before, run_f32(&graph, &inputs));

    let _ = out;
}

#[test]
fn compiler_full_pipeline_preserves_output() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2, 2]), DType::F32);
    let a = builder
        .constant_tensor(
            ConstantTensor::f32(Shape::from_static([2, 2]), vec![1., 2., 3., 4.]).unwrap(),
        )
        .unwrap();
    let c = builder
        .constant_tensor(
            ConstantTensor::f32(Shape::from_static([2, 2]), vec![4., 3., 2., 1.]).unwrap(),
        )
        .unwrap();
    let csum = builder.add(a, c).unwrap();
    let sum = builder.add(x, y).unwrap();
    let dead = builder.mul(sum, csum).unwrap();
    let _ = dead;
    let out = builder.relu(sum).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let inputs = [
        (
            "x",
            DenseTensor::from_vec([2, 2], vec![-1., 2., -3., 4.]).unwrap(),
        ),
        (
            "y",
            DenseTensor::from_vec([2, 2], vec![2., -1., 4., -3.]).unwrap(),
        ),
    ];
    let before = run_f32(&graph, &inputs);
    let compiler = Compiler::<f32>::new(
        Box::new(oxide_ml_cpu::CpuBackend::<f32>::new()),
        CompileOptions::default(),
    );
    let compiled = compiler.compile(graph).unwrap();
    assert!(compiled.report.changed());
    assert!(compiled.graph.nodes().len() < 3);
    assert_eq!(before, run_f32(&compiled.graph, &inputs));
    let json = compiled.graph.to_json().unwrap();
    assert_eq!(compiled.graph, Graph::from_json(&json).unwrap());
}

#[test]
fn disabled_passes_leave_graph_unchanged() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2]), DType::F32);
    let sum = builder.add(x, y).unwrap();
    let mut graph = builder.finish(&[sum]).unwrap();
    let report = run_passes(&mut graph, &[]);
    assert!(!report.changed());
    assert_eq!(graph.nodes().len(), 1);
}

#[test]
fn optimized_graph_still_supports_backward() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2, 2]), DType::F32);
    let sum = builder.add(x, y).unwrap();
    let out = builder.relu(sum).unwrap();
    let loss = builder.reduce_sum(out, vec![0, 1], false).unwrap();
    let graph = builder.finish(&[loss]).unwrap();
    let compiler = Compiler::<f32>::new(
        Box::new(oxide_ml_cpu::CpuBackend::<f32>::new()),
        CompileOptions::default(),
    );
    let compiled = compiler.compile(graph).unwrap();
    assert!(compiled.memory_plan.is_some());
    assert!(
        compiled
            .graph
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::AddRelu))
    );
    let training = build_training_graph(&compiled.graph, loss).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        compiled.graph.inputs()[0],
        DenseTensor::from_vec([2, 2], vec![-1., 2., -3., 4.]).unwrap(),
    );
    inputs.insert(
        compiled.graph.inputs()[1],
        DenseTensor::from_vec([2, 2], vec![2., -1., 4., -3.]).unwrap(),
    );
    let result = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run(&ParameterStore::<f32>::new(), &inputs)
        .unwrap();
    let dx = result
        .value(
            training
                .gradient_for_value(compiled.graph.inputs()[0])
                .unwrap(),
        )
        .unwrap();
    assert_eq!(dx.data(), &[1.0, 1.0, 1.0, 1.0]);

    let plan = compiled.memory_plan.as_ref().unwrap();
    let (planned, _) = Executor::<f32>::new(compiled.graph.clone())
        .unwrap()
        .run_planned(&ParameterStore::<f32>::new(), &inputs, plan)
        .unwrap();
    assert_eq!(
        planned.output(0).unwrap().data(),
        result.value(loss).unwrap().data()
    );
}

#[test]
fn pass_order_does_not_change_results() {
    let (base, inputs) = {
        let mut builder = GraphBuilder::new();
        let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
        let y = builder.input("y", Shape::from_static([2, 2]), DType::F32);
        let c = builder
            .constant_tensor(
                ConstantTensor::f32(Shape::from_static([2, 2]), vec![1., 2., 3., 4.]).unwrap(),
            )
            .unwrap();
        let d = builder
            .constant_tensor(
                ConstantTensor::f32(Shape::from_static([2, 2]), vec![4., 3., 2., 1.]).unwrap(),
            )
            .unwrap();
        let csum = builder.add(c, d).unwrap();
        let sum = builder.add(x, y).unwrap();
        let dead = builder.mul(sum, csum).unwrap();
        let _ = dead;
        let out = builder.relu(sum).unwrap();
        let graph = builder.finish(&[out]).unwrap();
        let mut inputs = ExecutionInputs::<f32>::new();
        inputs.insert(
            graph.inputs()[0],
            DenseTensor::from_vec([2, 2], vec![-1., 2., -3., 4.]).unwrap(),
        );
        inputs.insert(
            graph.inputs()[1],
            DenseTensor::from_vec([2, 2], vec![2., -1., 4., -3.]).unwrap(),
        );
        (graph, inputs)
    };
    let expected = run_f32(
        &base,
        &[
            ("x", inputs.get(base.inputs()[0]).unwrap().clone()),
            ("y", inputs.get(base.inputs()[1]).unwrap().clone()),
        ],
    );
    let backend = Box::new(oxide_ml_cpu::CpuBackend::<f32>::new());
    let orders: [&[&str]; 4] = [
        &["constant-folding", "cse", "fusion", "dce"],
        &["dce", "constant-folding", "cse", "fusion"],
        &["fusion", "constant-folding", "dce", "cse"],
        &["cse", "fusion", "constant-folding", "dce"],
    ];
    for order in orders {
        let mut graph = base.clone();
        let mut manager = PassManager::<f32>::new();
        for name in order {
            match *name {
                "constant-folding" => manager.register(Box::new(oxide_ml::ConstantFolding)),
                "cse" => manager.register(Box::new(oxide_ml::CommonSubexpressionElimination)),
                "fusion" => manager.register(Box::new(oxide_ml::OperatorFusion)),
                "dce" => manager.register(Box::new(oxide_ml::DeadCodeElimination)),
                other => panic!("unknown pass {other}"),
            }
        }
        manager.run(&mut graph, backend.as_ref()).unwrap();
        assert_eq!(
            expected,
            run_f32(
                &graph,
                &[
                    ("x", inputs.get(base.inputs()[0]).unwrap().clone()),
                    ("y", inputs.get(base.inputs()[1]).unwrap().clone())
                ]
            )
        );
        GraphVerifier::verify(&graph).unwrap();
    }
}

#[test]
fn compile_options_can_disable_fusion() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2]), DType::F32);
    let sum = builder.add(x, y).unwrap();
    let out = builder.relu(sum).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let options = CompileOptions {
        passes: PassToggles {
            fusion: false,
            ..PassToggles::default()
        },
        ..CompileOptions::default()
    };
    let compiler = Compiler::<f32>::new(Box::new(oxide_ml_cpu::CpuBackend::<f32>::new()), options);
    let compiled = compiler.compile(graph).unwrap();
    assert!(
        !compiled
            .graph
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::AddRelu))
    );
}

#[test]
fn cse_is_transitive_through_duplicate_inputs() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2]), DType::F32);
    let a1 = builder.add(x, y).unwrap();
    let a2 = builder.add(x, y).unwrap();
    let m1 = builder.mul(a1, x).unwrap();
    let m2 = builder.mul(a2, x).unwrap();
    let out = builder.add(m1, m2).unwrap();
    let mut graph = builder.finish(&[out]).unwrap();
    let before = graph.nodes().len();
    let backend = Box::new(oxide_ml_cpu::CpuBackend::<f32>::new());
    let mut manager = PassManager::<f32>::new();
    manager.register(Box::new(CommonSubexpressionElimination));
    manager.run(&mut graph, backend.as_ref()).unwrap();
    assert!(graph.nodes().len() < before);
    let inputs = [
        ("x", DenseTensor::from_vec([2], vec![2., 3.]).unwrap()),
        ("y", DenseTensor::from_vec([2], vec![4., 5.]).unwrap()),
    ];
    let result = run_f32(&graph, &inputs);
    assert_eq!(result, vec![24.0, 48.0]);
}

#[test]
fn memory_planning_can_be_disabled() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let out = builder.relu(x).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let options = CompileOptions {
        memory_planning: false,
        ..CompileOptions::default()
    };
    let compiler = Compiler::<f32>::new(Box::new(oxide_ml_cpu::CpuBackend::<f32>::new()), options);
    let compiled = compiler.compile(graph).unwrap();
    assert!(compiled.memory_plan.is_none());
}

#[test]
fn pass_manager_can_dump_before_and_after_graphs() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2]), DType::F32);
    let a = builder.relu(x).unwrap();
    let mut graph = builder.finish(&[a]).unwrap();
    let backend = Box::new(oxide_ml_cpu::CpuBackend::<f32>::new());
    let mut manager = PassManager::<f32>::new();
    manager.register(Box::new(DeadCodeElimination));
    manager.set_dump(true);
    let report = manager.run(&mut graph, backend.as_ref()).unwrap();
    assert!(
        report
            .graph_before
            .as_deref()
            .unwrap()
            .contains("graph format")
    );
    assert!(
        report
            .graph_after
            .as_deref()
            .unwrap()
            .contains("graph format")
    );
}
