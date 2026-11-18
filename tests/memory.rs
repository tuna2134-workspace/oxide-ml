use oxide_ml::prelude::*;

fn inputs(graph: &Graph, x: DenseTensor, y: DenseTensor) -> ExecutionInputs<f32> {
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(graph.inputs()[0], x);
    inputs.insert(graph.inputs()[1], y);
    inputs
}

#[test]
fn planned_and_unplanned_outputs_match() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 8]), DType::F32);
    let y = builder.input("y", Shape::from_static([4, 8]), DType::F32);
    let a = builder.relu(x).unwrap();
    let b = builder.sigmoid(a).unwrap();
    let c = builder.tanh(b).unwrap();
    let out = builder.add(c, y).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let plan = MemoryPlanner::default().plan(&graph).unwrap();
    let executor = Executor::<f32>::new(graph.clone()).unwrap();
    let params = ParameterStore::<f32>::new();
    let inputs = inputs(
        &graph,
        DenseTensor::from_vec([4, 8], (0..32).map(|i| (i as f32 - 16.0) * 0.2).collect()).unwrap(),
        DenseTensor::from_vec([4, 8], vec![0.5; 32]).unwrap(),
    );
    let normal = executor.run(&params, &inputs).unwrap();
    let (planned, stats) = executor.run_planned(&params, &inputs, &plan).unwrap();
    assert_eq!(
        normal.output(0).unwrap().data(),
        planned.output(0).unwrap().data()
    );
    assert_eq!(stats.buffer_allocations, plan.allocation_count());
    assert_eq!(stats.planned_peak_bytes, plan.peak_bytes());
    assert!(plan.allocation_count() < plan.assignments.len());
    assert!(plan.peak_bytes() < plan.total_value_bytes);
}

#[test]
fn memory_plan_liveness_marks_outputs_live_to_end() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([8]), DType::F32);
    let a = builder.relu(x).unwrap();
    let out = builder.sigmoid(a).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let plan = MemoryPlanner::default().plan(&graph).unwrap();
    let output_lifetime = plan
        .lifetimes
        .iter()
        .find(|lifetime| lifetime.value == out)
        .unwrap();
    assert_eq!(
        output_lifetime.last_used_at,
        oxide_ml_core::LifetimePoint::End
    );
    let input_lifetime = plan.lifetimes.iter().find(|lifetime| lifetime.value == x);
    assert!(input_lifetime.is_none(), "external inputs are not planned");
}

#[test]
fn planned_training_exposes_loss_and_parameter_gradients() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 2]), DType::F32);
    let target = builder.input("y", Shape::from_static([4, 1]), DType::F32);
    let w = builder
        .try_parameter("w", Shape::from_static([2, 1]), DType::F32, true)
        .unwrap();
    let b = builder
        .try_parameter("b", Shape::from_static([1]), DType::F32, true)
        .unwrap();
    let product = builder.matmul(x, w).unwrap();
    let prediction = builder.add(product, b).unwrap();
    let loss = builder
        .mse_loss(prediction, target, Reduction::Mean)
        .unwrap();
    let forward = builder.finish(&[prediction, loss]).unwrap();
    let training = build_training_graph(&forward, loss).unwrap();
    assert!(training.graph.outputs().contains(&loss));
    let plan = MemoryPlanner::default().plan(&training.graph).unwrap();
    let mut params = init_parameters::<f32>(&forward, 4).unwrap();
    params.get_mut(ParameterId(0)).unwrap().data =
        DenseTensor::<f32>::from_vec([2, 1], vec![0.5, -0.25]).unwrap();
    params.get_mut(ParameterId(1)).unwrap().data =
        DenseTensor::<f32>::from_vec([1], vec![0.1]).unwrap();
    let inputs = inputs(
        &training.graph,
        DenseTensor::from_vec([4, 2], vec![1., 2., -1., 0.5, 2., -2., 0.25, 1.5]).unwrap(),
        DenseTensor::from_vec([4, 1], vec![1., -0.5, 0.25, 2.]).unwrap(),
    );
    let (result, _) = Executor::<f32>::new(training.graph.clone())
        .unwrap()
        .run_planned(&params, &inputs, &plan)
        .unwrap();
    assert!(result.value(loss).is_ok());
    let gradients = collect_gradients(&training, &result).unwrap();
    assert_eq!(gradients.len(), 2);
}

#[test]
fn input_and_output_buffers_are_not_reused_incorrectly() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4]), DType::F32);
    let a = builder.relu(x).unwrap();
    let b = builder.sigmoid(a).unwrap();
    let graph = builder.finish(&[x, b]).unwrap();
    let plan = MemoryPlanner::default().plan(&graph).unwrap();
    let mut params = ParameterStore::<f32>::new();
    let mut inputs = ExecutionInputs::<f32>::new();
    let original = DenseTensor::from_vec([4], vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    inputs.insert(graph.inputs()[0], original.clone());
    let (result, _) = Executor::<f32>::new(graph)
        .unwrap()
        .run_planned(&params, &inputs, &plan)
        .unwrap();
    assert_eq!(result.output(0).unwrap().data(), original.data());
    let expected = [
        0.5,
        1.0 / (1.0 + (-2.0f32).exp()),
        0.5,
        1.0 / (1.0 + (-4.0f32).exp()),
    ];
    assert_eq!(result.output(1).unwrap().data(), &expected);
    let _ = &mut params;
}

#[test]
fn planned_slots_are_aligned() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([3, 5]), DType::F32);
    let a = builder.relu(x).unwrap();
    let b = builder.sigmoid(a).unwrap();
    let graph = builder.finish(&[b]).unwrap();
    let plan = MemoryPlanner::default().plan(&graph).unwrap();
    for slot in &plan.slots {
        assert_eq!(slot.size_bytes % slot.alignment, 0);
    }
    for assignment in plan.assignments.values() {
        assert_eq!(assignment.offset % 64, 0);
    }
}
