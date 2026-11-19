//! Phase 7F-19 Recovery: constant H2D copy regression.
//!
//! Reproduces the planned-slot capacity case directly (device capacity 16,
//! logical scalar 1) and then exercises scalar / tensor / broadcast constants
//! through normal, planned, arena and async planned execution.

use std::collections::HashMap;

use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{DType, DenseTensor, ParameterId, ParameterStore, Shape, ValueId};
use oxide_ml_cuda::{
    CudaBufferPool, CudaDevice, CudaExecution, CudaExecutor, CudaMemoryPlan, CudaParameterStore,
    CudaPlannedArena, CudaTensor,
};
use oxide_ml_graph::{Graph, GraphBuilder};

fn fail(message: impl Into<String>) -> Result<(), Box<dyn std::error::Error>> {
    Err(message.into().into())
}

fn check_close(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    tol: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return fail(format!(
            "{label}: length {} vs {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || (a - e).abs() > tol {
            return fail(format!("{label}[{index}]: actual={a} expected={e}"));
        }
    }
    println!("constant_copy_check: {label} PASS");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let executor = CudaExecutor::new(device.clone())?;
    let pool = CudaBufferPool::new(device.context().clone(), device.default_stream());
    let stream = device.default_stream();

    // 1. Direct capacity > logical case. This is the exact 16 vs 1 shape of
    //    the reported panic: a planned slot has capacity 16 while the value
    //    is a scalar (1 element).
    let mut slot = CudaTensor::from_pool(&pool, vec![16])?;
    slot.set_shape(vec![1])?;
    assert_eq!(slot.capacity(), 16);
    assert_eq!(slot.numel(), 1);
    slot.copy_from_host(&stream, &[7.5])?;
    let scalar_host = slot.to_host(&stream)?;
    check_close("capacity16_scalar_copy", &scalar_host[..1], &[7.5], 0.0)?;
    // Only the logical prefix is written; the rest of the slot keeps its
    // previous contents (zeros here) and is not part of the tensor value.
    let raw = slot.to_host(&stream)?;
    check_close("capacity16_scalar_prefix", &raw[..1], &[7.5], 0.0)?;

    let mut slot4 = CudaTensor::from_pool(&pool, vec![16])?;
    slot4.set_shape(vec![4])?;
    slot4.copy_from_host(&stream, &[1.0, 2.0, 3.0, 4.0])?;
    check_close(
        "capacity16_tensor4_copy",
        &slot4.to_host(&stream)?[..4],
        &[1.0, 2.0, 3.0, 4.0],
        0.0,
    )?;

    // 2. Scalar constant graph: normal / planned / arena.
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 4]), DType::F32);
    let c = builder.constant_scalar(1.5);
    let y = builder.add(x, c)?;
    let graph = builder.finish(&[y])?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([2, 4], vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])?,
    );
    let params = CudaParameterStore::from_host(&graph, &ParameterStore::<f32>::new(), &pool)?;
    let plan = CudaMemoryPlan::from_memory_plan(&graph, &MemoryPlanner::default().plan(&graph)?)?;
    let arena = CudaPlannedArena::new(plan.clone(), pool.clone(), 2)?;
    let arena_stream = device.new_stream()?;
    let normal = executor.run(&graph, &params, &inputs)?;
    let expected = normal.download(&stream, graph.outputs()[0])?;
    let planned = executor.run_planned(&graph, &params, &inputs, &plan)?;
    check_close(
        "scalar_constant_planned",
        planned.download(&stream, graph.outputs()[0])?.data(),
        expected.data(),
        1e-6,
    )?;
    let arena_run = executor.run_arena(&graph, &params, &inputs, &arena, arena_stream.clone())?;
    check_close(
        "scalar_constant_arena",
        arena_run.download(&stream, graph.outputs()[0])?.data(),
        expected.data(),
        1e-6,
    )?;
    let mut async_exec =
        executor.execute_async_planned(&graph, &params, &inputs, &plan, stream.clone())?;
    let async_out = async_exec.wait()?;
    check_close(
        "scalar_constant_async_planned",
        async_out[&graph.outputs()[0]].data(),
        expected.data(),
        1e-6,
    )?;
    let _: &mut CudaExecution = &mut async_exec;

    // 3. Multi-element tensor constant through planned execution.
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 4]), DType::F32);
    let t = builder.constant_tensor(oxide_ml_graph::ConstantTensor::f32(
        Shape::from_static([4]),
        vec![0.5, -0.5, 1.5, -1.5],
    )?)?;
    let y = builder.add(x, t)?;
    let graph = builder.finish(&[y])?;
    let params = CudaParameterStore::from_host(&graph, &ParameterStore::<f32>::new(), &pool)?;
    let plan = CudaMemoryPlan::from_memory_plan(&graph, &MemoryPlanner::default().plan(&graph)?)?;
    let mut inputs = HashMap::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([2, 4], vec![0.0; 8])?,
    );
    let normal = executor.run(&graph, &params, &inputs)?;
    let expected = normal.download(&stream, graph.outputs()[0])?;
    let planned = executor.run_planned(&graph, &params, &inputs, &plan)?;
    check_close(
        "tensor_constant_planned",
        planned.download(&stream, graph.outputs()[0])?.data(),
        expected.data(),
        1e-6,
    )?;

    // 4. Linear + MSE training graph: the original blocked shape.
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 1]), DType::F32);
    let target = builder.input("y", Shape::from_static([4, 1]), DType::F32);
    let w = builder.try_parameter("w", Shape::from_static([1, 1]), DType::F32, true)?;
    let b = builder.try_parameter("b", Shape::from_static([1]), DType::F32, true)?;
    let product = builder.matmul(x, w)?;
    let pred = builder.add(product, b)?;
    let loss = builder.mse_loss(pred, target, oxide_ml_graph::Reduction::Mean)?;
    let forward = builder.finish(&[pred, loss])?;
    let training = oxide_ml_autograd::build_training_graph(&forward, loss)?;
    let mut host_params = ParameterStore::<f32>::new();
    host_params.add(
        ParameterId(0),
        "w",
        DenseTensor::from_vec([1, 1], vec![0.0])?,
        true,
    )?;
    host_params.add(
        ParameterId(1),
        "b",
        DenseTensor::from_vec([1], vec![0.0])?,
        true,
    )?;
    let params = CudaParameterStore::from_host(&training.graph, &host_params, &pool)?;
    let plan = CudaMemoryPlan::from_memory_plan(
        &training.graph,
        &MemoryPlanner::default().plan(&training.graph)?,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert(
        training.graph.inputs()[0],
        DenseTensor::from_vec([4, 1], vec![0.0, 1.0, 2.0, 3.0])?,
    );
    inputs.insert(
        training.graph.inputs()[1],
        DenseTensor::from_vec([4, 1], vec![1.0, 3.0, 5.0, 7.0])?,
    );
    let normal = executor.run(&training.graph, &params, &inputs)?;
    let planned = executor.run_planned(&training.graph, &params, &inputs, &plan)?;
    for &value in training.graph.outputs() {
        let expected = normal.download(&stream, value)?;
        let actual = planned.download(&stream, value)?;
        check_close(
            &format!("linear_training_planned_output_{}", value.0),
            actual.data(),
            expected.data(),
            1e-6,
        )?;
    }
    let arena = CudaPlannedArena::new(plan.clone(), pool.clone(), 2)?;
    let arena_run = executor.run_arena(
        &training.graph,
        &params,
        &inputs,
        &arena,
        arena_stream.clone(),
    )?;
    for &value in training.graph.outputs() {
        let expected = normal.download(&stream, value)?;
        let actual = arena_run.download(&stream, value)?;
        check_close(
            &format!("linear_training_arena_output_{}", value.0),
            actual.data(),
            expected.data(),
            1e-6,
        )?;
    }
    let mut async_exec =
        executor.execute_async_planned(&training.graph, &params, &inputs, &plan, stream.clone())?;
    let async_out = async_exec.wait()?;
    for &value in training.graph.outputs() {
        let expected = normal.download(&stream, value)?;
        check_close(
            &format!("linear_training_async_output_{}", value.0),
            async_out[&value].data(),
            expected.data(),
            1e-6,
        )?;
    }

    println!("constant_copy_check: PASS");
    println!("constant_copy_check: pool={:?}", executor.pool_stats());
    let _: Option<&Graph> = None;
    let _: Option<ValueId> = None;
    Ok(())
}
