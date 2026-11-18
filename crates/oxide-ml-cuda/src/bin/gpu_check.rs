use std::collections::HashMap;

use cuda_core::CudaStream;
use oxide_ml_compiler::MemoryPlanner;
use oxide_ml_core::{DType, DenseTensor, Dim, ParameterId, ParameterStore, Shape, ValueId};
use oxide_ml_cpu::ExecutionInputs;
use oxide_ml_cuda::{
    ArenaContextId, ArenaDescriptor, ArenaKey, ContextIdentity, CudaArenaRegistry, CudaBufferPool,
    CudaDevice, CudaEvent, CudaExecutor, CudaMemoryPlan, CudaParameterStore, CudaPlannedArena,
    CudaSgd, CudaTensor, CudaTrainer, DeviceIdentity, EventPool, EventState, ExecutionState,
};
use oxide_ml_graph::{Graph, GraphBuilder};
use oxide_ml_nn::{Mlp, init_parameters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    println!("device: {:?}", device.info());

    let stream = device.default_stream();
    let pool = CudaBufferPool::new(device.context().clone(), stream.clone());
    let host = vec![1.0f32, -2.0, 3.5, 0.0];
    let tensor = CudaTensor::from_host(&pool, vec![2, 2], &host)?;
    assert_eq!(host, tensor.to_host(&stream)?);
    println!("buffer roundtrip: PASS");

    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = builder.input("y", Shape::from_static([2, 1]), DType::F32);
    let w1 = builder.try_parameter("w1", Shape::from_static([3, 4]), DType::F32, true)?;
    let b1 = builder.try_parameter("b1", Shape::from_static([4]), DType::F32, true)?;
    let w2 = builder.try_parameter("w2", Shape::from_static([4, 1]), DType::F32, true)?;
    let b2 = builder.try_parameter("b2", Shape::from_static([1]), DType::F32, true)?;
    let h = builder.matmul(x, w1)?;
    let h = builder.add(h, b1)?;
    let h = builder.sigmoid(h)?;
    let y_hat = builder.matmul(h, w2)?;
    let y_hat = builder.add(y_hat, b2)?;
    let loss = builder.mse_loss(y_hat, y, oxide_ml_graph::Reduction::Mean)?;
    let graph = builder.finish(&[y_hat, loss])?;

    let mut params = ParameterStore::<f32>::new();
    params.add(
        ParameterId(0),
        "w1",
        DenseTensor::from_vec(
            [3, 4],
            vec![
                0.1, 0.2, -0.3, 0.4, 0.2, -0.5, 0.3, 0.1, -0.2, 0.4, 0.1, 0.2,
            ],
        )?,
        true,
    )?;
    params.add(
        ParameterId(1),
        "b1",
        DenseTensor::from_vec([4], vec![0.1, -0.2, 0.0, 0.3])?,
        true,
    )?;
    params.add(
        ParameterId(2),
        "w2",
        DenseTensor::from_vec([4, 1], vec![0.5, -0.4, 0.2, 0.1])?,
        true,
    )?;
    params.add(
        ParameterId(3),
        "b2",
        DenseTensor::from_vec([1], vec![0.05])?,
        true,
    )?;
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([2, 3], vec![1.0, 2.0, -1.0, 0.5, -1.0, 2.0])?,
    );
    inputs.insert(
        graph.inputs()[1],
        DenseTensor::from_vec([2, 1], vec![1.0, -1.0])?,
    );
    let cpu = oxide_ml_cpu::Executor::<f32>::new(graph.clone())?;
    let cpu_outputs = cpu.run(&params, &inputs)?;

    let cuda_params = CudaParameterStore::from_host(&graph, &params, &pool)?;
    let mut cuda_inputs = HashMap::new();
    cuda_inputs.insert(
        graph.inputs()[0],
        inputs.get(graph.inputs()[0]).unwrap().clone(),
    );
    cuda_inputs.insert(
        graph.inputs()[1],
        inputs.get(graph.inputs()[1]).unwrap().clone(),
    );
    let executor = CudaExecutor::new(device)?;
    let run = executor.run(&graph, &cuda_params, &cuda_inputs)?;
    for (index, &value) in graph.outputs().iter().enumerate() {
        let expected = cpu_outputs.output(index)?.data();
        let actual = run.download(&stream, value)?;
        for (&e, &a) in expected.iter().zip(actual.data()) {
            assert!(
                (e - a).abs() <= 3e-3 * e.abs().max(1.0),
                "index {index}: cpu={e} cuda={a}"
            );
        }
    }
    println!("mlp forward: PASS");

    // Phase 7: compiler MemoryPlan -> CUDA adapter -> planned slot execution.
    let memory_plan = oxide_ml_compiler::MemoryPlanner::default().plan(&graph)?;
    let cuda_plan = CudaMemoryPlan::from_memory_plan(&graph, &memory_plan)?;
    assert!(!cuda_plan.assignments.is_empty());
    let planned = executor.run_planned(&graph, &cuda_params, &cuda_inputs, &cuda_plan)?;
    for &value in graph.outputs() {
        let normal = run.download(&stream, value)?;
        let planned_value = planned.download(&stream, value)?;
        for (&a, &b) in normal.data().iter().zip(planned_value.data()) {
            assert!((a - b).abs() <= 1e-6, "planned mismatch: {a} vs {b}");
        }
    }
    println!(
        "memory plan: slots={} assignments={} logical_bytes={}",
        cuda_plan.slot_count(),
        cuda_plan.assignments.len(),
        cuda_plan.logical_bytes
    );
    println!("planned execution: PASS");

    let arena = CudaPlannedArena::new(cuda_plan.clone(), executor.pool().clone(), 2)?;
    let arena_stream = executor.device().new_stream()?;
    let arena_stream_b = executor.device().new_stream()?;
    let arena_run1 =
        executor.run_arena(&graph, &cuda_params, &cuda_inputs, &arena, arena_stream)?;
    let arena_run2 =
        executor.run_arena(&graph, &cuda_params, &cuda_inputs, &arena, arena_stream_b)?;
    for &value in graph.outputs() {
        let a = arena_run1.download(&stream, value)?;
        let b = arena_run2.download(&stream, value)?;
        for (&x, &y) in a.data().iter().zip(b.data()) {
            assert!((x - y).abs() <= 1e-6, "arena mismatch: {x} vs {y}");
        }
    }
    let arena_stats = arena.stats();
    println!("arena stats: {arena_stats:?}");
    assert_eq!(arena_stats.slot_set_count, 1);
    assert_eq!(arena_stats.allocation_count, 2);
    assert!(arena_stats.slot_set_reuse_count >= 1);
    for _ in 0..100 {
        let run = executor.run_arena(
            &graph,
            &cuda_params,
            &cuda_inputs,
            &arena,
            executor.device().new_stream()?,
        )?;
        let value = run.download(&stream, graph.outputs()[0])?;
        assert_eq!(value.data().len(), 2);
    }
    let arena_stats = arena.stats();
    assert_eq!(
        arena_stats.allocation_count, 2,
        "arena slots were reallocated"
    );
    assert!(arena_stats.slot_set_reuse_count >= 99);
    println!("persistent arena execution: PASS (100 runs, 2 slot allocations)");
    let registry = CudaArenaRegistry::default();
    let device_identity = executor.device().identity()?;
    let context_identity = executor.device().context_identity();
    let device_identity_again = executor.device().identity()?;
    let context_identity_again = executor.device().context_identity();
    assert_eq!(device_identity, device_identity_again);
    assert_eq!(context_identity.id(), context_identity_again.id());
    let keyed_a = registry.get_or_create(
        &graph,
        &cuda_plan,
        executor.pool().clone(),
        &device_identity,
        &context_identity,
        2,
    )?;
    let keyed_b = registry.get_or_create(
        &graph,
        &cuda_plan,
        executor.pool().clone(),
        &device_identity,
        &context_identity,
        2,
    )?;
    assert!(std::sync::Arc::ptr_eq(&keyed_a, &keyed_b));

    // A different graph must not share the arena.
    let mut other_builder = GraphBuilder::new();
    let other_x = other_builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let other_c = other_builder.constant_scalar(1.0);
    let other_out = other_builder.add(other_x, other_c)?;
    let other_graph = other_builder.finish(&[other_out])?;
    let other_plan = oxide_ml_compiler::MemoryPlanner::default().plan(&other_graph)?;
    let other_cuda_plan = CudaMemoryPlan::from_memory_plan(&other_graph, &other_plan)?;
    let keyed_other_graph = registry.get_or_create(
        &other_graph,
        &other_cuda_plan,
        executor.pool().clone(),
        &device_identity,
        &context_identity,
        2,
    )?;
    assert!(!std::sync::Arc::ptr_eq(&keyed_a, &keyed_other_graph));

    // A plan is bound to the raw graph numbering it was built from. Passing
    // it with a different graph must be rejected before any arena reuse.
    let mut mismatch_builder = GraphBuilder::new();
    let mismatch_x = mismatch_builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let mismatch_c = mismatch_builder.constant_scalar(2.0);
    let mismatch_out = mismatch_builder.add(mismatch_x, mismatch_c)?;
    let mismatch_graph = mismatch_builder.finish(&[mismatch_out])?;
    assert!(
        registry
            .get_or_create(
                &mismatch_graph,
                &other_cuda_plan,
                executor.pool().clone(),
                &device_identity,
                &context_identity,
                2,
            )
            .is_err(),
        "plan from a different graph numbering was accepted"
    );

    // A different plan for the same graph must not share the arena.
    let alt_plan = oxide_ml_compiler::MemoryPlanner::new(32)?.plan(&graph)?;
    let alt_cuda_plan = CudaMemoryPlan::from_memory_plan(&graph, &alt_plan)?;
    let keyed_other_plan = registry.get_or_create(
        &graph,
        &alt_cuda_plan,
        executor.pool().clone(),
        &device_identity,
        &context_identity,
        2,
    )?;
    assert!(!std::sync::Arc::ptr_eq(&keyed_a, &keyed_other_plan));

    assert_eq!(registry.len(), 3);
    let registry_stats = registry.stats();
    assert!(registry_stats.hit_count >= 1);
    assert!(registry_stats.miss_count >= 3);
    assert_eq!(registry_stats.collision_count, 0);

    // Hash-collision simulation: same key, different exact descriptor. The
    // second lookup must create a separate arena rather than sharing.
    let same_key = ArenaKey::new(
        graph.canonicalize()?.hash(),
        cuda_plan.plan_hash().unwrap(),
        device_identity.id().clone(),
        context_identity.id(),
        DType::F32,
        oxide_ml_compiler::MEMORY_PLAN_VERSION,
    );
    let mut colliding_descriptor = ArenaDescriptor::new(
        graph.canonicalize()?,
        cuda_plan.canonical_plan().unwrap().as_ref().clone(),
        device_identity.id().clone(),
        ArenaContextId(999),
        DType::F32,
    )
    .with_backend_layout(cuda_plan.raw_layout_bytes());
    colliding_descriptor.context_id = ArenaContextId(999);
    let collision_arena = registry.get_or_create_explicit(
        same_key,
        colliding_descriptor,
        2,
        &cuda_plan,
        executor.pool().clone(),
    )?;
    assert!(!std::sync::Arc::ptr_eq(&keyed_a, &collision_arena));
    assert_eq!(registry.len(), 4);
    assert!(registry.stats().collision_count >= 1);
    println!(
        "arena collision handling: PASS (stats={:?})",
        registry.stats()
    );
    println!(
        "arena key registry: PASS (device={:?} context={:?} stats={:?})",
        device_identity.id(),
        context_identity.id(),
        registry_stats
    );

    let cap_arena = CudaPlannedArena::new(cuda_plan.clone(), executor.pool().clone(), 1)?;
    let lease = cap_arena.lease(&stream)?;
    let busy = cap_arena.lease(&executor.device().new_stream()?);
    assert!(
        busy.is_err(),
        "capacity-1 arena accepted a concurrent lease"
    );
    drop(lease);
    cap_arena.poll()?;
    let lease2 = cap_arena.lease(&stream)?;
    drop(lease2);
    println!("arena capacity/busy: PASS");

    let concurrent = std::sync::Arc::new(CudaPlannedArena::new(
        cuda_plan.clone(),
        executor.pool().clone(),
        2,
    )?);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..2 {
            let concurrent = concurrent.clone();
            let executor_ref = &executor;
            let graph_ref = &graph;
            let params_ref = &cuda_params;
            let inputs_ref = &cuda_inputs;
            handles.push(scope.spawn(move || {
                let stream = executor_ref.device().new_stream().unwrap();
                executor_ref
                    .run_arena(graph_ref, params_ref, inputs_ref, &concurrent, stream)
                    .unwrap()
            }));
        }
        for handle in handles {
            let run = handle.join().unwrap();
            let value = run.download(&stream, graph.outputs()[0]).unwrap();
            assert_eq!(value.data().len(), 2);
        }
    });
    assert!(concurrent.stats().slot_set_count >= 1);
    println!("concurrent arena execution: PASS");

    let state_event = CudaEvent::new(executor.device().context())?;
    assert_eq!(state_event.state(), EventState::Unrecorded);
    assert!(!state_event.is_complete()?);
    state_event.record(&stream)?;
    assert_eq!(state_event.state(), EventState::Recorded);
    state_event.synchronize()?;
    assert_eq!(state_event.state(), EventState::Completed);
    println!("event state machine: PASS");

    phase7f13_event_safety_checks(&executor, &stream)?;

    // Read/Read must not wait; the following Write waits for both reads.
    let access_stream_a = executor.device().new_stream()?;
    let access_stream_b = executor.device().new_stream()?;
    let access_stream_c = executor.device().new_stream()?;
    let mut shared = CudaTensor::from_host(executor.pool(), vec![4], &[1.0, 2.0, 3.0, 4.0])?;
    let mut read_a = CudaTensor::zeros(executor.pool(), vec![4])?;
    let mut read_b = CudaTensor::zeros(executor.pool(), vec![4])?;
    let _delays = [
        enqueue_delay(&executor, &access_stream_a)?,
        enqueue_delay(&executor, &access_stream_b)?,
    ];
    executor
        .kernels()
        .broadcast(&access_stream_a, &shared, &mut read_a)?;
    executor
        .kernels()
        .broadcast(&access_stream_b, &shared, &mut read_b)?;
    executor
        .kernels()
        .scale(&access_stream_c, &mut shared, 2.0)?;
    let access_stats = shared.access_stats();
    assert_eq!(access_stats.event_waits, 2, "{access_stats:?}");
    assert_eq!(access_stats.cross_stream_waits, 2, "{access_stats:?}");
    assert_eq!(access_stats.same_stream_elided_waits, 0, "{access_stats:?}");
    assert_eq!(access_stats.read_read_concurrent, 1, "{access_stats:?}");
    access_stream_a.synchronize()?;
    access_stream_b.synchronize()?;
    access_stream_c.synchronize()?;
    assert_eq!(read_a.to_host(&access_stream_a)?, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_b.to_host(&access_stream_b)?, vec![1.0, 2.0, 3.0, 4.0]);
    println!("buffer access read/read/write: PASS ({access_stats:?})");

    phase7f16_access_checks(&executor, &arena)?;
    println!("event pool stats: {:?}", executor.event_stats());

    let async_stream = executor.device().new_stream()?;
    let mut async_exec = executor.execute_async_planned(
        &graph,
        &cuda_params,
        &cuda_inputs,
        &cuda_plan,
        async_stream,
    )?;
    let _ = async_exec.is_complete()?;
    let async_outputs = async_exec.wait()?;
    for &value in graph.outputs() {
        let normal = run.download(&stream, value)?;
        let async_value = async_outputs.get(&value).unwrap();
        for (&a, &b) in normal.data().iter().zip(async_value.data()) {
            assert!((a - b).abs() <= 1e-6, "async mismatch: {a} vs {b}");
        }
    }
    println!("async planned execution: PASS");

    let output_stream = executor.device().new_stream()?;
    let dropped_exec = executor.execute_async_planned(
        &graph,
        &cuda_params,
        &cuda_inputs,
        &cuda_plan,
        output_stream.clone(),
    )?;
    let external_output = dropped_exec.output_tensor(0)?;
    drop(dropped_exec);
    output_stream.synchronize()?;
    let external = external_output.to_host(&output_stream)?;
    let reference = run.download(&stream, graph.outputs()[0])?;
    for (&a, &b) in external.iter().zip(reference.data()) {
        assert!(
            (a - b).abs() <= 1e-6,
            "external output mismatch: {a} vs {b}"
        );
    }
    println!("async drop external output: PASS");

    phase7f15_async_checks(
        &executor,
        &graph,
        &params,
        &cuda_params,
        &cuda_inputs,
        &cuda_plan,
        &arena,
        &device_identity,
        &context_identity,
        &stream,
    )?;

    // Deferred reclamation: queue a real kernel, record an event, drop the
    // output immediately, and observe the pending queue before/after poll.
    let big_a = CudaTensor::from_host(executor.pool(), vec![512, 512], &vec![0.5f32; 512 * 512])?;
    let big_b = CudaTensor::from_host(executor.pool(), vec![512, 512], &vec![0.25f32; 512 * 512])?;
    let mut big_c = CudaTensor::zeros(executor.pool(), vec![512, 512])?;
    executor.kernels().gemm_naive(
        &stream, &big_a, &big_b, &mut big_c, 512, 512, 512, 512, 512, false, false,
    )?;
    let drop_event =
        std::sync::Arc::new(oxide_ml_cuda::CudaEvent::new(executor.device().context())?);
    drop_event.record(&stream)?;
    big_c.mark_used(drop_event.clone())?;
    drop(big_c);
    let pending = executor.pool_stats();
    println!("pending release immediately after drop: {pending:?}");
    assert!(pending.pending_release_count > 0);
    stream.synchronize()?;
    executor.pool().poll_pending();
    let reclaimed = executor.pool_stats();
    assert_eq!(reclaimed.pending_release_count, 0);
    println!("deferred reclamation: PASS");

    // Compare the complete backward graph (not just forward outputs).
    let training = oxide_ml_autograd::build_training_graph(&graph, loss)?;
    let cpu_train_exec = oxide_ml_cpu::Executor::<f32>::new(training.graph.clone())?;
    let cpu_train = cpu_train_exec.run(&params, &inputs)?;
    let cuda_train = executor.run(&training.graph, &cuda_params, &cuda_inputs)?;
    for (id, value) in training.parameter_gradients() {
        let expected = cpu_train.value(value)?.data();
        let actual = cuda_train.download(&stream, value)?;
        for (&e, &a) in expected.iter().zip(actual.data()) {
            assert!(
                (e - a).abs() <= 5e-3 * e.abs().max(1.0),
                "gradient {id:?}: cpu={e} cuda={a}"
            );
        }
    }
    println!("mlp backward CPU vs CUDA: PASS");
    println!("pool stats: {:?}", executor.pool_stats());

    // Linear regression trained entirely on the GPU: parameters and
    // gradients stay in device memory; only the scalar loss crosses to host.
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 1]), DType::F32);
    let y = builder.input("y", Shape::from_static([4, 1]), DType::F32);
    let w = builder.try_parameter("w", Shape::from_static([1, 1]), DType::F32, true)?;
    let b = builder.try_parameter("b", Shape::from_static([1]), DType::F32, true)?;
    let product = builder.matmul(x, w)?;
    let prediction = builder.add(product, b)?;
    let loss = builder.mse_loss(prediction, y, oxide_ml_graph::Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;
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
    phase7f17_dynamic_checks(&executor, &pool)?;

    let mut trainer = CudaTrainer::new(
        CudaDevice::new(0)?,
        training,
        &host_params,
        Box::new(CudaSgd::new(0.1)?),
    )?;
    let mut train_inputs = HashMap::new();
    train_inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([4, 1], vec![0.0, 1.0, 2.0, 3.0])?,
    );
    train_inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([4, 1], vec![1.0, 3.0, 5.0, 7.0])?,
    );
    let mut last_loss = f32::INFINITY;
    for _ in 0..300 {
        last_loss = trainer.step(&train_inputs)?;
    }
    let learned_w = trainer.download_parameter(ParameterId(0))?;
    let learned_b = trainer.download_parameter(ParameterId(1))?;
    println!(
        "linear regression loss={last_loss} w={} b={}",
        learned_w.data()[0],
        learned_b.data()[0]
    );
    assert!(
        last_loss < 1e-3,
        "linear regression did not converge: {last_loss}"
    );
    assert!((learned_w.data()[0] - 2.0).abs() < 0.05);
    assert!((learned_b.data()[0] - 1.0).abs() < 0.05);
    println!("linear regression training: PASS");

    // XOR MLP: 2 -> 8 -> 1, Tanh + Sigmoid, MSE.
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4, 2]), DType::F32);
    let y = builder.input("y", Shape::from_static([4, 1]), DType::F32);
    let mlp = Mlp::new(
        &mut builder,
        "mlp",
        &[2, 8, 1],
        oxide_ml_graph::ActivationKind::Tanh,
        true,
    )?;
    let logits = mlp.forward(&mut builder, x)?;
    let prediction = builder.sigmoid(logits)?;
    let loss = builder.mse_loss(prediction, y, oxide_ml_graph::Reduction::Mean)?;
    let forward = builder.finish(&[prediction, loss])?;
    let training = oxide_ml_autograd::build_training_graph(&forward, loss)?;
    let host_params = init_parameters::<f32>(&forward, 42)?;
    let mut trainer = CudaTrainer::new(
        CudaDevice::new(0)?,
        training,
        &host_params,
        Box::new(CudaSgd::new(0.5)?),
    )?;
    let mut xor_inputs = HashMap::new();
    xor_inputs.insert(
        forward.inputs()[0],
        DenseTensor::from_vec([4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?,
    );
    xor_inputs.insert(
        forward.inputs()[1],
        DenseTensor::from_vec([4, 1], vec![0.0, 1.0, 1.0, 0.0])?,
    );
    let mut xor_loss = f32::INFINITY;
    for _ in 0..2500 {
        xor_loss = trainer.step(&xor_inputs)?;
    }
    println!("xor loss={xor_loss}");
    assert!(xor_loss < 1e-3, "XOR did not converge: {xor_loss}");
    println!("xor training: PASS");
    println!("training pool stats: {:?}", trainer.executor.pool_stats());
    Ok(())
}

fn phase7f13_event_safety_checks(
    executor: &CudaExecutor,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    let context = executor.device().context().clone();
    let pool = EventPool::new(context.clone());
    assert_eq!(pool.stats().allocation_count, 0);

    // Create / Record / Query / Synchronize / Reuse.
    let first = pool.acquire()?;
    let first_id = first.id();
    first.record(stream)?;
    assert_eq!(first.state(), Some(EventState::Recorded));
    let first_complete = first.is_complete()?;
    assert_eq!(
        first.state(),
        Some(if first_complete {
            EventState::Completed
        } else {
            EventState::Recorded
        }),
        "query must only promote a recorded event after confirmed completion"
    );
    first.synchronize()?;
    assert_eq!(first.state(), Some(EventState::Completed));
    drop(first);
    let second = pool.acquire()?;
    assert_eq!(second.id(), first_id, "completed event was not reused");
    assert_eq!(pool.stats().reuse_count, 1);
    drop(second);
    println!("event pool create/record/query/synchronize/reuse: PASS");

    // Pending retention: record immediately after a large kernel and drop the
    // lease before the GPU has finished. The event must not enter the free
    // list until a later poll observes completion.
    let big_a = CudaTensor::from_host(executor.pool(), vec![512, 512], &vec![0.5f32; 512 * 512])?;
    let big_b = CudaTensor::from_host(executor.pool(), vec![512, 512], &vec![0.25f32; 512 * 512])?;
    let mut big_c = CudaTensor::zeros(executor.pool(), vec![512, 512])?;
    executor.kernels().gemm_naive(
        stream, &big_a, &big_b, &mut big_c, 512, 512, 512, 512, 512, false, false,
    )?;
    let pending = pool.acquire()?;
    pending.record(stream)?;
    drop(pending);
    let observed = pool.stats();
    assert!(
        observed.pending_count > 0,
        "event completed too early to observe pending retention: {observed:?}"
    );
    stream.synchronize()?;
    let reclaimed = pool.stats();
    assert_eq!(reclaimed.pending_count, 0);
    assert_eq!(reclaimed.quarantine_count, 0);
    println!("event pool pending retention: PASS ({observed:?})");

    // Normal destroy path: shutdown must attempt destruction exactly once for
    // an event solely owned by the pool.
    let shutdown_pool = EventPool::new(context.clone());
    let free = shutdown_pool.acquire()?;
    drop(free);
    let report = shutdown_pool.shutdown();
    assert_eq!(report.free_events, 1);
    assert_eq!(report.destroy_attempts, 1);
    assert_eq!(report.destroy_failures, 0);
    let stats = shutdown_pool.stats();
    assert_eq!(stats.destroy_failure_count, 0);
    println!("event pool destroy statistics: PASS ({report:?})");

    // EventPool Drop with a pending event. Dropping the pool must not call a
    // synchronous wait or panic; the CUDA driver defers handle destruction.
    let drop_pool = EventPool::new(context.clone());
    let drop_free = drop_pool.acquire()?;
    drop(drop_free);
    executor.kernels().gemm_naive(
        stream, &big_a, &big_b, &mut big_c, 512, 512, 512, 512, 512, false, false,
    )?;
    let drop_pending = drop_pool.acquire()?;
    drop_pending.record(stream)?;
    drop(drop_pending);
    drop(drop_pool);
    stream.synchronize()?;
    println!("event pool drop safety: PASS");

    // Context lifetime: an event held outside the pool keeps its context
    // alive, and remains queryable after the executor's pool is gone.
    let lifetime_pool = EventPool::new(context.clone());
    let lifetime_event = {
        let lease = lifetime_pool.acquire()?;
        lease.record(stream)?;
        let event = lease.arc()?;
        drop(lease);
        drop(lifetime_pool);
        event
    };
    stream.synchronize()?;
    assert!(lifetime_event.is_complete()?);
    assert_eq!(lifetime_event.state(), EventState::Completed);
    assert_eq!(
        std::sync::Arc::as_ptr(lifetime_event.context()),
        std::sync::Arc::as_ptr(&context)
    );
    drop(lifetime_event);
    println!("event context lifetime: PASS");

    // A buffer must reject an event from a different context identity. This
    // uses a second handle to the primary context, so it exercises the
    // conservative identity check without creating a second device.
    match cuda_core::CudaContext::new(0) {
        Ok(other_context) => {
            let other_event = std::sync::Arc::new(CudaEvent::new(&other_context)?);
            let tensor = CudaTensor::zeros(executor.pool(), vec![4])?;
            assert!(
                tensor.mark_used(other_event).is_err(),
                "buffer accepted an event from a different context"
            );
            println!("buffer context mismatch rejection: PASS");
        }
        Err(error) => println!("buffer context mismatch rejection: SKIP ({error})"),
    }

    assert_eq!(executor.event_stats().quarantine_count, 0);
    assert_eq!(executor.pool().quarantine_stats().quarantine_total, 0);
    println!("event/buffer quarantine stats normal path: PASS");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn phase7f15_async_checks(
    executor: &CudaExecutor,
    graph: &Graph,
    host_params: &ParameterStore<f32>,
    cuda_params: &CudaParameterStore,
    cuda_inputs: &HashMap<ValueId, DenseTensor<f32>>,
    cuda_plan: &CudaMemoryPlan,
    arena: &CudaPlannedArena,
    device_identity: &DeviceIdentity,
    context_identity: &ContextIdentity,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(), Box<dyn std::error::Error>> {
    // wait -> Completed, collect -> Collected, double collect rejected.
    let lifecycle_stream = executor.device().new_stream()?;
    let mut execution = executor.execute_async_planned(
        graph,
        cuda_params,
        cuda_inputs,
        cuda_plan,
        lifecycle_stream.clone(),
    )?;
    let _ = execution.is_complete()?;
    let waited = execution.wait()?;
    assert_eq!(execution.state(), ExecutionState::Completed);
    let external = execution.collect_output(0)?;
    let collected = execution.collect()?.expect("completed execution collect");
    assert_eq!(execution.state(), ExecutionState::Collected);
    assert!(execution.collect().is_err(), "double collect was accepted");
    drop(execution);
    let external_host = external.to_host(&lifecycle_stream)?;
    let collected_first = collected
        .get(&graph.outputs()[0])
        .expect("first collected output");
    assert_eq!(&external_host[..external.numel()], collected_first.data());
    assert!(!waited.is_empty());
    println!("execution state/wait/collect/double collect: PASS");

    // Executor Drop while execution is alive.
    let temp_device = CudaDevice::new(0)?;
    let temp_executor = CudaExecutor::new(temp_device)?;
    let temp_params = CudaParameterStore::from_host(graph, host_params, temp_executor.pool())?;
    let temp_stream = temp_executor.device().new_stream()?;
    let mut detached = temp_executor.execute_async_planned(
        graph,
        &temp_params,
        cuda_inputs,
        cuda_plan,
        temp_stream.clone(),
    )?;
    drop(temp_executor);
    drop(temp_params);
    let detached_outputs = detached.wait()?;
    assert_eq!(detached.state(), ExecutionState::Completed);
    assert!(!detached_outputs.is_empty());
    println!("executor drop before completion: PASS");

    // Execution Drop while GPU work is still in flight. Observe deferred
    // reclamation rather than trusting a device synchronize.
    let drop_stream = executor.device().new_stream()?;
    let big_a = CudaTensor::from_host(executor.pool(), vec![512, 512], &vec![0.5f32; 512 * 512])?;
    let big_b = CudaTensor::from_host(executor.pool(), vec![512, 512], &vec![0.25f32; 512 * 512])?;
    let mut big_c = CudaTensor::zeros(executor.pool(), vec![512, 512])?;
    executor.kernels().gemm_naive(
        &drop_stream,
        &big_a,
        &big_b,
        &mut big_c,
        512,
        512,
        512,
        512,
        512,
        false,
        false,
    )?;
    let dropped_execution = executor.execute_async_planned(
        graph,
        cuda_params,
        cuda_inputs,
        cuda_plan,
        drop_stream.clone(),
    )?;
    let _ = dropped_execution.is_complete()?;
    drop(dropped_execution);
    let pending = executor.pool_stats();
    assert!(
        pending.pending_release_count > 0,
        "execution drop did not defer buffers: {pending:?}"
    );
    drop_stream.synchronize()?;
    executor.pool().poll_pending();
    let reclaimed = executor.pool_stats();
    assert_eq!(reclaimed.pending_release_count, 0);
    println!("execution drop before completion: PASS ({pending:?})");

    // External output outlives the execution, then multiple owners.
    let output_stream = executor.device().new_stream()?;
    let external_output = {
        let mut execution = executor.execute_async_planned(
            graph,
            cuda_params,
            cuda_inputs,
            cuda_plan,
            output_stream.clone(),
        )?;
        execution.wait()?;
        execution.collect_output(0)?
    };
    let clone = external_output.clone();
    drop(external_output);
    let host = clone.to_host(&output_stream)?;
    assert!(!host.is_empty());
    println!("external output/multiple owners: PASS");

    // Arena lease outlives registry entry; wait finishes the lease and the
    // arena can reclaim the slot set after completion.
    let registry = CudaArenaRegistry::default();
    let arena_from_registry = registry.get_or_create(
        graph,
        cuda_plan,
        executor.pool().clone(),
        device_identity,
        context_identity,
        2,
    )?;
    let mut arena_execution = executor.execute_async_arena(
        graph,
        cuda_params,
        cuda_inputs,
        &arena_from_registry,
        executor.device().new_stream()?,
    )?;
    drop(registry);
    let arena_outputs = arena_execution.wait()?;
    assert_eq!(arena_execution.state(), ExecutionState::Completed);
    assert!(!arena_outputs.is_empty());
    let freed = arena_from_registry.poll()?;
    assert!(freed >= 1, "arena lease was not reclaimed after completion");
    println!("arena registry removal / async lease: PASS");
    let _ = stream;
    let _ = arena;
    Ok(())
}

fn enqueue_delay(
    executor: &CudaExecutor,
    stream: &std::sync::Arc<CudaStream>,
) -> Result<(CudaTensor, CudaTensor, CudaTensor), Box<dyn std::error::Error>> {
    let a = CudaTensor::from_host(executor.pool(), vec![512, 512], &vec![0.5f32; 512 * 512])?;
    let b = CudaTensor::from_host(executor.pool(), vec![512, 512], &vec![0.25f32; 512 * 512])?;
    let mut c = CudaTensor::zeros(executor.pool(), vec![512, 512])?;
    for _ in 0..8 {
        executor.kernels().gemm_naive(
            stream, &a, &b, &mut c, 512, 512, 512, 512, 512, false, false,
        )?;
    }
    Ok((a, b, c))
}

fn phase7f16_access_checks(
    executor: &CudaExecutor,
    arena: &CudaPlannedArena,
) -> Result<(), Box<dyn std::error::Error>> {
    let s1 = executor.device().new_stream()?;
    let s2 = executor.device().new_stream()?;
    let s3 = executor.device().new_stream()?;
    let s4 = executor.device().new_stream()?;
    let source = CudaTensor::from_host(executor.pool(), vec![4], &[1.0, 2.0, 3.0, 4.0])?;

    let mut delays = vec![enqueue_delay(executor, &s1)?];
    // Write -> Read.
    let mut target = CudaTensor::zeros(executor.pool(), vec![4])?;
    let mut out = CudaTensor::zeros(executor.pool(), vec![4])?;
    executor.kernels().broadcast(&s1, &source, &mut target)?;
    executor.kernels().broadcast(&s2, &target, &mut out)?;
    let stats = target.access_stats();
    assert_eq!(stats.cross_stream_waits, 1, "{stats:?}");
    s1.synchronize()?;
    s2.synchronize()?;
    assert_eq!(out.to_host(&s2)?, vec![1.0, 2.0, 3.0, 4.0]);
    println!("access write/read: PASS ({stats:?})");

    // Write -> Write.
    let mut target_ww = CudaTensor::zeros(executor.pool(), vec![4])?;
    delays.push(enqueue_delay(executor, &s1)?);
    executor.kernels().broadcast(&s1, &source, &mut target_ww)?;
    executor.kernels().broadcast(&s2, &source, &mut target_ww)?;
    let stats = target_ww.access_stats();
    assert_eq!(stats.cross_stream_waits, 1, "{stats:?}");
    s1.synchronize()?;
    s2.synchronize()?;
    println!("access write/write: PASS ({stats:?})");

    // ReadWrite -> Read.
    let mut shared = CudaTensor::from_host(executor.pool(), vec![4], &[1.0, 2.0, 3.0, 4.0])?;
    delays.push(enqueue_delay(executor, &s1)?);
    executor.kernels().scale(&s1, &mut shared, 2.0)?;
    executor.kernels().broadcast(&s2, &shared, &mut out)?;
    let stats = shared.access_stats();
    assert_eq!(stats.cross_stream_waits, 1, "{stats:?}");
    s1.synchronize()?;
    s2.synchronize()?;
    println!("access read-write/read: PASS ({stats:?})");

    // Same-stream ordering must not add Event waits.
    let mut same_target = CudaTensor::zeros(executor.pool(), vec![4])?;
    delays.push(enqueue_delay(executor, &s1)?);
    executor
        .kernels()
        .broadcast(&s1, &source, &mut same_target)?;
    executor.kernels().broadcast(&s1, &same_target, &mut out)?;
    let stats = same_target.access_stats();
    assert_eq!(stats.event_waits, 0, "{stats:?}");
    assert!(stats.same_stream_elided_waits >= 1, "{stats:?}");
    s1.synchronize()?;
    println!("access same-stream elision: PASS ({stats:?})");

    // Three streams: Write -> two Readers -> Write.
    let mut chain = CudaTensor::zeros(executor.pool(), vec![4])?;
    let mut out_b = CudaTensor::zeros(executor.pool(), vec![4])?;
    let mut out_c = CudaTensor::zeros(executor.pool(), vec![4])?;
    delays.push(enqueue_delay(executor, &s1)?);
    delays.push(enqueue_delay(executor, &s2)?);
    delays.push(enqueue_delay(executor, &s3)?);
    executor.kernels().broadcast(&s1, &source, &mut chain)?;
    executor.kernels().broadcast(&s2, &chain, &mut out_b)?;
    executor.kernels().broadcast(&s3, &chain, &mut out_c)?;
    executor.kernels().scale(&s4, &mut chain, 3.0)?;
    let stats = chain.access_stats();
    assert_eq!(stats.cross_stream_waits, 3, "{stats:?}");
    assert_eq!(stats.pending_readers, 0, "{stats:?}");
    s1.synchronize()?;
    s2.synchronize()?;
    s3.synchronize()?;
    s4.synchronize()?;
    println!("access three-stream readers/writer: PASS ({stats:?})");

    // Alias / Clone share the same access state.
    let alias = shared.reshape(vec![4])?;
    assert_eq!(alias.identity(), shared.identity());
    delays.push(enqueue_delay(executor, &s1)?);
    executor.kernels().broadcast(&s1, &alias, &mut out)?;
    let alias_stats = shared.access_stats();
    assert!(alias_stats.pending_readers >= 1, "{alias_stats:?}");
    s1.synchronize()?;
    println!("access alias/clone shared state: PASS ({alias_stats:?})");

    // Pool reuse: same physical BufferId, new generation.
    let first = executor.pool().acquire(37)?;
    let first_identity = first.identity();
    drop(first);
    executor.pool().poll_pending();
    let second = executor.pool().acquire(37)?;
    let second_identity = second.identity();
    assert_eq!(first_identity.id, second_identity.id);
    assert_eq!(second_identity.generation, first_identity.generation + 1);
    drop(second);
    println!("access pool generation: PASS");

    // Arena slot reuse: same BufferId, new generation, no stale access state.
    let arena_generation = CudaPlannedArena::new(arena.plan().clone(), executor.pool().clone(), 1)?;
    let gen_stream = executor.device().new_stream()?;
    let mut first_lease = arena_generation.lease(&gen_stream)?;
    let (slots, events, streams) = first_lease.take()?;
    let first_slot = slots[0].identity();
    first_lease.finish(slots, events, streams)?;
    gen_stream.synchronize()?;
    let _ = arena_generation.poll()?;
    let mut second_lease = arena_generation.lease(&gen_stream)?;
    let (slots, events, streams) = second_lease.take()?;
    let second_slot = slots[0].identity();
    second_lease.finish(slots, events, streams)?;
    assert_eq!(first_slot.id, second_slot.id);
    assert_eq!(second_slot.generation, first_slot.generation + 1);
    println!("access arena generation: PASS");

    let _ = (&s4, &stats);
    Ok(())
}

fn phase7f17_dynamic_checks(
    executor: &CudaExecutor,
    pool: &CudaBufferPool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = GraphBuilder::new();
    let batch = builder.symbol("batch");
    let x = builder.input(
        "x",
        Shape::new(vec![Dim::Symbol(batch), Dim::Static(2)]),
        DType::F32,
    );
    let w = builder.try_parameter("w", Shape::from_static([2, 1]), DType::F32, true)?;
    let y = builder.matmul(x, w)?;
    let graph = builder.finish(&[y])?;

    let mut params = ParameterStore::<f32>::new();
    params.add(
        ParameterId(0),
        "w",
        DenseTensor::from_vec([2, 1], vec![0.5, -0.25])?,
        true,
    )?;
    let cuda_params = CudaParameterStore::from_host(&graph, &params, pool)?;
    let plan = MemoryPlanner::default().plan(&graph)?;
    let plan = CudaMemoryPlan::from_memory_plan(&graph, &plan)?;

    for batch_size in [1usize, 4, 8] {
        let mut data = Vec::with_capacity(batch_size * 2);
        for index in 0..batch_size {
            data.push(index as f32 + 1.0);
            data.push((index as f32 + 1.0) * 2.0);
        }
        let input = DenseTensor::from_vec([batch_size, 2], data)?;
        let mut inputs = HashMap::new();
        inputs.insert(graph.inputs()[0], input);
        let expected = (0..batch_size)
            .map(|index| {
                let a = index as f32 + 1.0;
                let b = a * 2.0;
                a * 0.5 + b * -0.25
            })
            .collect::<Vec<_>>();

        let signature = executor.shape_signature(&graph, &inputs)?;
        assert_eq!(signature.as_entries(), &[(batch.0, batch_size)]);
        let normal = executor.run(&graph, &cuda_params, &inputs)?;
        let normal_out = normal.output(0)?;
        assert_eq!(normal_out.shape(), vec![batch_size, 1]);
        let normal_host = normal_out.to_host(&executor.device().default_stream())?;
        assert_eq!(&normal_host[..batch_size], expected.as_slice());

        let planned = executor.run_planned(&graph, &cuda_params, &inputs, &plan)?;
        let planned_out = planned.output(0)?;
        let planned_host = planned_out.to_host(&executor.device().default_stream())?;
        assert_eq!(&planned_host[..batch_size], expected.as_slice());

        let (specialized, specialized_signature) = executor.specialize_graph(&graph, &inputs)?;
        assert_eq!(specialized_signature.as_entries(), &[(batch.0, batch_size)]);
        let specialized_plan = MemoryPlanner::default().plan(&specialized)?;
        let specialized_plan = CudaMemoryPlan::from_memory_plan(&specialized, &specialized_plan)?;
        let specialized_arena = CudaPlannedArena::new(specialized_plan, pool.clone(), 2)?;
        for _ in 0..2 {
            let arena_run = executor.run_arena(
                &specialized,
                &cuda_params,
                &inputs,
                &specialized_arena,
                executor.device().default_stream(),
            )?;
            let arena_host = arena_run
                .output(0)?
                .to_host(&executor.device().default_stream())?;
            assert_eq!(&arena_host[..batch_size], expected.as_slice());
        }
    }
    println!("dynamic symbol batch (normal/planned): PASS");
    Ok(())
}
