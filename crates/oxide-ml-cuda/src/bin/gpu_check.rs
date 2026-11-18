use std::collections::HashMap;

use cuda_core::CudaStream;
use oxide_ml_core::{DType, DenseTensor, ParameterId, ParameterStore, Shape};
use oxide_ml_cpu::ExecutionInputs;
use oxide_ml_cuda::{
    ArenaContextId, ArenaDescriptor, ArenaKey, BufferAccess, BufferAccessState, CudaArenaRegistry,
    CudaBufferPool, CudaDevice, CudaEvent, CudaExecutor, CudaMemoryPlan, CudaParameterStore,
    CudaPlannedArena, CudaSgd, CudaTensor, CudaTrainer, EventPool, EventState,
};
use oxide_ml_graph::GraphBuilder;
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
    let mut access = BufferAccessState::default();
    let event_a = std::sync::Arc::new(oxide_ml_cuda::CudaEvent::new(executor.device().context())?);
    executor
        .kernels()
        .broadcast(&access_stream_a, &shared, &mut read_a)?;
    event_a.record(&access_stream_a)?;
    access.record(access_stream_a.clone(), BufferAccess::Read, event_a);
    let event_b = std::sync::Arc::new(oxide_ml_cuda::CudaEvent::new(executor.device().context())?);
    executor
        .kernels()
        .broadcast(&access_stream_b, &shared, &mut read_b)?;
    event_b.record(&access_stream_b)?;
    access.record(access_stream_b.clone(), BufferAccess::Read, event_b);
    assert_eq!(access.wait_for(&access_stream_c, BufferAccess::Read)?, 0);
    let waits = access.wait_for(&access_stream_c, BufferAccess::Write)?;
    assert_eq!(waits, 2);
    let event_c = std::sync::Arc::new(oxide_ml_cuda::CudaEvent::new(executor.device().context())?);
    executor
        .kernels()
        .scale(&access_stream_c, &mut shared, 2.0)?;
    event_c.record(&access_stream_c)?;
    access.record(access_stream_c.clone(), BufferAccess::Write, event_c);
    access_stream_a.synchronize()?;
    access_stream_b.synchronize()?;
    access_stream_c.synchronize()?;
    assert_eq!(read_a.to_host(&access_stream_a)?, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_b.to_host(&access_stream_b)?, vec![1.0, 2.0, 3.0, 4.0]);
    println!("buffer access read/read/write: PASS");
    println!("event pool stats: {:?}", executor.event_stats());

    let async_stream = executor.device().new_stream()?;
    let async_exec = executor.execute_async_planned(
        &graph,
        &cuda_params,
        &cuda_inputs,
        &cuda_plan,
        async_stream,
    )?;
    assert!(async_exec.is_complete()? || !async_exec.is_complete()?);
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
