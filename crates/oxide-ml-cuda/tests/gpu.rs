use std::collections::HashMap;

use oxide_ml_core::{DType, DenseTensor, ParameterId, ParameterStore, Shape};
use oxide_ml_cpu::ExecutionInputs;
use oxide_ml_cuda::{CudaBufferPool, CudaDevice, CudaExecutor, CudaParameterStore, CudaTensor};
use oxide_ml_graph::GraphBuilder;

#[test]
fn device_is_a_cuda_t4_or_later() {
    let device = CudaDevice::new(0).expect("CUDA device 0");
    let info = device.info();
    assert!(info.total_memory > 0);
    assert!(info.warp_size == 32);
    assert!(info.compute_capability.0 >= 7);
    println!(
        "GPU {} cc {}.{} mem {} warp {}",
        info.name,
        info.compute_capability.0,
        info.compute_capability.1,
        info.total_memory,
        info.warp_size
    );
}

#[test]
fn device_buffer_roundtrip() {
    let device = CudaDevice::new(0).unwrap();
    let stream = device.default_stream();
    let pool = CudaBufferPool::new(device.context().clone(), stream.clone());
    let host = vec![1.0f32, -2.0, 3.5, 0.0];
    let tensor = CudaTensor::from_host(&pool, vec![2, 2], &host).unwrap();
    let back = tensor.to_host(&stream).unwrap();
    assert_eq!(host, back);
}

#[test]
fn cuda_mlp_forward_matches_cpu() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 3]), DType::F32);
    let y = builder.input("y", Shape::from_static([2, 1]), DType::F32);
    let w1 = builder
        .try_parameter("w1", Shape::from_static([3, 4]), DType::F32, true)
        .unwrap();
    let b1 = builder
        .try_parameter("b1", Shape::from_static([4]), DType::F32, true)
        .unwrap();
    let w2 = builder
        .try_parameter("w2", Shape::from_static([4, 1]), DType::F32, true)
        .unwrap();
    let b2 = builder
        .try_parameter("b2", Shape::from_static([1]), DType::F32, true)
        .unwrap();
    let h = builder.matmul(x, w1).unwrap();
    let h = builder.add(h, b1).unwrap();
    let h = builder.sigmoid(h).unwrap();
    let y_hat = builder.matmul(h, w2).unwrap();
    let y_hat = builder.add(y_hat, b2).unwrap();
    let loss = builder
        .mse_loss(y_hat, y, oxide_ml_graph::Reduction::Mean)
        .unwrap();
    let graph = builder.finish(&[y_hat, loss]).unwrap();

    let mut host_params = ParameterStore::<f32>::new();
    host_params
        .add(
            ParameterId(0),
            "w1",
            DenseTensor::from_vec(
                [3, 4],
                vec![
                    0.1, 0.2, -0.3, 0.4, 0.2, -0.5, 0.3, 0.1, -0.2, 0.4, 0.1, 0.2,
                ],
            )
            .unwrap(),
            true,
        )
        .unwrap();
    host_params
        .add(
            ParameterId(1),
            "b1",
            DenseTensor::from_vec([4], vec![0.1, -0.2, 0.0, 0.3]).unwrap(),
            true,
        )
        .unwrap();
    host_params
        .add(
            ParameterId(2),
            "w2",
            DenseTensor::from_vec([4, 1], vec![0.5, -0.4, 0.2, 0.1]).unwrap(),
            true,
        )
        .unwrap();
    host_params
        .add(
            ParameterId(3),
            "b2",
            DenseTensor::from_vec([1], vec![0.05]).unwrap(),
            true,
        )
        .unwrap();

    let mut host_inputs = ExecutionInputs::<f32>::new();
    host_inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec([2, 3], vec![1.0, 2.0, -1.0, 0.5, -1.0, 2.0]).unwrap(),
    );
    host_inputs.insert(
        graph.inputs()[1],
        DenseTensor::from_vec([2, 1], vec![1.0, -1.0]).unwrap(),
    );
    let cpu = oxide_ml_cpu::Executor::<f32>::new(graph.clone()).unwrap();
    let cpu_outputs = cpu.run(&host_params, &host_inputs).unwrap();

    let device = CudaDevice::new(0).unwrap();
    let stream = device.default_stream();
    let pool = CudaBufferPool::new(device.context().clone(), stream.clone());
    let cuda_params = CudaParameterStore::from_host(&graph, &host_params, &pool).unwrap();
    let mut cuda_inputs = HashMap::new();
    cuda_inputs.insert(
        graph.inputs()[0],
        host_inputs.get(graph.inputs()[0]).unwrap().clone(),
    );
    cuda_inputs.insert(
        graph.inputs()[1],
        host_inputs.get(graph.inputs()[1]).unwrap().clone(),
    );
    let executor = CudaExecutor::new(device).unwrap();
    let run = executor.run(&graph, &cuda_params, &cuda_inputs).unwrap();
    for (index, &value) in graph.outputs().iter().enumerate() {
        let cpu_value = cpu_outputs.output(index).unwrap().data();
        let cuda_value = run.download(&stream, value).unwrap();
        assert_eq!(cpu_value.len(), cuda_value.numel());
        for (&expected, &actual) in cpu_value.iter().zip(cuda_value.data()) {
            assert!(
                (expected - actual).abs() <= 2e-3 * expected.abs().max(1.0),
                "output {index}: cpu={expected} cuda={actual}"
            );
        }
    }
}

#[test]
fn unsupported_operation_is_rejected() {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([2, 2]), DType::F32);
    let y = builder.input("y", Shape::from_static([2, 2]), DType::F32);
    let out = builder.softmax(x, -1).unwrap();
    let _ = y;
    let _graph = builder.finish(&[out]).unwrap();
    // Softmax is implemented; use CrossEntropyLoss instead.
    let mut builder = GraphBuilder::new();
    let logits = builder.input("logits", Shape::from_static([2, 2]), DType::F32);
    let targets = builder.input("targets", Shape::from_static([2]), DType::F32);
    let loss = builder
        .cross_entropy_loss(logits, targets, oxide_ml_graph::Reduction::Mean)
        .unwrap();
    let graph2 = builder.finish(&[loss]).unwrap();
    let device = CudaDevice::new(0).unwrap();
    let stream = device.default_stream();
    let pool = CudaBufferPool::new(device.context().clone(), stream);
    let params =
        CudaParameterStore::from_host(&graph2, &ParameterStore::<f32>::new(), &pool).unwrap();
    let mut inputs = HashMap::new();
    inputs.insert(graph2.inputs()[0], DenseTensor::zeros([2, 2]));
    inputs.insert(graph2.inputs()[1], DenseTensor::zeros([2]));
    let executor = CudaExecutor::new(device).unwrap();
    let result = executor.run(&graph2, &params, &inputs);
    assert!(result.is_err());
    let error = result.err().unwrap();
    assert!(error.to_string().contains("not implemented"));
}
