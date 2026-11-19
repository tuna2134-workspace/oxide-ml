//! CPU/CUDA parity for the new elementwise IR ops and BatchNorm (which uses
//! sqrt + reduction + broadcasting).

use oxide_ml_core::{DType, DenseTensor, Shape};
use oxide_ml_cuda::{CudaDevice, CudaError, CudaModel};
use oxide_ml_graph::GraphBuilder;
use oxide_ml_model::{
    CompiledModel, InputSpec, ModelCompileOptions, ModelContext, Module, ModulePath,
    ParameterRegistry, Tensor,
};
use oxide_ml_nn::modules::BatchNorm;

fn compare(
    device: CudaDevice,
    name: &str,
    graph: oxide_ml_graph::Graph,
    inputs: &[(&str, Vec<f32>)],
    shape: Vec<usize>,
) -> oxide_ml_cuda::Result<f32> {
    let parameters = oxide_ml_core::ParameterStore::<f32>::new();
    let mut cpu_inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
    let mut host = std::collections::HashMap::new();
    for (index, (_, values)) in inputs.iter().enumerate() {
        let id = graph.inputs()[index];
        cpu_inputs.insert(id, DenseTensor::from_vec(shape.clone(), values.clone())?);
        host.insert(id, DenseTensor::from_vec(shape.clone(), values.clone())?);
    }
    let cpu = oxide_ml_cpu::Executor::<f32>::new(graph.clone())?.run(&parameters, &cpu_inputs)?;
    let cpu_out = cpu.output(0)?.data().to_vec();

    let executor = oxide_ml_cuda::CudaExecutor::new(device)?;
    let cuda_params =
        oxide_ml_cuda::CudaParameterStore::from_host(&graph, &parameters, executor.pool())?;
    let run = executor.run(&graph, &cuda_params, &host)?;
    let stream = executor.device().default_stream();
    let gpu_out = run.download(&stream, graph.outputs()[0])?;
    let mut max_diff = 0.0f32;
    for (a, b) in cpu_out.iter().zip(gpu_out.data()) {
        max_diff = max_diff.max((a - b).abs());
    }
    if max_diff > 1e-4 {
        println!("[FAIL] {name}: max |diff| = {max_diff:.3e}");
        return Err(CudaError::Execution(format!("{name} mismatch")));
    }
    println!("[PASS] {name}: max |diff| = {max_diff:.3e}");
    Ok(max_diff)
}

fn unary_check(
    device: CudaDevice,
    name: &str,
    op: oxide_ml_graph::Op,
) -> oxide_ml_cuda::Result<()> {
    let shape = vec![4];
    let values = vec![0.25f32, 1.0, 2.5, 4.0];
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4]), DType::F32);
    let y = builder
        .single(op, oxide_ml_graph::OpAttrs::None, &[x])
        .unwrap();
    let graph = builder.finish(&[y]).unwrap();
    compare(device, name, graph, &[("x", values)], shape).map(|_| ())
}

#[derive(Debug)]
struct BnModel {
    norm: BatchNorm,
}

impl Module for BnModel {
    fn module_type(&self) -> &'static str {
        "BnModel"
    }

    fn visit_parameters(
        &self,
        path: &ModulePath,
        registry: &mut ParameterRegistry,
    ) -> oxide_ml_core::Result<()> {
        self.norm.visit_parameters(path, registry)
    }

    fn forward(
        &self,
        cx: &mut ModelContext,
        input: oxide_ml_core::ValueId,
    ) -> oxide_ml_core::Result<Vec<oxide_ml_core::ValueId>> {
        self.norm.forward(cx, input)
    }

    fn visit_tensors(
        &self,
        path: &ModulePath,
        visitor: &mut dyn FnMut(
            &ModulePath,
            &DenseTensor<f32>,
            oxide_ml_model::ParameterRole,
        ) -> oxide_ml_core::Result<()>,
    ) -> oxide_ml_core::Result<()> {
        self.norm.visit_tensors(path, visitor)
    }
}

fn batchnorm_check(device: CudaDevice) -> oxide_ml_cuda::Result<()> {
    let model = BnModel {
        norm: BatchNorm::new(2),
    };
    let specs = [InputSpec::unbatched("input0", [2, 2])];
    let cpu = CompiledModel::compile(&model, &specs, ModelCompileOptions::training())?;
    let store = cpu.parameter_store(&model)?;
    let input = Tensor::<f32>::from_vec([2, 2], vec![1.0, 1.0, 3.0, 3.0])?;
    let cpu_out = cpu.run(&store, &[("input0", &input)])?;
    let gpu = CudaModel::compile(device, &model, &specs, ModelCompileOptions::training())?;
    let gpu_out = gpu.run(&[("input0", input.dense().clone())])?;
    let mut max_diff = 0.0f32;
    for (a, b) in cpu_out[0].data().iter().zip(gpu_out[0].data()) {
        max_diff = max_diff.max((a - b).abs());
    }
    println!(
        "[{}] BatchNorm(training) CPU/CUDA: max |diff| = {max_diff:.3e}",
        if max_diff < 1e-4 { "PASS" } else { "FAIL" }
    );
    if max_diff >= 1e-4 {
        return Err(CudaError::Execution("batchnorm mismatch".into()));
    }
    Ok(())
}

fn main() -> oxide_ml_cuda::Result<()> {
    unary_check(CudaDevice::new(0)?, "sqrt", oxide_ml_graph::Op::Sqrt)?;
    unary_check(CudaDevice::new(0)?, "exp", oxide_ml_graph::Op::Exp)?;
    unary_check(CudaDevice::new(0)?, "log", oxide_ml_graph::Op::Log)?;
    unary_check(CudaDevice::new(0)?, "abs", oxide_ml_graph::Op::Abs)?;

    // AbsBackward(x, dy) = dy * sign(x)
    let shape = vec![4];
    let x_values = vec![-2.0f32, -1.0, 0.0, 3.0];
    let dy_values = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([4]), DType::F32);
    let dy = builder.input("dy", Shape::from_static([4]), DType::F32);
    let out = builder.abs_backward(x, dy).unwrap();
    let graph = builder.finish(&[out]).unwrap();
    let parameters = oxide_ml_core::ParameterStore::<f32>::new();
    let mut inputs = oxide_ml_cpu::ExecutionInputs::<f32>::new();
    inputs.insert(x, DenseTensor::from_vec(shape.clone(), x_values.clone())?);
    inputs.insert(dy, DenseTensor::from_vec(shape.clone(), dy_values.clone())?);
    let cpu = oxide_ml_cpu::Executor::<f32>::new(graph.clone())?.run(&parameters, &inputs)?;
    let cpu_out = cpu.output(0)?.data().to_vec();
    let executor = oxide_ml_cuda::CudaExecutor::new(device)?;
    let cuda_params =
        oxide_ml_cuda::CudaParameterStore::from_host(&graph, &parameters, executor.pool())?;
    let mut host = std::collections::HashMap::new();
    host.insert(x, DenseTensor::from_vec(shape.clone(), x_values)?);
    host.insert(dy, DenseTensor::from_vec(shape.clone(), dy_values)?);
    let run = executor.run(&graph, &cuda_params, &host)?;
    let stream = executor.device().default_stream();
    let gpu_out = run.download(&stream, out)?;
    let mut max_diff = 0.0f32;
    for (a, b) in cpu_out.iter().zip(gpu_out.data()) {
        max_diff = max_diff.max((a - b).abs());
    }
    println!(
        "[{}] abs_backward: max |diff| = {max_diff:.3e}",
        if max_diff < 1e-4 { "PASS" } else { "FAIL" }
    );
    if max_diff >= 1e-4 {
        return Err(CudaError::Execution("abs_backward mismatch".into()));
    }

    batchnorm_check(CudaDevice::new(0)?)?;
    println!("NEW_OPS_CHECK: PASS");
    Ok(())
}
