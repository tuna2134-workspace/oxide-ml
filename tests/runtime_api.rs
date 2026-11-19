//! Runtime selection, cached compilation and compiler-report visibility.

use oxide_ml::prelude::*;

#[derive(Module)]
struct Net {
    fc1: Linear,
    fc2: Linear,
}

impl Net {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = relu(self.fc1.forward(x));
        self.fc2.forward(hidden)
    }
}

fn net() -> Net {
    Net {
        fc1: Linear::with_seed(2, 4, true, 1),
        fc2: Linear::with_seed(4, 1, true, 2),
    }
}

#[test]
fn runtime_compiles_and_runs_through_its_cache() {
    let runtime = Runtime::builder()
        .device(Device::Cpu)
        .cache_capacity(2)
        .build()
        .unwrap();
    assert_eq!(runtime.device(), Device::Cpu);
    let model = net();
    let specs = [InputSpec::unbatched("input0", [2, 2])];
    let compiled = runtime
        .compile(&model, &specs, ModelCompileOptions::default())
        .unwrap();
    let again = runtime
        .compile(&model, &specs, ModelCompileOptions::default())
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&compiled, &again));
    assert_eq!(runtime.cache().stats().hits, 1);

    let store = compiled.parameter_store(&model).unwrap();
    let input = Tensor::<f32>::from_vec([2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
    let direct = compiled.run(&store, &[("input0", &input)]).unwrap();
    let via_runtime = compiled
        .run_with(&runtime, &store, &[("input0", &input)])
        .unwrap();
    assert_eq!(direct[0].data(), via_runtime[0].data());
}

#[test]
fn cuda_runtime_is_explicit_not_a_fallback() {
    let device = Device::cuda(0).unwrap();
    assert_eq!(device, Device::Cuda(0));
    let runtime = Runtime::builder().device(device).build().unwrap();
    let model = net();
    let error = runtime
        .compile(
            &model,
            &[InputSpec::unbatched("input0", [1, 2])],
            ModelCompileOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");

    let cpu = Runtime::builder().build().unwrap();
    let compiled = cpu
        .compile(
            &model,
            &[InputSpec::unbatched("input0", [1, 2])],
            ModelCompileOptions::default(),
        )
        .unwrap();
    let store = compiled.parameter_store(&model).unwrap();
    let input = Tensor::<f32>::from_vec([1, 2], vec![0.1, 0.2]).unwrap();
    let error = compiled
        .run_with(&runtime, &store, &[("input0", &input)])
        .unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
}

#[test]
fn compiler_report_and_kernel_plan_are_visible() {
    let model = net();
    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [2, 2])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    let report = compiled.report();
    assert!(report.optimized);
    assert!(report.kernels > 0);
    assert_eq!(report.kernels, compiled.kernel_plan().selected());
    assert_eq!(
        compiled.kernel_plan().entries.len(),
        compiled.forward_graph().nodes().len()
    );
    for entry in &compiled.kernel_plan().entries {
        assert!(entry.provider.is_some(), "{} has no provider", entry.op);
    }
    assert!(report.memory_plan_bytes > 0);
}
