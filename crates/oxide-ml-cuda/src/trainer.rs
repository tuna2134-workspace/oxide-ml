use std::collections::HashMap;

use oxide_ml_autograd::TrainingGraph;
use oxide_ml_core::{DenseTensor, ParameterId, ValueId};

use crate::buffer::CudaTensor;
use crate::device::CudaDevice;
use crate::error::Result;
use crate::executor::CudaExecutor;
use crate::optimizer::CudaOptimizer;
use crate::param::CudaParameterStore;

pub struct CudaTrainer {
    pub executor: CudaExecutor,
    pub training: TrainingGraph,
    pub parameters: CudaParameterStore,
    pub optimizer: Box<dyn CudaOptimizer>,
}

impl CudaTrainer {
    pub fn new(
        device: CudaDevice,
        training: TrainingGraph,
        host_parameters: &oxide_ml_core::ParameterStore<f32>,
        optimizer: Box<dyn CudaOptimizer>,
    ) -> Result<Self> {
        let executor = CudaExecutor::new(device)?;
        let parameters =
            CudaParameterStore::from_host(&training.graph, host_parameters, executor.pool())?;
        Ok(CudaTrainer {
            executor,
            training,
            parameters,
            optimizer,
        })
    }

    pub fn loss(&self, run: &crate::executor::CudaRun) -> Result<f32> {
        let tensor = run.download(&self.executor.device().default_stream(), self.training.loss)?;
        if tensor.numel() != 1 {
            return Err(crate::CudaError::Shape(
                "training loss must be scalar".into(),
            ));
        }
        Ok(tensor.data()[0])
    }

    pub fn step(&mut self, inputs: &HashMap<ValueId, DenseTensor<f32>>) -> Result<f32> {
        let run = self
            .executor
            .run(&self.training.graph, &self.parameters, inputs)?;
        let loss = self.loss(&run)?;
        let mut gradients: HashMap<ParameterId, CudaTensor> = HashMap::new();
        for (id, value) in self.training.parameter_gradients() {
            gradients.insert(id, run.value(value)?.clone());
        }
        // Drop the run's parameter aliases before mutating the parameter store.
        // The optimizer needs a unique DeviceBuffer for each updated tensor.
        drop(run);
        let stream = self.executor.device().default_stream();
        self.optimizer.step(
            self.executor.kernels(),
            &stream,
            self.executor.pool(),
            &mut self.parameters,
            &gradients,
        )?;
        stream.synchronize()?;
        Ok(loss)
    }

    pub fn download_parameter(&self, id: ParameterId) -> Result<DenseTensor<f32>> {
        let stream = self.executor.device().default_stream();
        self.parameters.download(id, &stream)
    }
}
