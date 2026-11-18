use std::collections::HashMap;

use oxide_ml_core::{DenseTensor, ParameterId, ParameterStore};
use oxide_ml_graph::Graph;

use crate::buffer::{CudaBufferPool, CudaTensor};
use crate::error::{CudaError, Result};

pub struct CudaParameterStore {
    values: HashMap<ParameterId, CudaTensor>,
}

impl CudaParameterStore {
    pub fn from_host(
        graph: &Graph,
        source: &ParameterStore<f32>,
        pool: &CudaBufferPool,
    ) -> Result<Self> {
        let mut values = HashMap::new();
        for parameter in graph.parameters() {
            if parameter.dtype != oxide_ml_core::DType::F32 {
                return Err(CudaError::Unsupported(format!(
                    "parameter {:?} has dtype {}; CUDA backend currently supports F32 only",
                    parameter.name, parameter.dtype
                )));
            }
            let host = source.get(parameter.id)?;
            let tensor = CudaTensor::from_host(
                pool,
                parameter
                    .shape
                    .static_dims()
                    .ok_or_else(|| CudaError::Shape("parameter shape must be static".into()))?,
                host.data.data(),
            )?;
            values.insert(parameter.id, tensor);
        }
        Ok(CudaParameterStore { values })
    }

    pub fn get(&self, id: ParameterId) -> Result<&CudaTensor> {
        self.values
            .get(&id)
            .ok_or_else(|| CudaError::Internal(format!("missing CUDA parameter {id}")))
    }

    pub fn get_mut(&mut self, id: ParameterId) -> Result<&mut CudaTensor> {
        self.values
            .get_mut(&id)
            .ok_or_else(|| CudaError::Internal(format!("missing CUDA parameter {id}")))
    }

    pub fn iter(&self) -> impl Iterator<Item = (ParameterId, &CudaTensor)> {
        self.values.iter().map(|(&id, tensor)| (id, tensor))
    }

    pub fn download(
        &self,
        id: ParameterId,
        stream: &cuda_core::CudaStream,
    ) -> Result<DenseTensor<f32>> {
        let tensor = self.get(id)?;
        Ok(DenseTensor::from_vec(
            tensor.shape(),
            tensor.to_host(stream)?,
        )?)
    }
}
