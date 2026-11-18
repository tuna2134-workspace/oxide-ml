use std::collections::HashMap;
use std::sync::Arc;

use cuda_core::CudaStream;
use oxide_ml_core::ParameterId;

use crate::buffer::{CudaBufferPool, CudaTensor};
use crate::error::{CudaError, Result};
use crate::kernels::{AdamWParams, CudaKernels};
use crate::param::CudaParameterStore;

pub trait CudaOptimizer {
    fn step(
        &mut self,
        kernels: &CudaKernels,
        stream: &Arc<CudaStream>,
        pool: &CudaBufferPool,
        parameters: &mut CudaParameterStore,
        gradients: &HashMap<ParameterId, CudaTensor>,
    ) -> Result<()>;
}

pub struct CudaSgd {
    pub learning_rate: f32,
}

impl CudaSgd {
    pub fn new(learning_rate: f32) -> Result<Self> {
        if !(learning_rate.is_finite() && learning_rate > 0.0) {
            return Err(CudaError::Shape("learning rate must be positive".into()));
        }
        Ok(CudaSgd { learning_rate })
    }
}

impl CudaOptimizer for CudaSgd {
    fn step(
        &mut self,
        kernels: &CudaKernels,
        stream: &Arc<CudaStream>,
        _pool: &CudaBufferPool,
        parameters: &mut CudaParameterStore,
        gradients: &HashMap<ParameterId, CudaTensor>,
    ) -> Result<()> {
        for (id, gradient) in gradients {
            let parameter = parameters.get_mut(*id)?;
            kernels.sgd(stream, parameter, gradient, self.learning_rate)?;
        }
        Ok(())
    }
}

struct AdamState {
    m: CudaTensor,
    v: CudaTensor,
}

pub struct CudaAdamW {
    pub config: AdamWParams,
    t: u32,
    state: HashMap<ParameterId, AdamState>,
}

impl CudaAdamW {
    pub fn new(config: AdamWParams) -> Result<Self> {
        if !(config.learning_rate.is_finite() && config.learning_rate > 0.0) {
            return Err(CudaError::Shape("learning rate must be positive".into()));
        }
        Ok(CudaAdamW {
            config,
            t: 0,
            state: HashMap::new(),
        })
    }
}

impl CudaOptimizer for CudaAdamW {
    fn step(
        &mut self,
        kernels: &CudaKernels,
        stream: &Arc<CudaStream>,
        pool: &CudaBufferPool,
        parameters: &mut CudaParameterStore,
        gradients: &HashMap<ParameterId, CudaTensor>,
    ) -> Result<()> {
        self.t += 1;
        let bias_correction1 = 1.0 - self.config.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.config.beta2.powi(self.t as i32);
        for (id, gradient) in gradients {
            let shape = gradient.shape().to_vec();
            let state = match self.state.get_mut(id) {
                Some(state) => state,
                None => {
                    let state = AdamState {
                        m: CudaTensor::zeros(pool, shape.clone())?,
                        v: CudaTensor::zeros(pool, shape)?,
                    };
                    self.state.insert(*id, state);
                    self.state.get_mut(id).expect("just inserted")
                }
            };
            let parameter = parameters.get_mut(*id)?;
            kernels.adamw(
                stream,
                parameter,
                gradient,
                &mut state.m,
                &mut state.v,
                self.config,
                bias_correction1,
                bias_correction2,
            )?;
        }
        Ok(())
    }
}
