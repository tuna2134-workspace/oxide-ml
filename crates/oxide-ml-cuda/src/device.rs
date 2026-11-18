use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, IntoResult};

use crate::error::Result;

#[derive(Clone, Debug)]
pub struct CudaDeviceInfo {
    pub ordinal: usize,
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub free_memory: usize,
    pub max_threads_per_block: u32,
    pub max_block_dim: (u32, u32, u32),
    pub max_grid_dim: (u32, u32, u32),
    pub shared_memory_per_block: u32,
    pub warp_size: u32,
}

#[derive(Clone)]
pub struct CudaDevice {
    context: Arc<CudaContext>,
    info: CudaDeviceInfo,
}

impl CudaDevice {
    pub fn new(ordinal: usize) -> Result<Self> {
        let context = CudaContext::new(ordinal)?;
        let name = context.device_name()?;
        let compute_capability = context.compute_capability()?;
        let limits = context.launch_limits()?;
        let (free_memory, total_memory) = memory_info(&context)?;
        let mut warp_size = 0i32;
        context.bind_to_thread()?;
        unsafe {
            cuda_core::sys::cuDeviceGetAttribute(
                &mut warp_size,
                cuda_core::sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                context.cu_device(),
            )
        }
        .result()?;
        Ok(CudaDevice {
            info: CudaDeviceInfo {
                ordinal,
                name,
                compute_capability,
                total_memory,
                free_memory,
                max_threads_per_block: limits.max_threads_per_block(),
                max_block_dim: limits.max_block_dim(),
                max_grid_dim: limits.max_grid_dim(),
                shared_memory_per_block: limits.max_shared_memory_per_block(),
                warp_size: warp_size.max(0) as u32,
            },
            context,
        })
    }

    pub fn info(&self) -> &CudaDeviceInfo {
        &self.info
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.context.default_stream()
    }

    pub fn new_stream(&self) -> Result<Arc<CudaStream>> {
        Ok(self.context.new_stream()?)
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.context.synchronize()?)
    }

    pub fn refresh_free_memory(&self) -> Result<usize> {
        Ok(memory_info(&self.context)?.0)
    }
}

fn memory_info(context: &Arc<CudaContext>) -> Result<(usize, usize)> {
    context.bind_to_thread()?;
    let mut free = 0usize;
    let mut total = 0usize;
    unsafe {
        cuda_core::sys::cuMemGetInfo_v2(&mut free, &mut total).result()?;
    }
    Ok((free, total))
}

pub fn device_count() -> Result<i32> {
    Ok(CudaContext::device_count()?)
}

pub fn list_devices() -> Result<Vec<CudaDeviceInfo>> {
    let count = device_count()?;
    let mut devices = Vec::with_capacity(count.max(0) as usize);
    for ordinal in 0..count.max(0) as usize {
        let device = CudaDevice::new(ordinal)?;
        devices.push(device.info().clone());
    }
    Ok(devices)
}
