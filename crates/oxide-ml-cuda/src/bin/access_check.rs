use oxide_ml_cuda::{CudaDevice, CudaKernels, CudaTensor};

fn delay(
    kernels: &CudaKernels,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    pool: &oxide_ml_cuda::CudaBufferPool,
) -> Result<Vec<CudaTensor>, Box<dyn std::error::Error>> {
    let a = CudaTensor::from_host(pool, vec![512, 512], &vec![0.5f32; 512 * 512])?;
    let b = CudaTensor::from_host(pool, vec![512, 512], &vec![0.25f32; 512 * 512])?;
    let mut c = CudaTensor::zeros(pool, vec![512, 512])?;
    for _ in 0..8 {
        kernels.gemm_naive(
            stream, &a, &b, &mut c, 512, 512, 512, 512, 512, false, false,
        )?;
    }
    Ok(vec![a, b, c])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let pool =
        oxide_ml_cuda::CudaBufferPool::new(device.context().clone(), device.default_stream());
    let kernels = CudaKernels::load(device.context())?;
    let s1 = device.new_stream()?;
    let s2 = device.new_stream()?;
    let s3 = device.new_stream()?;
    let s4 = device.new_stream()?;
    let source = CudaTensor::from_host(&pool, vec![4], &[1.0, 2.0, 3.0, 4.0])?;
    let mut read_a = CudaTensor::zeros(&pool, vec![4])?;
    let mut read_b = CudaTensor::zeros(&pool, vec![4])?;
    let mut shared = CudaTensor::from_host(&pool, vec![4], &[1.0, 2.0, 3.0, 4.0])?;

    let _d1 = delay(&kernels, &s1, &pool)?;
    let _d2 = delay(&kernels, &s2, &pool)?;
    kernels.broadcast(&s1, &shared, &mut read_a)?;
    kernels.broadcast(&s2, &shared, &mut read_b)?;
    kernels.scale(&s3, &mut shared, 2.0)?;
    let stats = shared.access_stats();
    assert_eq!(stats.event_waits, 2, "{stats:?}");
    assert_eq!(stats.cross_stream_waits, 2, "{stats:?}");
    assert_eq!(stats.read_read_concurrent, 1, "{stats:?}");
    s1.synchronize()?;
    s2.synchronize()?;
    s3.synchronize()?;
    assert_eq!(read_a.to_host(&s1)?, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(read_b.to_host(&s2)?, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(shared.to_host(&s3)?, vec![2.0, 4.0, 6.0, 8.0]);

    let mut chain = CudaTensor::zeros(&pool, vec![4])?;
    let mut out_b = CudaTensor::zeros(&pool, vec![4])?;
    let mut out_c = CudaTensor::zeros(&pool, vec![4])?;
    let _d3 = delay(&kernels, &s1, &pool)?;
    let _d4 = delay(&kernels, &s2, &pool)?;
    let _d5 = delay(&kernels, &s3, &pool)?;
    kernels.broadcast(&s1, &source, &mut chain)?;
    kernels.broadcast(&s2, &chain, &mut out_b)?;
    kernels.broadcast(&s3, &chain, &mut out_c)?;
    kernels.scale(&s4, &mut chain, 3.0)?;
    let chain_stats = chain.access_stats();
    assert_eq!(chain_stats.event_waits, 3, "{chain_stats:?}");
    s1.synchronize()?;
    s2.synchronize()?;
    s3.synchronize()?;
    s4.synchronize()?;
    assert_eq!(chain.to_host(&s4)?, vec![3.0, 6.0, 9.0, 12.0]);

    let alias = shared.reshape(vec![4])?;
    assert_eq!(alias.identity(), shared.identity());
    let mut out_alias = CudaTensor::zeros(&pool, vec![4])?;
    let _d6 = delay(&kernels, &s1, &pool)?;
    kernels.broadcast(&s1, &alias, &mut out_alias)?;
    assert!(shared.access_stats().pending_readers >= 1);
    s1.synchronize()?;
    println!("access_check: PASS ({stats:?} / {chain_stats:?})");
    Ok(())
}
