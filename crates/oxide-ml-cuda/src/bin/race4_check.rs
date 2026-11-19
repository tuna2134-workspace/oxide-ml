use oxide_ml_cuda::{CudaBufferPool, CudaDevice, CudaKernels, CudaTensor};

const N: usize = 128;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let pool = CudaBufferPool::new(device.context().clone(), device.default_stream());
    let kernels = CudaKernels::load(device.context())?;
    let streams = (0..4)
        .map(|_| device.new_stream())
        .collect::<Result<Vec<_>, _>>()?;
    let mut shared = CudaTensor::from_host(&pool, vec![N], &vec![1.0f32; N])?;
    let source = CudaTensor::from_host(&pool, vec![N], &vec![2.0f32; N])?;
    let mut outputs = Vec::new();
    for stream in &streams {
        let mut out = CudaTensor::zeros(&pool, vec![N])?;
        for _ in 0..8 {
            kernels.broadcast(stream, &source, &mut out)?;
        }
        kernels.broadcast(stream, &shared, &mut out)?;
        outputs.push(out);
    }
    let writer = device.new_stream()?;
    kernels.scale(&writer, &mut shared, 2.0)?;
    let stats = shared.access_stats();
    for stream in streams.iter().chain(std::iter::once(&writer)) {
        stream.synchronize()?;
    }
    for (index, output) in outputs.iter().enumerate() {
        let host = output.to_host(&streams[index])?;
        assert_eq!(host, vec![1.0f32; N]);
    }
    assert_eq!(shared.to_host(&writer)?, vec![2.0f32; N]);
    println!("race4_check: PASS ({stats:?})");
    Ok(())
}
