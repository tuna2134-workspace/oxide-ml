use std::time::Instant;

use oxide_ml_cuda::{CudaDevice, CudaEvent, CudaKernels, CudaTensor};

const N: usize = 1024;
const REPEATS: usize = 4;

fn make_set(
    pool: &oxide_ml_cuda::CudaBufferPool,
) -> Result<(CudaTensor, CudaTensor, CudaTensor), Box<dyn std::error::Error>> {
    let a = CudaTensor::from_host(pool, vec![N, N], &vec![0.5f32; N * N])?;
    let b = CudaTensor::from_host(pool, vec![N, N], &vec![0.25f32; N * N])?;
    let c = CudaTensor::zeros(pool, vec![N, N])?;
    Ok((a, b, c))
}

fn run_gemms(
    kernels: &CudaKernels,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    a: &CudaTensor,
    b: &CudaTensor,
    mut c: CudaTensor,
) -> Result<CudaTensor, Box<dyn std::error::Error>> {
    for _ in 0..REPEATS {
        kernels.gemm_naive(stream, a, b, &mut c, N, N, N, N, N, false, false)?;
    }
    Ok(c)
}

#[allow(clippy::type_complexity)]
fn measure_pair(
    device: &CudaDevice,
    pool: &oxide_ml_cuda::CudaBufferPool,
    kernels: &CudaKernels,
    a_stream: &std::sync::Arc<cuda_core::CudaStream>,
    b_stream: &std::sync::Arc<cuda_core::CudaStream>,
    serial: bool,
) -> Result<(f32, f32, f32, f32, f32), Box<dyn std::error::Error>> {
    let (a1, b1, c1) = make_set(pool)?;
    let (a2, b2, c2) = make_set(pool)?;
    let start_a = CudaEvent::new(device.context())?;
    let start_b = CudaEvent::new(device.context())?;
    let end_a = CudaEvent::new(device.context())?;
    let end_b = CudaEvent::new(device.context())?;

    start_a.record(a_stream)?;
    start_b.record(b_stream)?;
    let host_submit = Instant::now();
    let c1 = run_gemms(kernels, a_stream, &a1, &b1, c1)?;
    end_a.record(a_stream)?;
    if serial {
        end_a.wait_on(b_stream)?;
    }
    let c2 = run_gemms(kernels, b_stream, &a2, &b2, c2)?;
    end_b.record(b_stream)?;
    let host_submit_ms = host_submit.elapsed().as_secs_f64() * 1000.0;
    let _ = (&c1, &c2);

    let a_start = 0.0;
    let b_start = start_a.elapsed_ms(&start_b)?;
    let a_end = start_a.elapsed_ms(&end_a)?;
    let b_end = start_a.elapsed_ms(&end_b)?;
    let span = start_a.elapsed_ms(&end_b)?;
    let overlap = f32::min(a_end, b_end) - f32::max(a_start, b_start);
    Ok((
        span,
        host_submit_ms as f32,
        a_end - a_start,
        b_end - b_start,
        overlap,
    ))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let kernels = CudaKernels::load(device.context())?;
    let a_stream = device.new_stream()?;
    let b_stream = device.new_stream()?;

    let pool =
        oxide_ml_cuda::CudaBufferPool::new(device.context().clone(), device.default_stream());
    let (serial_span, serial_submit, serial_a, serial_b, _) =
        measure_pair(&device, &pool, &kernels, &a_stream, &b_stream, true)?;
    a_stream.synchronize()?;
    b_stream.synchronize()?;
    let (parallel_span, parallel_submit, parallel_a, parallel_b, overlap) =
        measure_pair(&device, &pool, &kernels, &a_stream, &b_stream, false)?;
    a_stream.synchronize()?;
    b_stream.synchronize()?;

    println!(
        "concurrency_check: serial_span_ms={serial_span:.3} parallel_span_ms={parallel_span:.3} \
         serial_submit_ms={serial_submit:.3} parallel_submit_ms={parallel_submit:.3} \
         a_ms={parallel_a:.3} b_ms={parallel_b:.3} overlap_ms={overlap:.3}"
    );
    let speedup = serial_span / parallel_span;
    println!("concurrency_check: speedup={speedup:.3}");
    assert!(serial_a > 0.0 && serial_b > 0.0);
    if overlap <= 0.0 {
        println!("concurrency_check: WARN no GPU overlap observed (resource serialization)");
    } else {
        println!("concurrency_check: PASS actual GPU overlap");
    }
    assert!(parallel_span > 0.0);
    Ok(())
}
