use std::time::Instant;

use oxide_ml_cuda::{CudaDevice, CudaEvent, CudaKernels, CudaTensor};

const N: usize = 1024;
const REPEATS: usize = 3;
const STREAMS: usize = 4;

fn delay(
    kernels: &CudaKernels,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    pool: &oxide_ml_cuda::CudaBufferPool,
) -> Result<Vec<CudaTensor>, Box<dyn std::error::Error>> {
    let a = CudaTensor::from_host(pool, vec![N, N], &vec![0.5f32; N * N])?;
    let b = CudaTensor::from_host(pool, vec![N, N], &vec![0.25f32; N * N])?;
    let mut c = CudaTensor::zeros(pool, vec![N, N])?;
    for _ in 0..REPEATS {
        kernels.gemm_naive(stream, &a, &b, &mut c, N, N, N, N, N, false, false)?;
    }
    Ok(vec![a, b, c])
}

fn run_stream(
    kernels: &CudaKernels,
    stream: &std::sync::Arc<cuda_core::CudaStream>,
    pool: &oxide_ml_cuda::CudaBufferPool,
) -> Result<Vec<CudaTensor>, Box<dyn std::error::Error>> {
    let a = CudaTensor::from_host(pool, vec![N, N], &vec![0.5f32; N * N])?;
    let b = CudaTensor::from_host(pool, vec![N, N], &vec![0.25f32; N * N])?;
    let mut c = CudaTensor::zeros(pool, vec![N, N])?;
    for _ in 0..REPEATS {
        kernels.gemm_naive(stream, &a, &b, &mut c, N, N, N, N, N, false, false)?;
    }
    Ok(vec![a, b, c])
}

#[allow(clippy::type_complexity)]
fn measure(
    device: &CudaDevice,
    pool: &oxide_ml_cuda::CudaBufferPool,
    kernels: &CudaKernels,
    streams: &[std::sync::Arc<cuda_core::CudaStream>],
    serial: bool,
) -> Result<(f32, f32, Vec<(f32, f32)>), Box<dyn std::error::Error>> {
    let starts = (0..STREAMS)
        .map(|_| CudaEvent::new(device.context()))
        .collect::<Result<Vec<_>, _>>()?;
    let ends = (0..STREAMS)
        .map(|_| CudaEvent::new(device.context()))
        .collect::<Result<Vec<_>, _>>()?;
    for stream in streams {
        starts[stream_index(streams, stream)].record(stream)?;
    }
    let host_start = Instant::now();
    let mut keep = Vec::new();
    for index in 0..STREAMS {
        if serial && index > 0 {
            ends[index - 1].wait_on(&streams[index])?;
        }
        keep.push(run_stream(kernels, &streams[index], pool)?);
        ends[index].record(&streams[index])?;
    }
    let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;
    let origin = &starts[0];
    let mut intervals = Vec::new();
    for index in 0..STREAMS {
        let start = origin.elapsed_ms(&starts[index])?;
        let end = origin.elapsed_ms(&ends[index])?;
        intervals.push((start, end));
    }
    let span = origin.elapsed_ms(&ends[STREAMS - 1])?;
    let _ = keep;
    Ok((span, host_ms as f32, intervals))
}

fn stream_index(
    streams: &[std::sync::Arc<cuda_core::CudaStream>],
    stream: &std::sync::Arc<cuda_core::CudaStream>,
) -> usize {
    streams
        .iter()
        .position(|candidate| std::sync::Arc::ptr_eq(candidate, stream))
        .expect("stream belongs to set")
}

fn total_overlap(intervals: &[(f32, f32)]) -> f32 {
    let mut events = Vec::new();
    for &(start, end) in intervals {
        events.push((start, 1i32));
        events.push((end, -1i32));
    }
    events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let mut active = 0;
    let mut previous = 0.0;
    let mut overlap = 0.0;
    for (time, delta) in events {
        if active >= 2 {
            overlap += time - previous;
        }
        active += delta;
        previous = time;
    }
    overlap
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let pool =
        oxide_ml_cuda::CudaBufferPool::new(device.context().clone(), device.default_stream());
    let kernels = CudaKernels::load(device.context())?;
    let streams = (0..STREAMS)
        .map(|_| device.new_stream())
        .collect::<Result<Vec<_>, _>>()?;

    let (serial, serial_host, serial_intervals) =
        measure(&device, &pool, &kernels, &streams, true)?;
    let (parallel, parallel_host, parallel_intervals) =
        measure(&device, &pool, &kernels, &streams, false)?;
    for stream in &streams {
        stream.synchronize()?;
    }
    let overlap = total_overlap(&parallel_intervals);
    println!(
        "concurrency4_check: serial_span_ms={serial:.3} parallel_span_ms={parallel:.3} \
         serial_host_ms={serial_host:.3} parallel_host_ms={parallel_host:.3} \
         total_overlap_ms={overlap:.3} speedup={:.3}",
        serial / parallel
    );
    println!("serial_intervals={serial_intervals:?}");
    println!("parallel_intervals={parallel_intervals:?}");

    // Shared Read: four readers, then one writer. All readers must be pending.
    let s1 = device.new_stream()?;
    let s2 = device.new_stream()?;
    let s3 = device.new_stream()?;
    let s4 = device.new_stream()?;
    let writer_stream = device.new_stream()?;
    let mut shared = CudaTensor::from_host(&pool, vec![4], &[1.0, 2.0, 3.0, 4.0])?;
    let mut outputs = Vec::new();
    let mut delays = Vec::new();
    for stream in [&s1, &s2, &s3, &s4] {
        delays.push(delay(&kernels, stream, &pool)?);
        let mut out = CudaTensor::zeros(&pool, vec![4])?;
        kernels.broadcast(stream, &shared, &mut out)?;
        outputs.push(out);
    }
    kernels.scale(&writer_stream, &mut shared, 2.0)?;
    let stats = shared.access_stats();
    assert_eq!(stats.cross_stream_waits, 4, "{stats:?}");
    assert_eq!(stats.read_read_concurrent, 3, "{stats:?}");
    for stream in [&s1, &s2, &s3, &s4, &writer_stream] {
        stream.synchronize()?;
    }
    println!("concurrency4_check: shared read/write PASS ({stats:?})");
    if overlap <= 0.0 {
        println!("concurrency4_check: WARN no GPU overlap");
    } else {
        println!("concurrency4_check: PASS actual GPU overlap");
    }
    Ok(())
}
