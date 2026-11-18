use std::time::Instant;

use oxide_ml_cuda::{CudaBufferPool, CudaDevice, CudaKernels, CudaTensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = CudaDevice::new(0)?;
    let stream = device.default_stream();
    let pool = CudaBufferPool::new(device.context().clone(), stream.clone());
    let kernels = CudaKernels::load(device.context())?;
    for size in [64usize, 128, 256, 512] {
        let a_host = vec![0.5f32; size * size];
        let b_host = vec![0.25f32; size * size];
        let a = CudaTensor::from_host(&pool, vec![size, size], &a_host)?;
        let b = CudaTensor::from_host(&pool, vec![size, size], &b_host)?;
        let mut c = CudaTensor::zeros(&pool, vec![size, size])?;
        let cpu = cpu_gemm(&a_host, &b_host, size);
        let cpu_us = time(5, || {
            let _ = cpu_gemm(&a_host, &b_host, size);
        });
        let mut outputs = Vec::new();
        for (name, tiled) in [("naive", false), ("tiled", true)] {
            let us = time(20, || {
                if tiled {
                    kernels
                        .gemm_tiled(&stream, &a, &b, &mut c, size, size, size)
                        .unwrap();
                } else {
                    kernels
                        .gemm_naive(
                            &stream, &a, &b, &mut c, size, size, size, size, size, false, false,
                        )
                        .unwrap();
                }
                stream.synchronize().unwrap();
            });
            let host = c.to_host(&stream)?;
            let max_error = host
                .iter()
                .zip(&cpu)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            outputs.push((name, us, max_error));
        }
        println!("M=N=K={size}");
        println!("  cpu: {cpu_us:.1} us");
        for (name, us, error) in outputs {
            println!("  cuda {name}: {us:.1} us max_error={error:.5}");
        }
    }
    let stats = pool.stats();
    println!("pool: {stats:?}");
    Ok(())
}

fn time(mut iterations: usize, mut f: impl FnMut()) -> f64 {
    iterations = iterations.max(1);
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed().as_secs_f64() * 1e6 / iterations as f64
}

fn cpu_gemm(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for row in 0..n {
        for k in 0..n {
            let av = a[row * n + k];
            for col in 0..n {
                c[row * n + col] += av * b[k * n + col];
            }
        }
    }
    c
}
