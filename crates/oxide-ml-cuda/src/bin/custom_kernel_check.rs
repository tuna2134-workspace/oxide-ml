//! Run the custom cuda-oxide kernel graph check.

fn main() -> oxide_ml_cuda::Result<()> {
    let device = oxide_ml_cuda::CudaDevice::new(0)?;
    let difference = oxide_ml_cuda::custom_check::run(device)?;
    println!("custom kernel max |CPU - CUDA| = {difference:.3e}");
    if difference < 1e-5 {
        println!("CUSTOM_KERNEL_CHECK: PASS");
        Ok(())
    } else {
        println!("CUSTOM_KERNEL_CHECK: FAIL");
        Err(oxide_ml_cuda::CudaError::Execution(
            "custom kernel numerical mismatch".into(),
        ))
    }
}
