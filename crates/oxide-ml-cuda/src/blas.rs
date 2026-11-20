//! Minimal cuBLAS bindings used by the convolution path.
//!
//! Conv2d is lowered to im2col + SGEMM. The reference tiled GEMM is
//! correctness-oriented; cuBLAS gives the convolution a competitive GEMM
//! backend on real GPUs.

use std::ffi::c_void;
use std::sync::Arc;

use cuda_core::CudaStream;

use crate::error::{CudaError, Result};

#[repr(C)]
struct CublasContext {
    _private: [u8; 0],
}

type Handle = *mut CublasContext;

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut Handle) -> i32;
    fn cublasDestroy_v2(handle: Handle) -> i32;
    fn cublasSetStream_v2(handle: Handle, stream: *mut c_void) -> i32;
    fn cublasSgemm_v2(
        handle: Handle,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> i32;
}

const OP_N: i32 = 0;

pub struct Cublas {
    handle: Handle,
}

// The handle is only used from the owning executor; cuBLAS handles are bound
// to a stream per call. Marking this Send is required because executors may move
// between threads (the handle itself is not shared concurrently).
unsafe impl Send for Cublas {}
unsafe impl Sync for Cublas {}

impl Cublas {
    pub fn new() -> Result<Self> {
        let mut handle: Handle = std::ptr::null_mut();
        let status = unsafe { cublasCreate_v2(&mut handle) };
        if status != 0 || handle.is_null() {
            return Err(CudaError::Unsupported(format!(
                "cublasCreate_v2 failed with status {status}"
            )));
        }
        Ok(Cublas { handle })
    }

    pub fn set_stream(&self, stream: &Arc<CudaStream>) -> Result<()> {
        let status = unsafe { cublasSetStream_v2(self.handle, stream.cu_stream() as *mut c_void) };
        if status != 0 {
            return Err(CudaError::Unsupported(format!(
                "cublasSetStream_v2 failed with status {status}"
            )));
        }
        Ok(())
    }

    /// Row-major `C[m,n] = alpha * A[m,k] * B[k,n] + beta * C`.
    ///
    /// cuBLAS is column-major, so this computes the transposed product with
    /// row-major buffers reinterpreted as their column-major transposes.
    #[allow(clippy::too_many_arguments)]
    pub fn sgemm(
        &self,
        stream: &Arc<CudaStream>,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: u64,
        b: u64,
        beta: f32,
        c: u64,
    ) -> Result<()> {
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let (m, n, k) = (m as i32, n as i32, k as i32);
        self.set_stream(stream)?;
        let status = unsafe {
            cublasSgemm_v2(
                self.handle,
                OP_N,
                OP_N,
                n,
                m,
                k,
                &alpha,
                b as *const f32,
                n,
                a as *const f32,
                k,
                &beta,
                c as *mut f32,
                n,
            )
        };
        if status != 0 {
            return Err(CudaError::Unsupported(format!(
                "cublasSgemm_v2 failed with status {status}"
            )));
        }
        Ok(())
    }
}

impl Drop for Cublas {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                cublasDestroy_v2(self.handle);
            }
        }
    }
}
