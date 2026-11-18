use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, LaunchConfig};
use cuda_host::cuda_module;

use crate::buffer::CudaTensor;
use crate::error::Result;

#[cuda_module]
mod kernels {
    use cuda_device::float::{ex2_approx_f32, sqrt_rn_f32, tanh_approx_f32};
    use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, thread};

    #[inline(always)]
    fn exp_f32(value: f32) -> f32 {
        ex2_approx_f32(value * std::f32::consts::LOG2_E)
    }

    #[inline(always)]
    fn sigmoid_f32(value: f32) -> f32 {
        0.5 * (tanh_approx_f32(value * 0.5) + 1.0)
    }

    #[inline(always)]
    fn gelu_f32(value: f32) -> f32 {
        let c = 0.797_884_6_f32;
        0.5 * value * (tanh_approx_f32(c * (value + 0.044_715 * value * value * value)) + 1.0)
    }

    #[inline(always)]
    fn broadcast_offset(
        c0: u32,
        c1: u32,
        c2: u32,
        c3: u32,
        s0: u32,
        s1: u32,
        s2: u32,
        s3: u32,
    ) -> u32 {
        let coords = [c0, c1, c2, c3];
        let shape = [s0, s1, s2, s3];
        let mut offset = 0u32;
        let mut stride = 1u32;
        let mut axis = 4usize;
        while axis > 0 {
            axis -= 1;
            let coordinate = if shape[axis] == 1 { 0 } else { coords[axis] };
            offset += coordinate * stride;
            stride *= shape[axis];
        }
        offset
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn unary(op: u32, input: &[f32], mut output: DisjointSlice<f32>, n: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        if let Some(slot) = output.get_mut(idx) {
            let value = input[i];
            *slot = match op {
                0 => {
                    if value > 0.0 {
                        value
                    } else {
                        0.0
                    }
                }
                1 => gelu_f32(value),
                2 => sigmoid_f32(value),
                3 => tanh_approx_f32(value),
                4 => -value,
                _ => value,
            };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn binary(
        op: u32,
        a: &[f32],
        b: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        _o0: u32,
        o1: u32,
        o2: u32,
        o3: u32,
        a0: u32,
        a1: u32,
        a2: u32,
        a3: u32,
        b0: u32,
        b1: u32,
        b2: u32,
        b3: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        let c0 = i as u32 / (o1 * o2 * o3);
        let r0 = i as u32 % (o1 * o2 * o3);
        let c1 = r0 / (o2 * o3);
        let r1 = r0 % (o2 * o3);
        let c2 = r1 / o3;
        let c3 = r1 % o3;
        let av = a[broadcast_offset(c0, c1, c2, c3, a0, a1, a2, a3) as usize];
        let bv = b[broadcast_offset(c0, c1, c2, c3, b0, b1, b2, b3) as usize];
        if let Some(slot) = output.get_mut(idx) {
            *slot = match op {
                0 => av + bv,
                1 => av - bv,
                2 => av * bv,
                3 => av / bv,
                4 => {
                    let sum = av + bv;
                    if sum > 0.0 { sum } else { 0.0 }
                }
                _ => av + bv,
            };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn add_relu_bwd(
        a: &[f32],
        b: &[f32],
        dy: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        _o0: u32,
        o1: u32,
        o2: u32,
        o3: u32,
        a0: u32,
        a1: u32,
        a2: u32,
        a3: u32,
        b0: u32,
        b1: u32,
        b2: u32,
        b3: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        let c0 = i as u32 / (o1 * o2 * o3);
        let r0 = i as u32 % (o1 * o2 * o3);
        let c1 = r0 / (o2 * o3);
        let r1 = r0 % (o2 * o3);
        let c2 = r1 / o3;
        let c3 = r1 % o3;
        let av = a[broadcast_offset(c0, c1, c2, c3, a0, a1, a2, a3) as usize];
        let bv = b[broadcast_offset(c0, c1, c2, c3, b0, b1, b2, b3) as usize];
        if let Some(slot) = output.get_mut(idx) {
            *slot = if av + bv > 0.0 { dy[i] } else { 0.0 };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn activation_bwd(op: u32, x: &[f32], dy: &[f32], mut output: DisjointSlice<f32>, n: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        let value = x[i];
        let derivative = match op {
            0 => {
                if value > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            1 => {
                let c = 0.797_884_6_f32;
                let inner = c * (value + 0.044_715 * value * value * value);
                let t = tanh_approx_f32(inner);
                0.5 * (1.0 + t)
                    + 0.5 * value * (1.0 - t * t) * c * (1.0 + 3.0 * 0.044_715 * value * value)
            }
            2 => {
                let s = sigmoid_f32(value);
                s * (1.0 - s)
            }
            3 => {
                let t = tanh_approx_f32(value);
                1.0 - t * t
            }
            _ => 1.0,
        };
        if let Some(slot) = output.get_mut(idx) {
            *slot = dy[i] * derivative;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn transpose(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        _i0: u32,
        i1: u32,
        i2: u32,
        i3: u32,
        _o0: u32,
        o1: u32,
        o2: u32,
        o3: u32,
        p0: u32,
        p1: u32,
        p2: u32,
        p3: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        let oc = [
            i as u32 / (o1 * o2 * o3),
            (i as u32 % (o1 * o2 * o3)) / (o2 * o3),
            (i as u32 % (o2 * o3)) / o3,
            i as u32 % o3,
        ];
        let perm = [p0, p1, p2, p3];
        let mut ic = [0u32, 0, 0, 0];
        for axis in 0..4 {
            ic[perm[axis] as usize] = oc[axis];
        }
        let offset = ((ic[0] * i1 + ic[1]) * i2 + ic[2]) * i3 + ic[3];
        if let Some(slot) = output.get_mut(idx) {
            *slot = input[offset as usize];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn broadcast(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        _o0: u32,
        o1: u32,
        o2: u32,
        o3: u32,
        i0: u32,
        i1: u32,
        i2: u32,
        i3: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        let c0 = i as u32 / (o1 * o2 * o3);
        let r0 = i as u32 % (o1 * o2 * o3);
        let c1 = r0 / (o2 * o3);
        let r1 = r0 % (o2 * o3);
        let c2 = r1 / o3;
        let c3 = r1 % o3;
        let offset = broadcast_offset(c0, c1, c2, c3, i0, i1, i2, i3);
        if let Some(slot) = output.get_mut(idx) {
            *slot = input[offset as usize];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn reduce(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        n_out: u32,
        i0: u32,
        i1: u32,
        i2: u32,
        i3: u32,
        _o0: u32,
        o1: u32,
        o2: u32,
        o3: u32,
        t0: u32,
        t1: u32,
        t2: u32,
        t3: u32,
        mask: u32,
        mean: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n_out {
            return;
        }
        let oc = [
            i as u32 / (o1 * o2 * o3),
            (i as u32 % (o1 * o2 * o3)) / (o2 * o3),
            (i as u32 % (o2 * o3)) / o3,
            i as u32 % o3,
        ];
        let map = [t0, t1, t2, t3];
        let mut ic = [0u32, 0, 0, 0];
        for axis in 0..4 {
            let target = map[axis];
            if target != u32::MAX {
                ic[target as usize] = oc[axis];
            }
        }
        let shape = [i0, i1, i2, i3];
        let mut start = [0u32, 0, 0, 0];
        let mut end = [1u32, 1, 1, 1];
        let mut count = 1u32;
        for axis in 0..4 {
            if mask & (1 << axis) != 0 {
                end[axis] = shape[axis];
                count *= shape[axis];
            } else {
                start[axis] = ic[axis];
                end[axis] = ic[axis] + 1;
            }
        }
        let mut sum = 0.0f32;
        let mut a = start[0];
        while a < end[0] {
            let mut b = start[1];
            while b < end[1] {
                let mut c = start[2];
                while c < end[2] {
                    let mut d = start[3];
                    while d < end[3] {
                        let offset = ((a * i1 + b) * i2 + c) * i3 + d;
                        sum += input[offset as usize];
                        d += 1;
                    }
                    c += 1;
                }
                b += 1;
            }
            a += 1;
        }
        if let Some(slot) = output.get_mut(idx) {
            *slot = if mean != 0 { sum / count as f32 } else { sum };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn softmax(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        axis_len: u32,
        inner: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= n {
            return;
        }
        let row = i / (axis_len * inner);
        let inner_index = i % inner;
        let base = row * axis_len * inner + inner_index;
        let mut max_value = -3.402_823_5e38f32;
        let mut a = 0u32;
        while a < axis_len {
            let value = input[(base + a * inner) as usize];
            if value > max_value {
                max_value = value;
            }
            a += 1;
        }
        let mut sum = 0.0f32;
        a = 0;
        while a < axis_len {
            sum += exp_f32(input[(base + a * inner) as usize] - max_value);
            a += 1;
        }
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = exp_f32(input[i as usize] - max_value) / sum;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn layer_norm(
        input: &[f32],
        gamma: &[f32],
        beta: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        norm: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= n {
            return;
        }
        let group = i / norm;
        let start = group * norm;
        let mut mean = 0.0f32;
        let mut a = 0u32;
        while a < norm {
            mean += input[(start + a) as usize];
            a += 1;
        }
        mean /= norm as f32;
        let mut variance = 0.0f32;
        a = 0;
        while a < norm {
            let d = input[(start + a) as usize] - mean;
            variance += d * d;
            a += 1;
        }
        variance /= norm as f32;
        let inv = 1.0 / sqrt_rn_f32(variance + 1.0e-5);
        if let Some(slot) = output.get_mut(thread::index_1d()) {
            *slot = (input[i as usize] - mean) * inv * gamma[(i % norm) as usize]
                + beta[(i % norm) as usize];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn mse(
        prediction: &[f32],
        target: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        mean: u32,
    ) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }
        let mut sum = 0.0f32;
        let mut i = 0u32;
        while i < n {
            let d = prediction[i as usize] - target[i as usize];
            sum += d * d;
            i += 1;
        }
        if let Some(slot) = output.get_mut(idx) {
            *slot = if mean != 0 { sum / n as f32 } else { sum };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn mse_bwd(
        prediction: &[f32],
        target: &[f32],
        dy: &[f32],
        mut dpred: DisjointSlice<f32>,
        mut dtarget: DisjointSlice<f32>,
        n: u32,
        mean: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        let scale = dy[0] * if mean != 0 { 2.0 / n as f32 } else { 2.0 };
        let d = prediction[i] - target[i];
        if let Some(slot) = dpred.get_mut(thread::index_1d()) {
            *slot = d * scale;
        }
        if let Some(slot) = dtarget.get_mut(thread::index_1d()) {
            *slot = -d * scale;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn scale(mut tensor: DisjointSlice<f32>, factor: f32, n: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        if let Some(slot) = tensor.get_mut(thread::index_1d()) {
            *slot *= factor;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn gemm_naive(
        a: &[f32],
        b: &[f32],
        mut c: DisjointSlice<f32>,
        m: u32,
        n: u32,
        k: u32,
        lda: u32,
        ldb: u32,
        trans_a: u32,
        trans_b: u32,
    ) {
        let idx = thread::index_1d();
        let linear = idx.get() as u32;
        if linear >= m * n {
            return;
        }
        let row = linear / n;
        let col = linear % n;
        let mut acc = 0.0f32;
        let mut inner = 0u32;
        while inner < k {
            let av = if trans_a != 0 {
                a[(inner * lda + row) as usize]
            } else {
                a[(row * lda + inner) as usize]
            };
            let bv = if trans_b != 0 {
                b[(col * ldb + inner) as usize]
            } else {
                b[(inner * ldb + col) as usize]
            };
            acc += av * bv;
            inner += 1;
        }
        if let Some(slot) = c.get_mut(thread::index_1d()) {
            *slot = acc;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn gemm_tiled(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>, m: u32, n: u32, k: u32) {
        static mut AS: SharedArray<f32, 256> = SharedArray::UNINIT;
        static mut BS: SharedArray<f32, 256> = SharedArray::UNINIT;
        let idx = thread::index_1d();
        let global = idx.get() as u32;
        let tile = global / 256;
        let tid = global % 256;
        let tx = tid % 16;
        let ty = tid / 16;
        let tiles_n = n.div_ceil(16);
        let tile_row = tile / tiles_n;
        let tile_col = tile % tiles_n;
        let row = tile_row * 16 + ty;
        let col = tile_col * 16 + tx;
        let mut acc = 0.0f32;
        let tiles_k = k.div_ceil(16);
        let mut t = 0u32;
        while t < tiles_k {
            let ar = row;
            let ac = t * 16 + tx;
            let av = if ar < m && ac < k {
                a[(ar * k + ac) as usize]
            } else {
                0.0
            };
            let br = t * 16 + ty;
            let bc = col;
            let bv = if br < k && bc < n {
                b[(br * n + bc) as usize]
            } else {
                0.0
            };
            unsafe {
                AS[(ty * 16 + tx) as usize] = av;
                BS[(ty * 16 + tx) as usize] = bv;
            }
            thread::sync_threads();
            let mut kk = 0u32;
            while kk < 16 {
                unsafe {
                    acc += AS[(ty * 16 + kk) as usize] * BS[(kk * 16 + tx) as usize];
                }
                kk += 1;
            }
            thread::sync_threads();
            t += 1;
        }
        if row < m && col < n {
            unsafe {
                *c.get_unchecked_mut((row * n + col) as usize) = acc;
            }
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn sgd(mut param: DisjointSlice<f32>, grad: &[f32], lr: f32, n: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        if let Some(slot) = param.get_mut(thread::index_1d()) {
            *slot -= lr * grad[i];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn adamw(
        mut param: DisjointSlice<f32>,
        grad: &[f32],
        mut m: DisjointSlice<f32>,
        mut v: DisjointSlice<f32>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        n: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        let g = grad[i];
        let mut m_value = 0.0f32;
        let mut v_value = 0.0f32;
        if let Some(slot) = m.get_mut(thread::index_1d()) {
            *slot = beta1 * *slot + (1.0 - beta1) * g;
            m_value = *slot;
        }
        if let Some(slot) = v.get_mut(thread::index_1d()) {
            *slot = beta2 * *slot + (1.0 - beta2) * g * g;
            v_value = *slot;
        }
        if let Some(slot) = param.get_mut(thread::index_1d()) {
            let m_hat = m_value / bias_correction1;
            let v_hat = v_value / bias_correction2;
            *slot -= lr * (m_hat / (sqrt_rn_f32(v_hat) + eps) + weight_decay * *slot);
        }
    }
}

pub struct CudaKernels {
    module: kernels::LoadedModule,
}

impl CudaKernels {
    pub fn load(context: &Arc<CudaContext>) -> Result<Self> {
        let module = kernels::load(context)?;
        Ok(CudaKernels { module })
    }

    pub fn unary(
        &self,
        stream: &CudaStream,
        op: u32,
        input: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let n = input.numel() as u32;
        unsafe {
            self.module.unary(
                stream,
                LaunchConfig::for_num_elems(n),
                op,
                input.buffer()?,
                output.buffer_mut()?,
                n,
            )?;
        }
        Ok(())
    }

    pub fn binary(
        &self,
        stream: &CudaStream,
        op: u32,
        lhs: &CudaTensor,
        rhs: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let o = padded4(output.shape());
        let a = padded4(lhs.shape());
        let b = padded4(rhs.shape());
        let n = output.numel() as u32;
        unsafe {
            self.module.binary(
                stream,
                LaunchConfig::for_num_elems(n),
                op,
                lhs.buffer()?,
                rhs.buffer()?,
                output.buffer_mut()?,
                n,
                o[0],
                o[1],
                o[2],
                o[3],
                a[0],
                a[1],
                a[2],
                a[3],
                b[0],
                b[1],
                b[2],
                b[3],
            )?;
        }
        Ok(())
    }

    pub fn add_relu_backward(
        &self,
        stream: &CudaStream,
        lhs: &CudaTensor,
        rhs: &CudaTensor,
        dy: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let o = padded4(output.shape());
        let a = padded4(lhs.shape());
        let b = padded4(rhs.shape());
        let n = output.numel() as u32;
        unsafe {
            self.module.add_relu_bwd(
                stream,
                LaunchConfig::for_num_elems(n),
                lhs.buffer()?,
                rhs.buffer()?,
                dy.buffer()?,
                output.buffer_mut()?,
                n,
                o[0],
                o[1],
                o[2],
                o[3],
                a[0],
                a[1],
                a[2],
                a[3],
                b[0],
                b[1],
                b[2],
                b[3],
            )?;
        }
        Ok(())
    }

    pub fn activation_backward(
        &self,
        stream: &CudaStream,
        op: u32,
        x: &CudaTensor,
        dy: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let n = x.numel() as u32;
        unsafe {
            self.module.activation_bwd(
                stream,
                LaunchConfig::for_num_elems(n),
                op,
                x.buffer()?,
                dy.buffer()?,
                output.buffer_mut()?,
                n,
            )?;
        }
        Ok(())
    }

    pub fn transpose(
        &self,
        stream: &CudaStream,
        input: &CudaTensor,
        output: &mut CudaTensor,
        axes: &[usize],
    ) -> Result<()> {
        let i = padded4(input.shape());
        let o = padded4(output.shape());
        let p = permutation4(axes, input.shape().len());
        let n = output.numel() as u32;
        unsafe {
            self.module.transpose(
                stream,
                LaunchConfig::for_num_elems(n),
                input.buffer()?,
                output.buffer_mut()?,
                n,
                i[0],
                i[1],
                i[2],
                i[3],
                o[0],
                o[1],
                o[2],
                o[3],
                p[0],
                p[1],
                p[2],
                p[3],
            )?;
        }
        Ok(())
    }

    pub fn broadcast(
        &self,
        stream: &CudaStream,
        input: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let o = padded4(output.shape());
        let i = padded4(input.shape());
        let n = output.numel() as u32;
        unsafe {
            self.module.broadcast(
                stream,
                LaunchConfig::for_num_elems(n),
                input.buffer()?,
                output.buffer_mut()?,
                n,
                o[0],
                o[1],
                o[2],
                o[3],
                i[0],
                i[1],
                i[2],
                i[3],
            )?;
        }
        Ok(())
    }

    pub fn reduce(
        &self,
        stream: &CudaStream,
        input: &CudaTensor,
        output: &mut CudaTensor,
        axes: &[usize],
        mean: bool,
    ) -> Result<()> {
        let i = padded4(input.shape());
        let o = padded4(output.shape());
        let (to_in, mask) = reduction_mapping(input.shape().len(), output.shape().len(), axes);
        let n = output.numel() as u32;
        unsafe {
            self.module.reduce(
                stream,
                LaunchConfig::for_num_elems(n),
                input.buffer()?,
                output.buffer_mut()?,
                n,
                i[0],
                i[1],
                i[2],
                i[3],
                o[0],
                o[1],
                o[2],
                o[3],
                to_in[0],
                to_in[1],
                to_in[2],
                to_in[3],
                mask,
                u32::from(mean),
            )?;
        }
        Ok(())
    }

    pub fn softmax(
        &self,
        stream: &CudaStream,
        input: &CudaTensor,
        output: &mut CudaTensor,
        axis: usize,
    ) -> Result<()> {
        let dims = input.shape();
        let axis_len = dims[axis] as u32;
        let inner: u32 = dims[axis + 1..].iter().product::<usize>() as u32;
        let n = input.numel() as u32;
        unsafe {
            self.module.softmax(
                stream,
                LaunchConfig::for_num_elems(n),
                input.buffer()?,
                output.buffer_mut()?,
                n,
                axis_len,
                inner,
            )?;
        }
        Ok(())
    }

    pub fn layer_norm(
        &self,
        stream: &CudaStream,
        input: &CudaTensor,
        gamma: &CudaTensor,
        beta: &CudaTensor,
        output: &mut CudaTensor,
        norm: u32,
    ) -> Result<()> {
        let n = input.numel() as u32;
        unsafe {
            self.module.layer_norm(
                stream,
                LaunchConfig::for_num_elems(n),
                input.buffer()?,
                gamma.buffer()?,
                beta.buffer()?,
                output.buffer_mut()?,
                n,
                norm,
            )?;
        }
        Ok(())
    }

    pub fn mse(
        &self,
        stream: &CudaStream,
        prediction: &CudaTensor,
        target: &CudaTensor,
        output: &mut CudaTensor,
        mean: bool,
    ) -> Result<()> {
        let n = prediction.numel() as u32;
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.module.mse(
                stream,
                config,
                prediction.buffer()?,
                target.buffer()?,
                output.buffer_mut()?,
                n,
                u32::from(mean),
            )?;
        }
        Ok(())
    }

    pub fn mse_backward(
        &self,
        stream: &CudaStream,
        prediction: &CudaTensor,
        target: &CudaTensor,
        dy: &CudaTensor,
        dpred: &mut CudaTensor,
        dtarget: &mut CudaTensor,
        mean: bool,
    ) -> Result<()> {
        let n = prediction.numel() as u32;
        unsafe {
            self.module.mse_bwd(
                stream,
                LaunchConfig::for_num_elems(n),
                prediction.buffer()?,
                target.buffer()?,
                dy.buffer()?,
                dpred.buffer_mut()?,
                dtarget.buffer_mut()?,
                n,
                u32::from(mean),
            )?;
        }
        Ok(())
    }

    pub fn scale(&self, stream: &CudaStream, tensor: &mut CudaTensor, factor: f32) -> Result<()> {
        let n = tensor.numel() as u32;
        unsafe {
            self.module.scale(
                stream,
                LaunchConfig::for_num_elems(n),
                tensor.buffer_mut()?,
                factor,
                n,
            )?;
        }
        Ok(())
    }

    pub fn gemm_naive(
        &self,
        stream: &CudaStream,
        a: &CudaTensor,
        b: &CudaTensor,
        out: &mut CudaTensor,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Result<()> {
        let total = (m * n) as u32;
        unsafe {
            self.module.gemm_naive(
                stream,
                LaunchConfig::for_num_elems(total),
                a.buffer()?,
                b.buffer()?,
                out.buffer_mut()?,
                m as u32,
                n as u32,
                k as u32,
                lda as u32,
                ldb as u32,
                u32::from(trans_a),
                u32::from(trans_b),
            )?;
        }
        Ok(())
    }

    pub fn gemm_tiled(
        &self,
        stream: &CudaStream,
        a: &CudaTensor,
        b: &CudaTensor,
        out: &mut CudaTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let tiles = m.div_ceil(16) * n.div_ceil(16);
        let total = (tiles * 256) as u32;
        unsafe {
            self.module.gemm_tiled(
                stream,
                LaunchConfig::for_num_elems(total),
                a.buffer()?,
                b.buffer()?,
                out.buffer_mut()?,
                m as u32,
                n as u32,
                k as u32,
            )?;
        }
        Ok(())
    }
    pub fn sgd(
        &self,
        stream: &CudaStream,
        param: &mut CudaTensor,
        grad: &CudaTensor,
        lr: f32,
    ) -> Result<()> {
        let n = param.numel() as u32;
        unsafe {
            self.module.sgd(
                stream,
                LaunchConfig::for_num_elems(n),
                param.buffer_mut()?,
                grad.buffer()?,
                lr,
                n,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adamw(
        &self,
        stream: &CudaStream,
        param: &mut CudaTensor,
        grad: &CudaTensor,
        m: &mut CudaTensor,
        v: &mut CudaTensor,
        config: AdamWParams,
        bias_correction1: f32,
        bias_correction2: f32,
    ) -> Result<()> {
        let n = param.numel() as u32;
        unsafe {
            self.module.adamw(
                stream,
                LaunchConfig::for_num_elems(n),
                param.buffer_mut()?,
                grad.buffer()?,
                m.buffer_mut()?,
                v.buffer_mut()?,
                config.learning_rate,
                config.beta1,
                config.beta2,
                config.eps,
                config.weight_decay,
                bias_correction1,
                bias_correction2,
                n,
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AdamWParams {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

fn padded4(shape: &[usize]) -> [u32; 4] {
    let mut values = [1u32; 4];
    let offset = 4usize.saturating_sub(shape.len());
    for (index, dim) in shape.iter().take(4).enumerate() {
        values[offset + index] = *dim as u32;
    }
    values
}

fn permutation4(axes: &[usize], rank: usize) -> [u32; 4] {
    let mut values = [0u32, 1, 2, 3];
    let offset = 4usize.saturating_sub(rank);
    for (index, axis) in axes.iter().take(4).enumerate() {
        values[offset + index] = (offset + *axis) as u32;
    }
    values
}

fn reduction_mapping(input_rank: usize, output_rank: usize, axes: &[usize]) -> ([u32; 4], u32) {
    let mut mask = 0u32;
    for &axis in axes {
        if axis < input_rank {
            mask |= 1 << (4 - input_rank + axis);
        }
    }
    let mut mapping = [u32::MAX; 4];
    let mut output_axis = 0usize;
    let offset = 4usize.saturating_sub(output_rank);
    for axis in 0..input_rank {
        if axes.contains(&axis) {
            continue;
        }
        if output_axis < output_rank {
            mapping[offset + output_axis] = (4 - input_rank + axis) as u32;
            output_axis += 1;
        }
    }
    (mapping, mask)
}
