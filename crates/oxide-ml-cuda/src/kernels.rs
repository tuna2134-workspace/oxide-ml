use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, LaunchConfig};
use cuda_host::cuda_module;

use crate::access::{AccessRequest, launch_with_access};
use crate::buffer::CudaTensor;
use crate::error::Result;
use crate::event::{CudaEvent, EventPool};

#[cuda_module]
mod kernels {
    use cuda_device::float::{ex2_approx_f32, lg2_approx_f32, sqrt_rn_f32, tanh_approx_f32};
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
                5 => sqrt_rn_f32(value),
                6 => exp_f32(value),
                7 => lg2_approx_f32(value) * std::f32::consts::LN_2,
                8 => {
                    if value < 0.0 {
                        -value
                    } else {
                        value
                    }
                }
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
                5 => {
                    // AbsBackward(x, dy) = dy * sign(x)
                    if av > 0.0 {
                        bv
                    } else if av < 0.0 {
                        -bv
                    } else {
                        0.0
                    }
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
    pub fn copy_flat(input: &[f32], mut output: DisjointSlice<f32>, n: u32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        if let Some(slot) = output.get_mut(idx) {
            *slot = input[i as usize];
        }
    }


    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn moe(
        hidden: &[f32],
        router_scale: &[f32],
        per_expert_scale: &[f32],
        router: &[f32],
        gate_up: &[f32],
        down: &[f32],
        expert_norm: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        hidden_size: u32,
        num_experts: u32,
        top_k: u32,
        intermediate: u32,
        eps: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let total = rows * hidden_size;
        if i as u32 >= total {
            return;
        }
        let row = (i as u32) / hidden_size;
        let out_h = (i as u32) % hidden_size;
        let base = (row * hidden_size) as usize;
        let mut mean_square = 0.0f32;
        for h in 0..hidden_size as usize {
            let value = hidden[base + h];
            mean_square += value * value;
        }
        mean_square /= hidden_size as f32;
        let inv = 1.0f32 / (mean_square + eps).sqrt();

        let mut probs = [0.0f32; 64];
        let mut max = f32::NEG_INFINITY;
        let scale = 1.0f32 / (hidden_size as f32).sqrt();
        for e in 0..num_experts as usize {
            if e >= 64 {
                break;
            }
            let mut score = 0.0f32;
            for h in 0..hidden_size as usize {
                score += router[e * hidden_size as usize + h]
                    * hidden[base + h]
                    * inv
                    * router_scale[h];
            }
            probs[e] = score * scale;
            if probs[e] > max {
                max = probs[e];
            }
        }
        let mut sum = 0.0f32;
        for e in 0..num_experts as usize {
            if e >= 64 {
                break;
            }
            probs[e] = (probs[e] - max).exp();
            sum += probs[e];
        }
        for e in 0..num_experts as usize {
            if e >= 64 {
                break;
            }
            probs[e] /= sum;
        }
        let mut used = [false; 64];
        let mut chosen = [0u32; 8];
        let mut weights = [0.0f32; 8];
        let mut weight_sum = 0.0f32;
        for k in 0..top_k.min(8) as usize {
            let mut best = 0u32;
            let mut best_value = f32::NEG_INFINITY;
            for e in 0..num_experts as usize {
                if e < 64 && !used[e] && probs[e] > best_value {
                    best_value = probs[e];
                    best = e as u32;
                }
            }
            used[best as usize] = true;
            chosen[k] = best;
            weights[k] = best_value;
            weight_sum += best_value;
        }

        let mut accumulator = 0.0f32;
        for k in 0..top_k.min(8) as usize {
            let expert = chosen[k] as usize;
            let weight = weights[k] / weight_sum * per_expert_scale[expert];
            let mut expert_sum = 0.0f32;
            for j in 0..intermediate as usize {
                let gate_row = gate_up
                    [(expert * 2 * intermediate as usize + j) * hidden_size as usize..]
                    [..hidden_size as usize]
                    .iter()
                    .enumerate()
                    .map(|(h, value)| value * hidden[base + h] * inv * expert_norm[h])
                    .sum::<f32>();
                let up_row = gate_up
                    [(expert * 2 * intermediate as usize + intermediate as usize + j)
                        * hidden_size as usize..][..hidden_size as usize]
                    .iter()
                    .enumerate()
                    .map(|(h, value)| value * hidden[base + h] * inv * expert_norm[h])
                    .sum::<f32>();
                let gelu = 0.5
                    * gate_row
                    * (1.0
                        + (0.797_884_6
                            * (gate_row + 0.044_715 * gate_row * gate_row * gate_row))
                            .tanh());
                expert_sum += down[(expert * hidden_size as usize + out_h as usize)
                    * intermediate as usize
                    + j]
                    * gelu
                    * up_row;
            }
            accumulator += weight * expert_sum;
        }
        if let Some(slot) = output.get_mut(idx) {
            *slot = accumulator;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn embedding(
        weight: &[f32],
        indices: &[f32],
        mut output: DisjointSlice<f32>,
        rows: u32,
        cols: u32,
        vocab: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= rows * cols {
            return;
        }
        let row = i as u32 / cols;
        let col = i as u32 % cols;
        let token = indices[row as usize] as u32;
        if let Some(slot) = output.get_mut(idx) {
            *slot = if token < vocab {
                weight[(token * cols + col) as usize]
            } else {
                0.0
            };
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn slice_copy(
        input: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        o1: u32,
        o2: u32,
        o3: u32,
        s0: u32,
        s1: u32,
        s2: u32,
        s3: u32,
        axis: u32,
        start: u32,
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
        let mut coords = [c0, c1, c2, c3];
        coords[axis as usize] += start;
        let offset = coords[0] * s0 + coords[1] * s1 + coords[2] * s2 + coords[3] * s3;
        if let Some(slot) = output.get_mut(idx) {
            *slot = input[offset as usize];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn concat_last(
        i0: &[f32],
        i1: &[f32],
        i2: &[f32],
        i3: &[f32],
        i4: &[f32],
        i5: &[f32],
        i6: &[f32],
        i7: &[f32],
        l0: u32,
        l1: u32,
        l2: u32,
        l3: u32,
        l4: u32,
        l5: u32,
        l6: u32,
        l7: u32,
        mut output: DisjointSlice<f32>,
        n: u32,
        last: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        let row = i as u32 / last;
        let col = i as u32 % last;
        let lengths = [l0, l1, l2, l3, l4, l5, l6, l7];
        let mut segment = 0u32;
        let mut remaining = col;
        while segment < 8 {
            let length = lengths[segment as usize];
            if remaining < length {
                break;
            }
            remaining -= length;
            segment += 1;
        }
        if segment >= 8 {
            return;
        }
        let read = match segment {
            0 => i0,
            1 => i1,
            2 => i2,
            3 => i3,
            4 => i4,
            5 => i5,
            6 => i6,
            _ => i7,
        };
        let source = row * lengths[segment as usize] + remaining;
        if let Some(slot) = output.get_mut(idx) {
            *slot = read[source as usize];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn concat_flat(
        a: &[f32],
        b: &[f32],
        mut output: DisjointSlice<f32>,
        n: u32,
        split: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i as u32 >= n {
            return;
        }
        let value = if (i as u32) < split {
            a[i as usize]
        } else {
            b[(i as u32 - split) as usize]
        };
        if let Some(slot) = output.get_mut(idx) {
            *slot = value;
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
    pub fn log_softmax(
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
            sum += (input[(base + a * inner) as usize] - max_value).exp();
            a += 1;
        }
        let log_sum = lg2_approx_f32(sum) * std::f32::consts::LN_2;
        if let Some(slot) = output.get_mut(idx) {
            *slot = input[i as usize] - max_value - log_sum;
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
    #[kernel]
    #[launch_bounds(256)]
    pub fn im2col(
        input: &[f32],
        mut cols: DisjointSlice<f32>,
        _n: u32,
        c: u32,
        h: u32,
        w: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        oh: u32,
        ow: u32,
        cg: u32,
        group: u32,
        k: u32,
        m: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= m * k {
            return;
        }
        let mm = i / k;
        let kk = i % k;
        let cidx = kk / (kh * kw);
        let rem = kk % (kh * kw);
        let ky = rem / kw;
        let kx = rem % kw;
        let nidx = mm / (oh * ow);
        let spatial = mm % (oh * ow);
        let oy = spatial / ow;
        let ox = spatial % ow;
        let iy = oy * sh + ky * dh;
        let ix = ox * sw + kx * dw;
        let mut value = 0.0f32;
        if iy >= ph && ix >= pw {
            let yy = iy - ph;
            let xx = ix - pw;
            if yy < h && xx < w {
                let channel = group * cg + cidx;
                let offset = ((nidx * c + channel) * h + yy) * w + xx;
                value = input[offset as usize];
            }
        }
        if let Some(slot) = cols.get_mut(idx) {
            *slot = value;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn col2im(
        cols: &[f32],
        mut dx: DisjointSlice<f32>,
        n: u32,
        cg: u32,
        h: u32,
        w: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        oh: u32,
        ow: u32,
        k: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= n * cg * h * w {
            return;
        }
        let nidx = i / (cg * h * w);
        let rem = i % (cg * h * w);
        let cidx = rem / (h * w);
        let spatial = rem % (h * w);
        let yy = spatial / w;
        let xx = spatial % w;
        let mut sum = 0.0f32;
        let mut ky = 0u32;
        while ky < kh {
            let iy = yy + ph;
            let num = iy as i64 - (ky * dh) as i64;
            if num >= 0 && num % sh as i64 == 0 {
                let oy = (num / sh as i64) as u32;
                if oy < oh {
                    let mut kx = 0u32;
                    while kx < kw {
                        let ix = xx + pw;
                        let numx = ix as i64 - (kx * dw) as i64;
                        if numx >= 0 && numx % sw as i64 == 0 {
                            let ox = (numx / sw as i64) as u32;
                            if ox < ow {
                                let kk = (cidx * kh + ky) * kw + kx;
                                let mm = (nidx * oh + oy) * ow + ox;
                                sum += cols[(mm * k + kk) as usize];
                            }
                        }
                        kx += 1;
                    }
                }
            }
            ky += 1;
        }
        if let Some(slot) = dx.get_mut(idx) {
            *slot = sum;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn nchw_to_mat(
        input: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= n * c * h * w {
            return;
        }
        let nidx = i / (c * h * w);
        let rem = i % (c * h * w);
        let cidx = rem / (h * w);
        let spatial = rem % (h * w);
        if let Some(slot) = out.get_mut(idx) {
            *slot = input[((nidx * h * w + spatial) * c + cidx) as usize];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn mat_to_nchw(
        input: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= n * c * h * w {
            return;
        }
        let nidx = i / (c * h * w);
        let rem = i % (c * h * w);
        let cidx = rem / (h * w);
        let spatial = rem % (h * w);
        if let Some(slot) = out.get_mut(idx) {
            *slot = input[((nidx * h * w + spatial) * c + cidx) as usize];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn depthwise_conv(
        input: &[f32],
        weight: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        oh: u32,
        ow: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= n * c * oh * ow {
            return;
        }
        let nidx = i / (c * oh * ow);
        let rem = i % (c * oh * ow);
        let cidx = rem / (oh * ow);
        let spatial = rem % (oh * ow);
        let oy = spatial / ow;
        let ox = spatial % ow;
        let mut sum = 0.0f32;
        let mut ky = 0u32;
        while ky < kh {
            let iy = oy * sh + ky * dh;
            if iy >= ph {
                let yy = iy - ph;
                if yy < h {
                    let mut kx = 0u32;
                    while kx < kw {
                        let ix = ox * sw + kx * dw;
                        if ix >= pw {
                            let xx = ix - pw;
                            if xx < w {
                                let xv = input[(((nidx * c + cidx) * h + yy) * w + xx) as usize];
                                let wv = weight[((cidx * kh + ky) * kw + kx) as usize];
                                sum += xv * wv;
                            }
                        }
                        kx += 1;
                    }
                }
            }
            ky += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = sum;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn depthwise_conv_dx(
        dy: &[f32],
        weight: &[f32],
        mut dx: DisjointSlice<f32>,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dw: u32,
        oh: u32,
        ow: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= n * c * h * w {
            return;
        }
        let nidx = i / (c * h * w);
        let rem = i % (c * h * w);
        let cidx = rem / (h * w);
        let spatial = rem % (h * w);
        let yy = spatial / w;
        let xx = spatial % w;
        let mut sum = 0.0f32;
        let mut ky = 0u32;
        while ky < kh {
            let iy = yy + ph;
            let num = iy as i64 - (ky * dh) as i64;
            if num >= 0 && num % sh as i64 == 0 {
                let oy = (num / sh as i64) as u32;
                if oy < oh {
                    let mut kx = 0u32;
                    while kx < kw {
                        let ix = xx + pw;
                        let numx = ix as i64 - (kx * dw) as i64;
                        if numx >= 0 && numx % sw as i64 == 0 {
                            let ox = (numx / sw as i64) as u32;
                            if ox < ow {
                                let yv = dy[(((nidx * c + cidx) * oh + oy) * ow + ox) as usize];
                                let wv = weight[((cidx * kh + ky) * kw + kx) as usize];
                                sum += yv * wv;
                            }
                        }
                        kx += 1;
                    }
                }
            }
            ky += 1;
        }
        if let Some(slot) = dx.get_mut(idx) {
            *slot = sum;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn depthwise_conv_dw(
        dy: &[f32],
        input: &[f32],
        mut dw: DisjointSlice<f32>,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
        kh: u32,
        kw: u32,
        sh: u32,
        sw: u32,
        ph: u32,
        pw: u32,
        dh: u32,
        dwx: u32,
        oh: u32,
        ow: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= c * kh * kw {
            return;
        }
        let cidx = i / (kh * kw);
        let rem = i % (kh * kw);
        let ky = rem / kw;
        let kx = rem % kw;
        let mut sum = 0.0f32;
        let mut nidx = 0u32;
        while nidx < n {
            let mut oy = 0u32;
            while oy < oh {
                let iy = oy * sh + ky * dh;
                if iy >= ph {
                    let yy = iy - ph;
                    if yy < h {
                        let mut ox = 0u32;
                        while ox < ow {
                            let ix = ox * sw + kx * dwx;
                            if ix >= pw {
                                let xx = ix - pw;
                                if xx < w {
                                    let yv = dy[(((nidx * c + cidx) * oh + oy) * ow + ox) as usize];
                                    let xv = input[(((nidx * c + cidx) * h + yy) * w + xx) as usize];
                                    sum += yv * xv;
                                }
                            }
                            ox += 1;
                        }
                    }
                }
                oy += 1;
            }
            nidx += 1;
        }
        if let Some(slot) = dw.get_mut(idx) {
            *slot = sum;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn channel_sum(
        dy: &[f32],
        mut db: DisjointSlice<f32>,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= c {
            return;
        }
        let mut sum = 0.0f32;
        let mut nidx = 0u32;
        while nidx < n {
            let mut spatial = 0u32;
            while spatial < h * w {
                sum += dy[(((nidx * c + i) * h * w) + spatial) as usize];
                spatial += 1;
            }
            nidx += 1;
        }
        if let Some(slot) = db.get_mut(idx) {
            *slot = sum;
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn bias_add(
        bias: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        c: u32,
        h: u32,
        w: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= n * c * h * w {
            return;
        }
        let cidx = (i / (h * w)) % c;
        if let Some(slot) = out.get_mut(idx) {
            *slot += bias[cidx as usize];
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn cross_entropy(
        logits: &[f32],
        targets: &[f32],
        mut out: DisjointSlice<f32>,
        n: u32,
        c: u32,
    ) {
        static mut SHARED: SharedArray<f32, 256> = SharedArray::UNINIT;
        let tid = thread::index_1d().get();
        let mut local = 0.0f32;
        let mut row = tid as u32;
        while row < n {
            let base = (row * c) as usize;
            let mut max = logits[base];
            let mut i = 1u32;
            while i < c {
                let value = logits[base + i as usize];
                if value > max {
                    max = value;
                }
                i += 1;
            }
            let mut sum = 0.0f32;
            i = 0;
            while i < c {
                sum += exp_f32(logits[base + i as usize] - max);
                i += 1;
            }
            let target = targets[row as usize] as i32;
            let target = if target < 0 {
                0
            } else if target >= c as i32 {
                c - 1
            } else {
                target as u32
            };
            local += (max + lg2_approx_f32(sum) * std::f32::consts::LN_2)
                - logits[base + target as usize];
            row += 256;
        }
        unsafe {
            SHARED[tid] = local;
        }
        thread::sync_threads();
        let mut stride = 128u32;
        while stride > 0 {
            if tid < stride as usize {
                unsafe {
                    SHARED[tid] += SHARED[tid + stride as usize];
                }
            }
            thread::sync_threads();
            stride /= 2;
        }
        if tid == 0 {
            if let Some(slot) = out.get_mut(thread::index_1d()) {
                *slot = unsafe { SHARED[0] } / n as f32;
            }
        }
    }

    #[kernel]
    #[launch_bounds(256)]
    pub fn cross_entropy_backward(
        logits: &[f32],
        targets: &[f32],
        dy: &[f32],
        mut grad: DisjointSlice<f32>,
        n: u32,
        c: u32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get() as u32;
        if i >= n * c {
            return;
        }
        let row = i / c;
        let col = i % c;
        let base = (row * c) as usize;
        let mut max = logits[base];
        let mut k = 1u32;
        while k < c {
            let value = logits[base + k as usize];
            if value > max {
                max = value;
            }
            k += 1;
        }
        let mut sum = 0.0f32;
        k = 0;
        while k < c {
            sum += exp_f32(logits[base + k as usize] - max);
            k += 1;
        }
        let probability = exp_f32(logits[base + col as usize] - max) / sum;
        let target = targets[row as usize] as i32;
        let target = if target < 0 {
            0
        } else if target >= c as i32 {
            c - 1
        } else {
            target as u32
        };
        let delta = if col == target { 1.0f32 } else { 0.0f32 };
        if let Some(slot) = grad.get_mut(idx) {
            *slot = (probability - delta) * dy[0] / n as f32;
        }
    }

}

pub struct CudaKernels {
    module: kernels::LoadedModule,
    events: EventPool,
}

impl CudaKernels {
    pub fn load(context: &Arc<CudaContext>) -> Result<Self> {
        let events = EventPool::new(Arc::clone(context));
        Self::load_with_events(context, events)
    }

    pub fn load_with_events(context: &Arc<CudaContext>, events: EventPool) -> Result<Self> {
        let module = kernels::load(context)?;
        Ok(CudaKernels { module, events })
    }

    pub fn event_pool(&self) -> &EventPool {
        &self.events
    }

    fn launch<R>(
        &self,
        stream: &Arc<CudaStream>,
        requests: &[AccessRequest],
        launch: impl FnOnce() -> Result<R>,
    ) -> Result<(R, Arc<CudaEvent>)> {
        let completion = launch_with_access(&self.events, stream, requests, launch)?;
        Ok((completion.value, completion.event))
    }

    pub fn unary(
        &self,
        stream: &Arc<CudaStream>,
        op: u32,
        input: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let n = input.numel() as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(output)];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn binary(
        &self,
        stream: &Arc<CudaStream>,
        op: u32,
        lhs: &CudaTensor,
        rhs: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let o = padded4(output.shape());
        let a = padded4(lhs.shape());
        let b = padded4(rhs.shape());
        let n = output.numel() as u32;
        let requests = [
            AccessRequest::read(lhs),
            AccessRequest::read(rhs),
            AccessRequest::write(output),
        ];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 3] = [lhs, rhs, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 3] = [lhs, rhs, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn add_relu_backward(
        &self,
        stream: &Arc<CudaStream>,
        lhs: &CudaTensor,
        rhs: &CudaTensor,
        dy: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let o = padded4(output.shape());
        let a = padded4(lhs.shape());
        let b = padded4(rhs.shape());
        let n = output.numel() as u32;
        let requests = [
            AccessRequest::read(lhs),
            AccessRequest::read(rhs),
            AccessRequest::read(dy),
            AccessRequest::write(output),
        ];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 4] = [lhs, rhs, dy, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 4] = [lhs, rhs, dy, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn activation_backward(
        &self,
        stream: &Arc<CudaStream>,
        op: u32,
        x: &CudaTensor,
        dy: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let n = x.numel() as u32;
        let requests = [
            AccessRequest::read(x),
            AccessRequest::read(dy),
            AccessRequest::write(output),
        ];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 3] = [x, dy, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 3] = [x, dy, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn transpose(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        output: &mut CudaTensor,
        axes: &[usize],
    ) -> Result<()> {
        let i = padded4(input.shape());
        let o = padded4(output.shape());
        let p = permutation4(axes, input.shape().len());
        let n = output.numel() as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(output)];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn broadcast(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let o = padded4(output.shape());
        let i = padded4(input.shape());
        let n = output.numel() as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(output)];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }


    fn strides4(shape: &[usize]) -> [u32; 4] {
        let mut dims = [1usize; 4];
        let offset = 4usize.saturating_sub(shape.len());
        for (index, value) in shape.iter().take(4).enumerate() {
            dims[offset + index] = *value;
        }
        let mut strides = [0u32; 4];
        let mut stride = 1usize;
        for axis in (0..4).rev() {
            strides[axis] = stride as u32;
            stride *= dims[axis];
        }
        strides
    }

    fn finish(
        &self,
        launched: Result<((), Arc<CudaEvent>)>,
        inputs: &[&CudaTensor],
        output: &CudaTensor,
    ) -> Result<()> {
        match launched {
            Ok((_, event)) => {
                for tensor in inputs.iter().copied().chain(std::iter::once(output)) {
                    tensor.mark_used(event.clone())?;
                }
                Ok(())
            }
            Err(error) => {
                for tensor in inputs.iter().copied().chain(std::iter::once(output)) {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn copy_flat(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let n = output.numel() as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(output)];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.copy_flat(
                    stream,
                    LaunchConfig::for_num_elems(n),
                    input.buffer()?,
                    output.buffer_mut()?,
                    n,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[input], output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn im2col(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        cols: &mut CudaTensor,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
        dh: usize,
        dw: usize,
        oh: usize,
        ow: usize,
        cg: usize,
        group: usize,
        k: usize,
        m: usize,
    ) -> Result<()> {
        let total = (m * k) as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(cols)];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.im2col(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    input.buffer()?,
                    cols.buffer_mut()?,
                    n as u32,
                    c as u32,
                    h as u32,
                    w as u32,
                    kh as u32,
                    kw as u32,
                    sh as u32,
                    sw as u32,
                    ph as u32,
                    pw as u32,
                    dh as u32,
                    dw as u32,
                    oh as u32,
                    ow as u32,
                    cg as u32,
                    group as u32,
                    k as u32,
                    m as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[input], cols)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn col2im(
        &self,
        stream: &Arc<CudaStream>,
        cols: &CudaTensor,
        dx: &mut CudaTensor,
        n: usize,
        cg: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
        dh: usize,
        dw: usize,
        oh: usize,
        ow: usize,
        k: usize,
    ) -> Result<()> {
        let total = (n * cg * h * w) as u32;
        let requests = [AccessRequest::read(cols), AccessRequest::write(dx)];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.col2im(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    cols.buffer()?,
                    dx.buffer_mut()?,
                    n as u32,
                    cg as u32,
                    h as u32,
                    w as u32,
                    kh as u32,
                    kw as u32,
                    sh as u32,
                    sw as u32,
                    ph as u32,
                    pw as u32,
                    dh as u32,
                    dw as u32,
                    oh as u32,
                    ow as u32,
                    k as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[cols], dx)
    }

    pub fn nchw_to_mat(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        out: &mut CudaTensor,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<()> {
        let total = (n * c * h * w) as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(out)];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.nchw_to_mat(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    input.buffer()?,
                    out.buffer_mut()?,
                    n as u32,
                    c as u32,
                    h as u32,
                    w as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[input], out)
    }

    pub fn mat_to_nchw(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        out: &mut CudaTensor,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<()> {
        let total = (n * c * h * w) as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(out)];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.mat_to_nchw(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    input.buffer()?,
                    out.buffer_mut()?,
                    n as u32,
                    c as u32,
                    h as u32,
                    w as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[input], out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn depthwise_conv(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        weight: &CudaTensor,
        out: &mut CudaTensor,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
        dh: usize,
        dw: usize,
        oh: usize,
        ow: usize,
    ) -> Result<()> {
        let total = (n * c * oh * ow) as u32;
        let requests = [
            AccessRequest::read(input),
            AccessRequest::read(weight),
            AccessRequest::write(out),
        ];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.depthwise_conv(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    input.buffer()?,
                    weight.buffer()?,
                    out.buffer_mut()?,
                    n as u32,
                    c as u32,
                    h as u32,
                    w as u32,
                    kh as u32,
                    kw as u32,
                    sh as u32,
                    sw as u32,
                    ph as u32,
                    pw as u32,
                    dh as u32,
                    dw as u32,
                    oh as u32,
                    ow as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[input, weight], out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn depthwise_conv_dx(
        &self,
        stream: &Arc<CudaStream>,
        dy: &CudaTensor,
        weight: &CudaTensor,
        dx: &mut CudaTensor,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
        dh: usize,
        dw: usize,
        oh: usize,
        ow: usize,
    ) -> Result<()> {
        let total = (n * c * h * w) as u32;
        let requests = [
            AccessRequest::read(dy),
            AccessRequest::read(weight),
            AccessRequest::write(dx),
        ];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.depthwise_conv_dx(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    dy.buffer()?,
                    weight.buffer()?,
                    dx.buffer_mut()?,
                    n as u32,
                    c as u32,
                    h as u32,
                    w as u32,
                    kh as u32,
                    kw as u32,
                    sh as u32,
                    sw as u32,
                    ph as u32,
                    pw as u32,
                    dh as u32,
                    dw as u32,
                    oh as u32,
                    ow as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[dy, weight], dx)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn depthwise_conv_dw(
        &self,
        stream: &Arc<CudaStream>,
        dy: &CudaTensor,
        input: &CudaTensor,
        dw: &mut CudaTensor,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
        dh: usize,
        dwx: usize,
        oh: usize,
        ow: usize,
    ) -> Result<()> {
        let total = (c * kh * kw) as u32;
        let requests = [
            AccessRequest::read(dy),
            AccessRequest::read(input),
            AccessRequest::write(dw),
        ];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.depthwise_conv_dw(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    dy.buffer()?,
                    input.buffer()?,
                    dw.buffer_mut()?,
                    n as u32,
                    c as u32,
                    h as u32,
                    w as u32,
                    kh as u32,
                    kw as u32,
                    sh as u32,
                    sw as u32,
                    ph as u32,
                    pw as u32,
                    dh as u32,
                    dwx as u32,
                    oh as u32,
                    ow as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[dy, input], dw)
    }

    pub fn channel_sum(
        &self,
        stream: &Arc<CudaStream>,
        dy: &CudaTensor,
        db: &mut CudaTensor,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<()> {
        let total = c as u32;
        let requests = [AccessRequest::read(dy), AccessRequest::write(db)];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.channel_sum(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    dy.buffer()?,
                    db.buffer_mut()?,
                    n as u32,
                    c as u32,
                    h as u32,
                    w as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[dy], db)
    }

    pub fn bias_add(
        &self,
        stream: &Arc<CudaStream>,
        bias: &CudaTensor,
        out: &mut CudaTensor,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
    ) -> Result<()> {
        let total = (n * c * h * w) as u32;
        let requests = [AccessRequest::read(bias), AccessRequest::write(out)];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.bias_add(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    bias.buffer()?,
                    out.buffer_mut()?,
                    n as u32,
                    c as u32,
                    h as u32,
                    w as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[bias], out)
    }

    pub fn cross_entropy(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaTensor,
        targets: &CudaTensor,
        out: &mut CudaTensor,
        n: usize,
        c: usize,
    ) -> Result<()> {
        let requests = [
            AccessRequest::read(logits),
            AccessRequest::read(targets),
            AccessRequest::write(out),
        ];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.cross_entropy(
                    stream,
                    LaunchConfig::for_num_elems(256),
                    logits.buffer()?,
                    targets.buffer()?,
                    out.buffer_mut()?,
                    n as u32,
                    c as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[logits, targets], out)
    }

    pub fn cross_entropy_backward(
        &self,
        stream: &Arc<CudaStream>,
        logits: &CudaTensor,
        targets: &CudaTensor,
        dy: &CudaTensor,
        grad: &mut CudaTensor,
        n: usize,
        c: usize,
    ) -> Result<()> {
        let total = (n * c) as u32;
        let requests = [
            AccessRequest::read(logits),
            AccessRequest::read(targets),
            AccessRequest::read(dy),
            AccessRequest::write(grad),
        ];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.cross_entropy_backward(
                    stream,
                    LaunchConfig::for_num_elems(total),
                    logits.buffer()?,
                    targets.buffer()?,
                    dy.buffer()?,
                    grad.buffer_mut()?,
                    n as u32,
                    c as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[logits, targets, dy], grad)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe(
        &self,
        stream: &Arc<CudaStream>,
        hidden: &CudaTensor,
        router_scale: &CudaTensor,
        per_expert_scale: &CudaTensor,
        router: &CudaTensor,
        gate_up: &CudaTensor,
        down: &CudaTensor,
        expert_norm: &CudaTensor,
        output: &mut CudaTensor,
        num_experts: usize,
        top_k: usize,
        intermediate: usize,
        eps: f32,
    ) -> Result<()> {
        let hidden_size = *output.shape().last().unwrap_or(&1);
        let rows = output.numel() / hidden_size;
        let requests = [
            AccessRequest::read(hidden),
            AccessRequest::read(router_scale),
            AccessRequest::read(per_expert_scale),
            AccessRequest::read(router),
            AccessRequest::read(gate_up),
            AccessRequest::read(down),
            AccessRequest::read(expert_norm),
            AccessRequest::write(output),
        ];
        let n = (rows * hidden_size) as u32;
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.moe(
                    stream,
                    LaunchConfig::for_num_elems(n),
                    hidden.buffer()?,
                    router_scale.buffer()?,
                    per_expert_scale.buffer()?,
                    router.buffer()?,
                    gate_up.buffer()?,
                    down.buffer()?,
                    expert_norm.buffer()?,
                    output.buffer_mut()?,
                    rows as u32,
                    hidden_size as u32,
                    num_experts as u32,
                    top_k as u32,
                    intermediate as u32,
                    eps,
                )?;
            }
            Ok(())
        });
        self.finish(
            launched,
            &[
                hidden,
                router_scale,
                per_expert_scale,
                router,
                gate_up,
                down,
                expert_norm,
            ],
            output,
        )
    }

    pub fn embedding(
        &self,
        stream: &Arc<CudaStream>,
        weight: &CudaTensor,
        indices: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let rows = indices.numel() as u32;
        let cols = *output.shape().last().unwrap_or(&1) as u32;
        let vocab = weight.shape()[0] as u32;
        let n = rows * cols;
        let requests = [
            AccessRequest::read(weight),
            AccessRequest::read(indices),
            AccessRequest::write(output),
        ];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.embedding(
                    stream,
                    LaunchConfig::for_num_elems(n),
                    weight.buffer()?,
                    indices.buffer()?,
                    output.buffer_mut()?,
                    rows,
                    cols,
                    vocab,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[weight, indices], output)
    }

    pub fn slice_copy(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        output: &mut CudaTensor,
        axis: usize,
        start: usize,
    ) -> Result<()> {
        let o = padded4(output.shape());
        let strides = Self::strides4(input.shape());
        let axis = axis + 4usize.saturating_sub(input.shape().len());
        let n = output.numel() as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(output)];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.slice_copy(
                    stream,
                    LaunchConfig::for_num_elems(n),
                    input.buffer()?,
                    output.buffer_mut()?,
                    n,
                    o[1],
                    o[2],
                    o[3],
                    strides[0],
                    strides[1],
                    strides[2],
                    strides[3],
                    axis as u32,
                    start as u32,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[input], output)
    }

    pub fn concat_last(
        &self,
        stream: &Arc<CudaStream>,
        inputs: &[&CudaTensor],
        output: &mut CudaTensor,
    ) -> Result<()> {
        let n = output.numel() as u32;
        let last = *output.shape().last().unwrap_or(&1) as u32;
        let mut lengths = [0u32; 8];
        for (index, input) in inputs.iter().enumerate().take(8) {
            lengths[index] = *input.shape().last().unwrap_or(&1) as u32;
        }
        let mut requests: Vec<AccessRequest> =
            inputs.iter().map(|input| AccessRequest::read(*input)).collect();
        requests.push(AccessRequest::write(output));
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.concat_last(
                    stream,
                    LaunchConfig::for_num_elems(n),
                    input_buffer(inputs, 0)?,
                    input_buffer(inputs, 1)?,
                    input_buffer(inputs, 2)?,
                    input_buffer(inputs, 3)?,
                    input_buffer(inputs, 4)?,
                    input_buffer(inputs, 5)?,
                    input_buffer(inputs, 6)?,
                    input_buffer(inputs, 7)?,
                    lengths[0],
                    lengths[1],
                    lengths[2],
                    lengths[3],
                    lengths[4],
                    lengths[5],
                    lengths[6],
                    lengths[7],
                    output.buffer_mut()?,
                    n,
                    last,
                )?;
            }
            Ok(())
        });
        let mut all: Vec<&CudaTensor> = inputs.to_vec();
        all.push(output);
        match launched {
            Ok((_, event)) => {
                for tensor in all {
                    tensor.mark_used(event.clone())?;
                }
                Ok(())
            }
            Err(error) => {
                for tensor in all {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn concat_flat(
        &self,
        stream: &Arc<CudaStream>,
        first: &CudaTensor,
        second: &CudaTensor,
        output: &mut CudaTensor,
    ) -> Result<()> {
        let n = output.numel() as u32;
        let split = first.numel() as u32;
        let requests = [
            AccessRequest::read(first),
            AccessRequest::read(second),
            AccessRequest::write(output),
        ];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.concat_flat(
                    stream,
                    LaunchConfig::for_num_elems(n),
                    first.buffer()?,
                    second.buffer()?,
                    output.buffer_mut()?,
                    n,
                    split,
                )?;
            }
            Ok(())
        });
        self.finish(launched, &[first, second], output)
    }

    pub fn reduce(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        output: &mut CudaTensor,
        axes: &[usize],
        mean: bool,
    ) -> Result<()> {
        let i = padded4(input.shape());
        let o = padded4(output.shape());
        let (to_in, mask) = reduction_mapping(input.shape().len(), output.shape().len(), axes);
        let n = output.numel() as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(output)];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn softmax(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        output: &mut CudaTensor,
        axis: usize,
    ) -> Result<()> {
        let dims = input.shape();
        let axis_len = dims[axis] as u32;
        let inner: u32 = dims[axis + 1..].iter().product::<usize>() as u32;
        let n = input.numel() as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(output)];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 2] = [input, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn log_softmax(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        output: &mut CudaTensor,
        axis: usize,
    ) -> Result<()> {
        let dims = input.shape();
        let axis_len = dims[axis] as u32;
        let inner: u32 = dims[axis + 1..].iter().product::<usize>() as u32;
        let n = input.numel() as u32;
        let requests = [AccessRequest::read(input), AccessRequest::write(output)];
        let launched = self.launch(stream, &requests, || {
            unsafe {
                self.module.log_softmax(
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
        });
        self.finish(launched, &[input], output)
    }

    pub fn layer_norm(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaTensor,
        gamma: &CudaTensor,
        beta: &CudaTensor,
        output: &mut CudaTensor,
        norm: u32,
    ) -> Result<()> {
        let n = input.numel() as u32;
        let requests = [
            AccessRequest::read(input),
            AccessRequest::read(gamma),
            AccessRequest::read(beta),
            AccessRequest::write(output),
        ];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 4] = [input, gamma, beta, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 4] = [input, gamma, beta, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn mse(
        &self,
        stream: &Arc<CudaStream>,
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
        let requests = [
            AccessRequest::read(prediction),
            AccessRequest::read(target),
            AccessRequest::write(output),
        ];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 3] = [prediction, target, output];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 3] = [prediction, target, output];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn mse_backward(
        &self,
        stream: &Arc<CudaStream>,
        prediction: &CudaTensor,
        target: &CudaTensor,
        dy: &CudaTensor,
        dpred: &mut CudaTensor,
        dtarget: &mut CudaTensor,
        mean: bool,
    ) -> Result<()> {
        let n = prediction.numel() as u32;
        let requests = [
            AccessRequest::read(prediction),
            AccessRequest::read(target),
            AccessRequest::read(dy),
            AccessRequest::write(dpred),
            AccessRequest::write(dtarget),
        ];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 5] = [prediction, target, dy, dpred, dtarget];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 5] = [prediction, target, dy, dpred, dtarget];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn scale(
        &self,
        stream: &Arc<CudaStream>,
        tensor: &mut CudaTensor,
        factor: f32,
    ) -> Result<()> {
        let n = tensor.numel() as u32;
        let requests = [AccessRequest::read_write(tensor)];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 1] = [tensor];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 1] = [tensor];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn gemm_naive(
        &self,
        stream: &Arc<CudaStream>,
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
        let requests = [
            AccessRequest::read(a),
            AccessRequest::read(b),
            AccessRequest::write(out),
        ];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 3] = [a, b, out];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 3] = [a, b, out];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    pub fn gemm_tiled(
        &self,
        stream: &Arc<CudaStream>,
        a: &CudaTensor,
        b: &CudaTensor,
        out: &mut CudaTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let tiles = m.div_ceil(16) * n.div_ceil(16);
        let total = (tiles * 256) as u32;
        let requests = [
            AccessRequest::read(a),
            AccessRequest::read(b),
            AccessRequest::write(out),
        ];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 3] = [a, b, out];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 3] = [a, b, out];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }
    pub fn sgd(
        &self,
        stream: &Arc<CudaStream>,
        param: &mut CudaTensor,
        grad: &CudaTensor,
        lr: f32,
    ) -> Result<()> {
        let n = param.numel() as u32;
        let requests = [AccessRequest::read_write(param), AccessRequest::read(grad)];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 2] = [param, grad];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 2] = [param, grad];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adamw(
        &self,
        stream: &Arc<CudaStream>,
        param: &mut CudaTensor,
        grad: &CudaTensor,
        m: &mut CudaTensor,
        v: &mut CudaTensor,
        config: AdamWParams,
        bias_correction1: f32,
        bias_correction2: f32,
    ) -> Result<()> {
        let n = param.numel() as u32;
        let requests = [
            AccessRequest::read_write(param),
            AccessRequest::read(grad),
            AccessRequest::read_write(m),
            AccessRequest::read_write(v),
        ];
        let launched = self.launch(stream, &requests, || {
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
        });
        match launched {
            Ok((value, event)) => {
                let tensors: [&CudaTensor; 4] = [param, grad, m, v];
                for tensor in tensors {
                    if let Err(error) = tensor.mark_used(event.clone()) {
                        for tensor in tensors {
                            tensor.storage().invalidate();
                        }
                        return Err(error);
                    }
                }
                Ok(value)
            }
            Err(error) => {
                let tensors: [&CudaTensor; 4] = [param, grad, m, v];
                for tensor in tensors {
                    tensor.storage().invalidate();
                }
                Err(error)
            }
        }
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

fn input_buffer<'a>(
    inputs: &'a [&'a CudaTensor],
    index: usize,
) -> Result<&'a cuda_core::DeviceBuffer<f32>> {
    let tensor = inputs.get(index).copied().unwrap_or(inputs[0]);
    tensor.buffer()
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
    let input_offset = 4usize.saturating_sub(input_rank);
    let output_offset = 4usize.saturating_sub(output_rank);
    let keep_dims = output_rank == input_rank;
    let mut output_axis = 0usize;
    for axis in 0..input_rank {
        if axes.contains(&axis) {
            continue;
        }
        let input_position = input_offset + axis;
        let output_position = if keep_dims {
            input_position
        } else {
            output_offset + output_axis
        };
        mapping[output_position] = input_position as u32;
        output_axis += 1;
    }
    (mapping, mask)
}
