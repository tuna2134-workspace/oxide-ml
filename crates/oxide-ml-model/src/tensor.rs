//! Host-resident tensor values used at the model API boundary.
//!
//! `Tensor` is *data*, not a graph value. Graph values are `ValueId`s created
//! inside `#[model]` bodies. The same `Tensor` type is used for model inputs,
//! datasets, checkpoints and inference results; device placement is carried as
//! metadata so a CUDA caller can move a tensor without changing model code.

use oxide_ml_core::{DenseTensor, Device, Error, Result, Scalar, Shape};

/// Device-aware dense tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<T: Scalar = f32> {
    dense: DenseTensor<T>,
    device: Device,
}

impl<T: Scalar> Tensor<T> {
    pub fn from_dense(dense: DenseTensor<T>) -> Self {
        Tensor {
            dense,
            device: Device::Cpu,
        }
    }

    pub fn from_vec(shape: impl AsRef<[usize]>, data: Vec<T>) -> Result<Self> {
        Ok(Self::from_dense(DenseTensor::from_vec(shape, data)?))
    }

    pub fn zeros(shape: impl AsRef<[usize]>) -> Self {
        Self::from_dense(DenseTensor::zeros(shape))
    }

    pub fn ones(shape: impl AsRef<[usize]>) -> Self {
        Self::from_dense(DenseTensor::ones(shape))
    }

    pub fn full(shape: impl AsRef<[usize]>, value: T) -> Self {
        Self::from_dense(DenseTensor::full(shape, value))
    }

    pub fn scalar(value: T) -> Self {
        Self::from_dense(DenseTensor::scalar(value))
    }

    pub fn dense(&self) -> &DenseTensor<T> {
        &self.dense
    }

    pub fn dense_mut(&mut self) -> &mut DenseTensor<T> {
        &mut self.dense
    }

    pub fn into_dense(self) -> DenseTensor<T> {
        self.dense
    }

    pub fn shape(&self) -> &Shape {
        self.dense.shape()
    }

    pub fn dims(&self) -> Vec<usize> {
        self.dense.dims()
    }

    pub fn rank(&self) -> usize {
        self.dense.rank()
    }

    pub fn numel(&self) -> usize {
        self.dense.numel()
    }

    pub fn data(&self) -> &[T] {
        self.dense.data()
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        self.dense.data_mut()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn to(self, device: Device) -> Self {
        self.with_device(device)
    }

    pub fn is_scalar(&self) -> bool {
        self.dense.is_scalar()
    }

    pub fn reshape(&self, shape: impl AsRef<[usize]>) -> Result<Self> {
        Ok(Self::from_dense(self.dense.reshape(shape)?).with_device(self.device))
    }

    pub fn transpose(&self, axes: &[usize]) -> Result<Self> {
        Ok(Self::from_dense(self.dense.transpose(axes)?).with_device(self.device))
    }

    pub fn transpose_last_two(&self) -> Result<Self> {
        Ok(Self::from_dense(self.dense.transpose_last_two()?).with_device(self.device))
    }

    pub fn broadcast_to(&self, shape: impl AsRef<[usize]>) -> Result<Self> {
        Ok(Self::from_dense(self.dense.broadcast_to(shape)?).with_device(self.device))
    }

    pub fn map(&self, f: impl FnMut(T) -> T) -> Self {
        Self::from_dense(self.dense.map(f)).with_device(self.device)
    }

    pub fn exp(&self) -> Self {
        self.map(T::exp)
    }

    pub fn ln(&self) -> Self {
        self.map(T::ln)
    }

    pub fn sqrt(&self) -> Self {
        self.map(T::sqrt)
    }

    pub fn abs(&self) -> Self {
        self.map(T::abs)
    }

    pub fn tanh(&self) -> Self {
        self.map(T::tanh)
    }

    pub fn relu(&self) -> Self {
        self.map(|value| if value > T::zero() { value } else { T::zero() })
    }

    pub fn sigmoid(&self) -> Self {
        let one = T::one();
        self.map(|value| one / (one + (-value).exp()))
    }

    pub fn add(&self, other: &Self) -> Result<Self> {
        self.zip_broadcast(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Self) -> Result<Self> {
        self.zip_broadcast(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.zip_broadcast(other, |a, b| a * b)
    }

    pub fn div(&self, other: &Self) -> Result<Self> {
        self.zip_broadcast(other, |a, b| a / b)
    }

    pub fn mul_scalar(&self, value: T) -> Self {
        self.map(|element| element * value)
    }

    pub fn add_scalar(&self, value: T) -> Self {
        self.map(|element| element + value)
    }

    pub fn zip_broadcast(&self, other: &Self, mut f: impl FnMut(T, T) -> T) -> Result<Self> {
        let lhs_dims = self.dims();
        let rhs_dims = other.dims();
        let shape = self.shape().broadcast(other.shape())?;
        let out_dims = shape.static_dims().ok_or_else(|| {
            Error::invalid_shape("tensor broadcasting", "broadcast produced a symbolic shape")
        })?;
        let lhs = if lhs_dims == out_dims {
            self.dense.clone()
        } else {
            self.dense.broadcast_to(&out_dims)?
        };
        let rhs = if rhs_dims == out_dims {
            other.dense.clone()
        } else {
            other.dense.broadcast_to(&out_dims)?
        };
        Ok(Self::from_dense(lhs.zip_map(&rhs, &mut f)?).with_device(self.device))
    }

    /// Matrix multiplication for rank-2 tensors.
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        let lhs_dims = self.dims();
        let rhs_dims = other.dims();
        if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
            return Err(Error::RankMismatch {
                context: "Tensor::matmul".into(),
                expected: 2,
                got: lhs_dims.len().min(rhs_dims.len()),
            });
        }
        let (m, k) = (lhs_dims[0], lhs_dims[1]);
        let (k2, n) = (rhs_dims[0], rhs_dims[1]);
        if k != k2 {
            return Err(Error::shape_mismatch(
                "Tensor::matmul",
                Shape::from_static([k, n]),
                Shape::from_static([k2, n]),
            ));
        }
        let mut out = vec![T::zero(); m * n];
        for row in 0..m {
            for inner in 0..k {
                let lhs = self.data()[row * k + inner];
                if lhs == T::zero() {
                    continue;
                }
                for col in 0..n {
                    out[row * n + col] += lhs * other.data()[inner * n + col];
                }
            }
        }
        Ok(Self::from_vec([m, n], out)?.with_device(self.device))
    }

    pub fn get(&self, indices: &[usize]) -> Result<T> {
        self.dense.get(indices)
    }

    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        self.dense.set(indices, value)
    }

    pub fn sum_all(&self) -> T {
        self.dense.sum()
    }

    pub fn mean_all(&self) -> T {
        self.dense.mean()
    }

    /// Extract sample `index` from a tensor whose first dimension is the batch.
    pub fn row(&self, index: usize) -> Result<Self> {
        let dims = self.dims();
        if dims.is_empty() {
            return Err(Error::invalid_shape(
                "Tensor::row",
                "cannot index a scalar tensor",
            ));
        }
        if index >= dims[0] {
            return Err(Error::execution(format!(
                "row index {index} out of bounds for batch size {}",
                dims[0]
            )));
        }
        let row_len: usize = dims[1..].iter().product();
        let start = index * row_len;
        let data = self.data()[start..start + row_len].to_vec();
        Ok(Self::from_vec(&dims[1..], data)?.with_device(self.device))
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        self.dense.to_f64_vec()
    }

    pub fn from_f64_vec(shape: impl AsRef<[usize]>, values: Vec<f64>) -> Result<Self> {
        Ok(Self::from_dense(DenseTensor::from_f64_vec(shape, values)?))
    }
}

impl<T: Scalar> From<DenseTensor<T>> for Tensor<T> {
    fn from(dense: DenseTensor<T>) -> Self {
        Tensor::from_dense(dense)
    }
}

impl<T: Scalar> From<Tensor<T>> for DenseTensor<T> {
    fn from(tensor: Tensor<T>) -> Self {
        tensor.into_dense()
    }
}

/// Stack per-sample tensors along a new leading batch dimension.
pub fn stack<T: Scalar>(tensors: &[Tensor<T>]) -> Result<Tensor<T>> {
    let first = tensors
        .first()
        .ok_or_else(|| Error::invalid_shape("stack", "cannot stack an empty batch"))?;
    let sample_dims = first.dims();
    for tensor in tensors {
        if tensor.dims() != sample_dims {
            return Err(Error::invalid_shape(
                "stack",
                format!(
                    "batch elements have inconsistent shapes {:?} and {:?}",
                    sample_dims,
                    tensor.dims()
                ),
            ));
        }
    }
    let sample_len = first.numel();
    let mut dims = Vec::with_capacity(sample_dims.len() + 1);
    dims.push(tensors.len());
    dims.extend_from_slice(&sample_dims);
    let mut data = Vec::with_capacity(sample_len * tensors.len());
    for tensor in tensors {
        data.extend_from_slice(tensor.data());
    }
    Tensor::from_vec(&dims, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcasts_binary_ops() {
        let a = Tensor::<f32>::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Tensor::<f32>::from_vec([3], vec![1., 10., 100.]).unwrap();
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.dims(), vec![2, 3]);
        assert_eq!(sum.data(), &[2., 12., 103., 5., 15., 106.]);
    }

    #[test]
    fn matmul_matches_reference() {
        let a = Tensor::<f32>::from_vec([2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let b = Tensor::<f32>::from_vec([3, 2], vec![7., 8., 9., 10., 11., 12.]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dims(), vec![2, 2]);
        assert_eq!(c.data(), &[58., 64., 139., 154.]);
    }

    #[test]
    fn stack_and_row_round_trip() {
        let a = Tensor::<f32>::from_vec([2], vec![1., 2.]).unwrap();
        let b = Tensor::<f32>::from_vec([2], vec![3., 4.]).unwrap();
        let batch = stack(&[a, b]).unwrap();
        assert_eq!(batch.dims(), vec![2, 2]);
        assert_eq!(batch.row(1).unwrap().data(), &[3., 4.]);
    }
}
