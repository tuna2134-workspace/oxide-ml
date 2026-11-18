use crate::{DType, Error, Result, Scalar, Shape};

/// Contiguous row-major reference tensor.
///
/// The CPU backend is intentionally a correctness-first reference
/// implementation. `T` is `f32` by default, so existing code does not need to
/// spell the scalar type; F64 execution uses `DenseTensor<f64>`.
#[derive(Clone, Debug, PartialEq)]
pub struct DenseTensor<T = f32> {
    shape: Shape,
    data: Vec<T>,
}

impl<T: Scalar> DenseTensor<T> {
    pub fn new(shape: Shape, data: Vec<T>) -> Result<Self> {
        let dims = shape.static_dims().ok_or_else(|| {
            Error::invalid_shape(
                "DenseTensor",
                "reference tensors require a fully static shape",
            )
        })?;
        let expected = dims.iter().product::<usize>();
        if data.len() != expected {
            return Err(Error::invalid_shape(
                "DenseTensor",
                format!(
                    "shape {shape} requires {expected} elements, got {}",
                    data.len()
                ),
            ));
        }
        Ok(DenseTensor { shape, data })
    }

    pub fn from_vec(dims: impl AsRef<[usize]>, data: Vec<T>) -> Result<Self> {
        Self::new(Shape::from_static(dims), data)
    }

    pub fn zeros(dims: impl AsRef<[usize]>) -> Self {
        let shape = Shape::from_static(dims);
        let len = shape.numel_static().unwrap_or(0);
        DenseTensor {
            shape,
            data: vec![T::zero(); len],
        }
    }

    pub fn ones(dims: impl AsRef<[usize]>) -> Self {
        let shape = Shape::from_static(dims);
        let len = shape.numel_static().unwrap_or(0);
        DenseTensor {
            shape,
            data: vec![T::one(); len],
        }
    }

    pub fn full(dims: impl AsRef<[usize]>, value: T) -> Self {
        let shape = Shape::from_static(dims);
        let len = shape.numel_static().unwrap_or(0);
        DenseTensor {
            shape,
            data: vec![value; len],
        }
    }

    pub fn scalar(value: T) -> Self {
        DenseTensor {
            shape: Shape::scalar(),
            data: vec![value],
        }
    }

    pub fn dtype(&self) -> DType {
        T::DTYPE
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Concrete dimensions in row-major order.
    ///
    /// This allocates because `Shape` stores symbolic `Dim`s. Execution paths
    /// should call this once per kernel rather than in inner loops.
    pub fn dims(&self) -> Vec<usize> {
        self.static_dims()
    }

    pub fn static_dims(&self) -> Vec<usize> {
        self.shape
            .static_dims()
            .expect("DenseTensor shape invariant violated")
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    pub fn strides(&self) -> Vec<usize> {
        strides_for(&self.static_dims())
    }

    pub fn get(&self, indices: &[usize]) -> Result<T> {
        let offset = self.offset(indices)?;
        Ok(self.data[offset])
    }

    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        let offset = self.offset(indices)?;
        self.data[offset] = value;
        Ok(())
    }

    fn offset(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.rank() {
            return Err(Error::RankMismatch {
                context: "tensor indexing".into(),
                expected: self.rank(),
                got: indices.len(),
            });
        }
        let dims = self.static_dims();
        let strides = strides_for(&dims);
        let mut offset = 0;
        for ((index, dim), stride) in indices.iter().zip(&dims).zip(&strides) {
            if index >= dim {
                return Err(Error::execution(format!(
                    "index {index} out of bounds for dimension {dim} in {:?}",
                    dims
                )));
            }
            offset += index * stride;
        }
        Ok(offset)
    }

    pub fn reshape(&self, dims: impl AsRef<[usize]>) -> Result<Self> {
        let dims = dims.as_ref();
        let expected = dims.iter().product::<usize>();
        if expected != self.numel() {
            return Err(Error::invalid_shape(
                "reshape",
                format!(
                    "cannot reshape {} elements into {dims:?} ({expected} elements)",
                    self.numel()
                ),
            ));
        }
        Self::from_vec(dims, self.data.clone())
    }

    pub fn transpose(&self, axes: &[usize]) -> Result<Self> {
        let dims = self.static_dims();
        if axes.len() != dims.len() {
            return Err(Error::RankMismatch {
                context: "transpose axes".into(),
                expected: dims.len(),
                got: axes.len(),
            });
        }
        let mut seen = vec![false; dims.len()];
        for &axis in axes {
            if axis >= dims.len() || seen[axis] {
                return Err(Error::InvalidAxis {
                    axis: axis as i32,
                    rank: dims.len(),
                });
            }
            seen[axis] = true;
        }
        let out_dims: Vec<usize> = axes.iter().map(|&axis| dims[axis]).collect();
        let mut out = vec![T::zero(); self.numel()];
        let in_strides = strides_for(&dims);
        let out_strides = strides_for(&out_dims);
        for (linear, slot) in out.iter_mut().enumerate() {
            let out_index = unravel(linear, &out_dims, &out_strides);
            let mut in_index = vec![0usize; dims.len()];
            for (out_axis, &in_axis) in axes.iter().enumerate() {
                in_index[in_axis] = out_index[out_axis];
            }
            *slot = self.data[dot(&in_index, &in_strides)];
        }
        Self::from_vec(&out_dims, out)
    }

    pub fn transpose_last_two(&self) -> Result<Self> {
        let rank = self.rank();
        if rank < 2 {
            return Err(Error::RankMismatch {
                context: "transpose_last_two".into(),
                expected: 2,
                got: rank,
            });
        }
        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(rank - 1, rank - 2);
        self.transpose(&axes)
    }

    pub fn broadcast_to(&self, target: impl AsRef<[usize]>) -> Result<Self> {
        let target = target.as_ref();
        let source_dims = self.static_dims();
        if target.len() < source_dims.len() {
            return Err(Error::InvalidBroadcast {
                lhs: self.shape.clone(),
                rhs: Shape::from_static(target),
            });
        }
        let offset = target.len() - source_dims.len();
        for (index, &source) in source_dims.iter().enumerate() {
            let dest = target[offset + index];
            if source != 1 && source != dest {
                return Err(Error::InvalidBroadcast {
                    lhs: self.shape.clone(),
                    rhs: Shape::from_static(target),
                });
            }
        }
        let out_len = target.iter().product::<usize>();
        let in_strides = strides_for(&source_dims);
        let out_strides = strides_for(target);
        let mut out = Vec::with_capacity(out_len);
        for linear in 0..out_len {
            let out_index = unravel(linear, target, &out_strides);
            let mut in_index = vec![0usize; source_dims.len()];
            for axis in 0..source_dims.len() {
                in_index[axis] = if source_dims[axis] == 1 {
                    0
                } else {
                    out_index[offset + axis]
                };
            }
            out.push(self.data[dot(&in_index, &in_strides)]);
        }
        Self::from_vec(target, out)
    }

    pub fn map(&self, mut f: impl FnMut(T) -> T) -> Self {
        DenseTensor {
            shape: self.shape.clone(),
            data: self.data.iter().copied().map(&mut f).collect(),
        }
    }

    pub fn zip_map(&self, rhs: &DenseTensor<T>, mut f: impl FnMut(T, T) -> T) -> Result<Self> {
        if self.static_dims() != rhs.static_dims() {
            return Err(Error::shape_mismatch(
                "zip_map",
                self.shape.clone(),
                rhs.shape.clone(),
            ));
        }
        Ok(DenseTensor {
            shape: self.shape.clone(),
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(&a, &b)| f(a, b))
                .collect(),
        })
    }

    pub fn sum(&self) -> T {
        self.data
            .iter()
            .copied()
            .fold(T::zero(), |acc, value| acc + value)
    }

    pub fn mean(&self) -> T {
        if self.data.is_empty() {
            T::zero()
        } else {
            self.sum() / T::from_usize(self.data.len())
        }
    }

    pub fn scale(&self, factor: T) -> Self {
        self.map(|value| value * factor)
    }

    /// Copy `other` into this tensor, preserving the backing allocation when
    /// the element count matches. Used by the planned executor's buffer arena.
    pub fn copy_from(&mut self, other: &DenseTensor<T>) {
        if self.data.len() == other.data.len() {
            self.data.copy_from_slice(&other.data);
            self.shape = other.shape.clone();
        } else {
            self.shape = other.shape.clone();
            self.data = other.data.clone();
        }
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        self.data.iter().map(|value| value.to_f64()).collect()
    }

    pub fn from_f64_vec(dims: impl AsRef<[usize]>, values: Vec<f64>) -> Result<Self> {
        Self::from_vec(
            dims,
            values.into_iter().map(T::from_f64).collect::<Vec<_>>(),
        )
    }

    pub fn with_shape(mut self, shape: Shape) -> Result<Self> {
        if shape.numel_static() != Some(self.numel()) {
            return Err(Error::invalid_shape(
                "DenseTensor::with_shape",
                format!("shape {shape} does not match {} elements", self.numel()),
            ));
        }
        self.shape = shape;
        Ok(self)
    }
}

/// Statically ranked public tensor type.
///
/// This is the ergonomic data type for inputs, targets and model outputs.
/// Graph values use `ValueId`, not this type.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor<const R: usize, T = f32> {
    inner: DenseTensor<T>,
}

impl<const R: usize, T: Scalar> Tensor<R, T> {
    pub fn from_vec(shape: [usize; R], data: Vec<T>) -> Result<Self> {
        Ok(Tensor {
            inner: DenseTensor::from_vec(shape, data)?,
        })
    }

    pub fn zeros(shape: [usize; R]) -> Self {
        Tensor {
            inner: DenseTensor::zeros(shape),
        }
    }

    pub fn ones(shape: [usize; R]) -> Self {
        Tensor {
            inner: DenseTensor::ones(shape),
        }
    }

    pub fn shape(&self) -> [usize; R] {
        self.inner
            .static_dims()
            .try_into()
            .expect("Tensor rank invariant")
    }

    pub fn data(&self) -> &[T] {
        self.inner.data()
    }

    pub fn into_inner(self) -> DenseTensor<T> {
        self.inner
    }

    pub fn inner(&self) -> &DenseTensor<T> {
        &self.inner
    }
}

impl<const R: usize, T: Scalar> From<Tensor<R, T>> for DenseTensor<T> {
    fn from(value: Tensor<R, T>) -> Self {
        value.inner
    }
}

impl<const R: usize, T: Scalar> TryFrom<DenseTensor<T>> for Tensor<R, T> {
    type Error = Error;

    fn try_from(value: DenseTensor<T>) -> Result<Self> {
        if value.rank() != R {
            return Err(Error::RankMismatch {
                context: "Tensor::try_from".into(),
                expected: R,
                got: value.rank(),
            });
        }
        Ok(Tensor { inner: value })
    }
}

pub(crate) fn strides_for(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    let mut acc = 1usize;
    for index in (0..dims.len()).rev() {
        strides[index] = acc;
        acc *= dims[index];
    }
    strides
}

pub(crate) fn dot(indices: &[usize], strides: &[usize]) -> usize {
    indices
        .iter()
        .zip(strides)
        .map(|(index, stride)| index * stride)
        .sum()
}

pub(crate) fn unravel(mut linear: usize, dims: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut index = vec![0usize; dims.len()];
    for axis in 0..dims.len() {
        if dims[axis] == 0 {
            continue;
        }
        index[axis] = linear / strides[axis];
        linear %= strides[axis];
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reshape_and_transpose() {
        let tensor =
            DenseTensor::from_vec([2, 3], (0..6).map(|value| value as f32).collect()).unwrap();
        let transposed = tensor.transpose(&[1, 0]).unwrap();
        assert_eq!(transposed.static_dims(), vec![3, 2]);
        assert_eq!(transposed.data(), &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        let reshaped = transposed.reshape([6]).unwrap();
        assert_eq!(reshaped.data(), transposed.data());
    }

    #[test]
    fn broadcast_matches_numpy_layout() {
        let tensor = DenseTensor::from_vec([3, 1], vec![1.0, 2.0, 3.0]).unwrap();
        let out = tensor.broadcast_to([2, 3, 4]).unwrap();
        assert_eq!(out.static_dims(), vec![2, 3, 4]);
        assert_eq!(out.get(&[1, 2, 3]).unwrap(), 3.0);
    }

    #[test]
    fn f64_tensor_roundtrip() {
        let tensor = DenseTensor::from_vec([2, 2], vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.dtype(), DType::F64);
        assert_eq!(tensor.sum(), 10.0);
    }
}
