use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::DType;

/// Numeric element type accepted by the reference CPU backend.
///
/// Keeping kernels generic over this trait gives F32 and F64 the same code
/// path. Unsupported float types (F16/BF16) are intentionally not implemented:
/// the executor rejects them instead of silently converting to F32.
pub trait Scalar:
    Copy
    + Send
    + Sync
    + 'static
    + std::fmt::Debug
    + Default
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    const DTYPE: DType;

    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn from_usize(value: usize) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn tanh(self) -> Self;
    fn powi(self, exponent: i32) -> Self;
    fn abs(self) -> Self;
    fn fract(self) -> Self;
    fn is_finite(self) -> bool;

    fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }

    fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }

    fn recip(self) -> Self {
        Self::one() / self
    }
}

impl Scalar for f32 {
    const DTYPE: DType = DType::F32;

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn from_usize(value: usize) -> Self {
        value as f32
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    fn tanh(self) -> Self {
        f32::tanh(self)
    }

    fn powi(self, exponent: i32) -> Self {
        f32::powi(self, exponent)
    }

    fn abs(self) -> Self {
        f32::abs(self)
    }

    fn fract(self) -> Self {
        f32::fract(self)
    }

    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
}

impl Scalar for f64 {
    const DTYPE: DType = DType::F64;

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn from_usize(value: usize) -> Self {
        value as f64
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn tanh(self) -> Self {
        f64::tanh(self)
    }

    fn powi(self, exponent: i32) -> Self {
        f64::powi(self, exponent)
    }

    fn abs(self) -> Self {
        f64::abs(self)
    }

    fn fract(self) -> Self {
        f64::fract(self)
    }

    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}
