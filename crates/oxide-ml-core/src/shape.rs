use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// A named dimension, e.g. `Batch` in `[Batch, 784]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SymbolId(pub u32);

/// One dimension of a symbolic shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dim {
    /// Known at graph-construction time.
    Static(usize),
    /// Resolved from an input binding, e.g. batch size.
    Symbol(SymbolId),
    /// Accepted at graph-construction time but only known at execution time.
    Dynamic,
}

impl Dim {
    pub const fn static_usize(self) -> Option<usize> {
        match self {
            Dim::Static(value) => Some(value),
            _ => None,
        }
    }

    pub const fn is_dynamic(self) -> bool {
        matches!(self, Dim::Dynamic)
    }

    /// Resolve this dimension against a set of symbol bindings.
    pub fn bind(self, bindings: &HashMap<SymbolId, usize>) -> Result<usize> {
        match self {
            Dim::Static(value) => Ok(value),
            Dim::Symbol(symbol) => bindings.get(&symbol).copied().ok_or_else(|| {
                Error::invalid_shape(
                    "symbol binding",
                    format!("symbol {symbol:?} has not been bound by an input"),
                )
            }),
            Dim::Dynamic => Err(Error::invalid_shape(
                "symbol binding",
                "dynamic dimension was not specialized",
            )),
        }
    }

    /// Structural compatibility, not broadcast compatibility.
    pub fn is_compatible(self, other: Dim) -> bool {
        match (self, other) {
            (Dim::Static(a), Dim::Static(b)) => a == b,
            (Dim::Symbol(a), Dim::Symbol(b)) => a == b,
            (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
            _ => false,
        }
    }
}

/// Row-major logical tensor shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<Dim>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<Dim>>) -> Self {
        Shape { dims: dims.into() }
    }

    pub fn from_static(dims: impl AsRef<[usize]>) -> Self {
        Shape {
            dims: dims.as_ref().iter().copied().map(Dim::Static).collect(),
        }
    }

    pub fn scalar() -> Self {
        Shape { dims: Vec::new() }
    }

    /// Parse a compact notation: `"[Batch, 784]"`.
    ///
    /// Numeric components become static dims; identifiers are interned by the
    /// graph builder (the DSL uses the builder's symbol table).
    pub fn parse(input: &str) -> Result<Shape> {
        let trimmed = input.trim();
        let inner = trimmed
            .strip_prefix('[')
            .and_then(|value| value.strip_suffix(']'))
            .ok_or_else(|| {
                Error::invalid_shape("shape parser", format!("missing [ ] in {input:?}"))
            })?;
        if inner.trim().is_empty() {
            return Ok(Shape::scalar());
        }
        let mut dims = Vec::new();
        for part in inner.split(',') {
            let part = part.trim();
            if part.is_empty() {
                return Err(Error::invalid_shape("shape parser", "empty dimension"));
            }
            if part == "_" || part.eq_ignore_ascii_case("dynamic") {
                dims.push(Dim::Dynamic);
            } else if let Ok(value) = part.parse::<usize>() {
                dims.push(Dim::Static(value));
            } else {
                return Err(Error::invalid_shape(
                    "shape parser",
                    format!("use GraphBuilder::symbol for named dimension {part:?}"),
                ));
            }
        }
        Ok(Shape { dims })
    }

    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    pub fn into_dims(self) -> Vec<Dim> {
        self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn dim(&self, index: usize) -> Option<Dim> {
        self.dims.get(index).copied()
    }

    pub fn is_fully_static(&self) -> bool {
        self.dims.iter().all(|dim| matches!(dim, Dim::Static(_)))
    }

    pub fn static_dims(&self) -> Option<Vec<usize>> {
        self.dims
            .iter()
            .map(|dim| dim.static_usize())
            .collect::<Option<Vec<_>>>()
    }

    pub fn numel_static(&self) -> Option<usize> {
        self.static_dims()
            .map(|dims| dims.iter().product::<usize>())
    }

    /// Resolve all dimensions. `Dynamic` dims are rejected; callers must
    /// specialize them first.
    pub fn bind(&self, bindings: &HashMap<SymbolId, usize>) -> Result<Vec<usize>> {
        self.dims
            .iter()
            .map(|dim| dim.bind(bindings))
            .collect::<Result<Vec<_>>>()
    }

    /// Numpy-style broadcasting on symbolic shapes.
    pub fn broadcast(&self, rhs: &Shape) -> Result<Shape> {
        let rank = self.rank().max(rhs.rank());
        let mut out = Vec::with_capacity(rank);
        for index in 0..rank {
            let lhs_dim = if index < rank - self.rank() {
                Dim::Static(1)
            } else {
                self.dims[index - (rank - self.rank())]
            };
            let rhs_dim = if index < rank - rhs.rank() {
                Dim::Static(1)
            } else {
                rhs.dims[index - (rank - rhs.rank())]
            };
            out.push(
                broadcast_dim(lhs_dim, rhs_dim).ok_or_else(|| Error::InvalidBroadcast {
                    lhs: self.clone(),
                    rhs: rhs.clone(),
                })?,
            );
        }
        Ok(Shape { dims: out })
    }

    /// True when both shapes have the same rank and compatible dimensions.
    pub fn is_compatible(&self, other: &Shape) -> bool {
        self.rank() == other.rank()
            && self
                .dims
                .iter()
                .zip(&other.dims)
                .all(|(a, b)| a.is_compatible(*b))
    }

    pub fn with_dim(mut self, index: usize, dim: Dim) -> Result<Self> {
        if index >= self.dims.len() {
            return Err(Error::InvalidAxis {
                axis: index as i32,
                rank: self.dims.len(),
            });
        }
        self.dims[index] = dim;
        Ok(self)
    }

    pub fn product(&self) -> Option<usize> {
        self.numel_static()
    }
}

fn broadcast_dim(lhs: Dim, rhs: Dim) -> Option<Dim> {
    use Dim::*;
    match (lhs, rhs) {
        (Static(1), other) | (other, Static(1)) => Some(other),
        (Static(a), Static(b)) if a == b => Some(Static(a)),
        (Static(_), Static(_)) => None,
        (Symbol(a), Symbol(b)) if a == b => Some(Symbol(a)),
        (Dynamic, _) | (_, Dynamic) => Some(Dynamic),
        (Symbol(a), _) | (_, Symbol(a)) => Some(Symbol(a)),
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (index, dim) in self.dims.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            match dim {
                Dim::Static(value) => write!(f, "{value}")?,
                Dim::Symbol(symbol) => write!(f, "s{}", symbol.0)?,
                Dim::Dynamic => f.write_str("?")?,
            }
        }
        f.write_str("]")
    }
}

impl From<Vec<Dim>> for Shape {
    fn from(value: Vec<Dim>) -> Self {
        Shape::new(value)
    }
}

impl From<&[usize]> for Shape {
    fn from(value: &[usize]) -> Self {
        Shape::from_static(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_static() {
        let a = Shape::from_static([32, 1, 8]);
        let b = Shape::from_static([4, 8]);
        assert_eq!(a.broadcast(&b).unwrap(), Shape::from_static([32, 4, 8]));
    }

    #[test]
    fn broadcast_rejects_incompatible() {
        let a = Shape::from_static([2, 3]);
        let b = Shape::from_static([4, 3]);
        assert!(a.broadcast(&b).is_err());
    }

    #[test]
    fn symbolic_broadcast_keeps_batch_symbol() {
        let batch = SymbolId(0);
        let a = Shape::new(vec![Dim::Symbol(batch), Dim::Static(784)]);
        let b = Shape::from_static([784]);
        assert_eq!(
            a.broadcast(&b).unwrap(),
            Shape::new(vec![Dim::Symbol(batch), Dim::Static(784)])
        );
    }
}
