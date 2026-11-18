use crate::{DType, ParameterId, Shape, ValueId};

pub type Result<T> = std::result::Result<T, Error>;

/// Unified error type for the reference framework.
///
/// Backends may add context through `Backend` rather than growing this enum for
/// every device API difference.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    ShapeMismatch {
        context: String,
        expected: Shape,
        got: Shape,
    },
    RankMismatch {
        context: String,
        expected: usize,
        got: usize,
    },
    DTypeMismatch {
        context: String,
        expected: DType,
        got: DType,
    },
    InvalidBroadcast {
        lhs: Shape,
        rhs: Shape,
    },
    InvalidAxis {
        axis: i32,
        rank: usize,
    },
    InvalidShape {
        context: String,
        detail: String,
    },
    UnknownValue(ValueId),
    UnknownParameter(ParameterId),
    InvalidGraph(String),
    Unsupported(String),
    Execution(String),
    Autograd(String),
    Serialization(String),
    Io(String),
}

impl Error {
    pub fn shape_mismatch(context: impl Into<String>, expected: Shape, got: Shape) -> Self {
        Error::ShapeMismatch {
            context: context.into(),
            expected,
            got,
        }
    }

    pub fn invalid_shape(context: impl Into<String>, detail: impl Into<String>) -> Self {
        Error::InvalidShape {
            context: context.into(),
            detail: detail.into(),
        }
    }

    pub fn invalid_graph(detail: impl Into<String>) -> Self {
        Error::InvalidGraph(detail.into())
    }

    pub fn unsupported(detail: impl Into<String>) -> Self {
        Error::Unsupported(detail.into())
    }

    pub fn execution(detail: impl Into<String>) -> Self {
        Error::Execution(detail.into())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::ShapeMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected shape {expected}, got {got}"),
            Error::RankMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected rank {expected}, got rank {got}"),
            Error::DTypeMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected dtype {expected}, got {got}"),
            Error::InvalidBroadcast { lhs, rhs } => {
                write!(f, "cannot broadcast shapes {lhs} and {rhs}")
            }
            Error::InvalidAxis { axis, rank } => {
                write!(f, "axis {axis} is out of range for rank {rank}")
            }
            Error::InvalidShape { context, detail } => {
                write!(f, "invalid shape for {context}: {detail}")
            }
            Error::UnknownValue(id) => write!(f, "unknown graph value {id}"),
            Error::UnknownParameter(id) => write!(f, "unknown parameter {id}"),
            Error::InvalidGraph(detail) => write!(f, "invalid graph: {detail}"),
            Error::Unsupported(detail) => write!(f, "unsupported operation: {detail}"),
            Error::Execution(detail) => write!(f, "execution error: {detail}"),
            Error::Autograd(detail) => write!(f, "autograd error: {detail}"),
            Error::Serialization(detail) => write!(f, "serialization error: {detail}"),
            Error::Io(detail) => write!(f, "io error: {detail}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(value: serde_json::Error) -> Self {
        Error::Serialization(value.to_string())
    }
}
