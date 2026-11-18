use std::fmt;

pub type Result<T> = std::result::Result<T, CudaError>;

#[derive(Debug)]
pub enum CudaError {
    Driver(cuda_core::DriverError),
    Unsupported(String),
    Shape(String),
    InvalidPlan(String),
    ArenaBusy(String),
    Event(String),
    EventFailure(oxide_ml_event_safety::EventError),
    Internal(String),
    Module(String),
    Io(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::Driver(error) => write!(f, "CUDA driver error: {error}"),
            CudaError::Unsupported(message) => write!(f, "CUDA unsupported: {message}"),
            CudaError::Shape(message) => write!(f, "CUDA shape error: {message}"),
            CudaError::InvalidPlan(message) => write!(f, "CUDA memory plan error: {message}"),
            CudaError::ArenaBusy(message) => write!(f, "CUDA arena busy: {message}"),
            CudaError::Event(message) => write!(f, "CUDA event error: {message}"),
            CudaError::EventFailure(error) => write!(f, "CUDA event failure: {error}"),
            CudaError::Internal(message) => write!(f, "CUDA internal error: {message}"),
            CudaError::Module(message) => write!(f, "CUDA module error: {message}"),
            CudaError::Io(message) => write!(f, "CUDA I/O error: {message}"),
        }
    }
}

impl std::error::Error for CudaError {}

impl From<oxide_ml_event_safety::EventError> for CudaError {
    fn from(value: oxide_ml_event_safety::EventError) -> Self {
        CudaError::EventFailure(value)
    }
}

impl From<cuda_core::DriverError> for CudaError {
    fn from(value: cuda_core::DriverError) -> Self {
        CudaError::Driver(value)
    }
}

impl From<cuda_host::EmbeddedModuleError> for CudaError {
    fn from(value: cuda_host::EmbeddedModuleError) -> Self {
        CudaError::Module(value.to_string())
    }
}

impl From<oxide_ml_core::Error> for CudaError {
    fn from(value: oxide_ml_core::Error) -> Self {
        CudaError::Internal(value.to_string())
    }
}
