use oxide_ml_core::{Result, Scalar};
use oxide_ml_graph::Graph;
use oxide_ml_kernel::KernelBackend;

/// Context available to graph passes. The reference backend is used only by
/// constant folding; structural passes ignore it.
pub struct PassContext<'a, T: Scalar> {
    pub backend: &'a dyn KernelBackend<T>,
}

#[derive(Clone, Debug, Default)]
pub struct PassResult {
    pub changed: bool,
    pub notes: Vec<String>,
}

impl PassResult {
    pub fn unchanged() -> Self {
        Self::default()
    }

    pub fn changed(note: impl Into<String>) -> Self {
        PassResult {
            changed: true,
            notes: vec![note.into()],
        }
    }
}

/// An independently runnable optimization pass.
pub trait GraphPass<T: Scalar>: Send + Sync {
    fn name(&self) -> &'static str;
    fn run(&self, graph: &mut Graph, context: &PassContext<'_, T>) -> Result<PassResult>;
}
