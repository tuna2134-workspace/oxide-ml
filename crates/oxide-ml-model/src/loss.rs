//! Loss functions as graph-appending objects.

use oxide_ml_core::{Result, ValueId};
pub use oxide_ml_graph::Reduction;

use crate::context::ModelContext;

/// A loss that appends its computation to a [`ModelContext`].
///
/// Losses are not modules: they take a prediction and a target and produce a
/// scalar value. The trainer owns one and applies it after the model forward.
pub trait Loss: Send + Sync {
    fn name(&self) -> &'static str;

    fn forward(
        &self,
        cx: &mut ModelContext,
        prediction: ValueId,
        target: ValueId,
    ) -> Result<ValueId>;

    fn reduction(&self) -> Reduction {
        Reduction::Mean
    }
}

impl std::fmt::Debug for dyn Loss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Loss({})", self.name())
    }
}
