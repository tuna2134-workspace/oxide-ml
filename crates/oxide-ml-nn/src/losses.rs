//! Loss functions for the model API.

use oxide_ml_core::{Error, Shape, ValueId};
use oxide_ml_graph::Reduction;
use oxide_ml_model::{Loss, ModelContext};

fn mean_all(cx: &mut ModelContext, value: ValueId) -> oxide_ml_core::Result<ValueId> {
    let shape = cx.value_shape(value)?;
    let axes: Vec<usize> = (0..shape.rank()).collect();
    if axes.is_empty() {
        return Ok(value);
    }
    cx.builder().reduce_mean(value, axes, false)
}

/// Mean squared error.
#[derive(Clone, Copy, Debug, Default)]
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        MSELoss
    }
}

impl Loss for MSELoss {
    fn name(&self) -> &'static str {
        "MSELoss"
    }

    fn forward(
        &self,
        cx: &mut ModelContext,
        prediction: ValueId,
        target: ValueId,
    ) -> oxide_ml_core::Result<ValueId> {
        cx.builder().mse_loss(prediction, target, Reduction::Mean)
    }
}

/// Softmax cross entropy over class indices.
#[derive(Clone, Copy, Debug, Default)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }
}

impl Loss for CrossEntropyLoss {
    fn name(&self) -> &'static str {
        "CrossEntropyLoss"
    }

    fn forward(
        &self,
        cx: &mut ModelContext,
        logits: ValueId,
        targets: ValueId,
    ) -> oxide_ml_core::Result<ValueId> {
        let logits_shape = cx.value_shape(logits)?;
        if logits_shape.rank() != 2 {
            return Err(Error::RankMismatch {
                context: "CrossEntropyLoss logits".into(),
                expected: 2,
                got: logits_shape.rank(),
            });
        }
        cx.builder()
            .cross_entropy_loss(logits, targets, Reduction::Mean)
    }
}

/// Binary cross entropy with logits, evaluated in a numerically stable form.
#[derive(Clone, Copy, Debug, Default)]
pub struct BCEWithLogitsLoss;

impl BCEWithLogitsLoss {
    pub fn new() -> Self {
        BCEWithLogitsLoss
    }
}

impl Loss for BCEWithLogitsLoss {
    fn name(&self) -> &'static str {
        "BCEWithLogitsLoss"
    }

    fn forward(
        &self,
        cx: &mut ModelContext,
        logits: ValueId,
        targets: ValueId,
    ) -> oxide_ml_core::Result<ValueId> {
        let logits_shape = cx.value_shape(logits)?;
        let target_shape = cx.value_shape(targets)?;
        if logits_shape != target_shape {
            return Err(Error::shape_mismatch(
                "BCEWithLogitsLoss",
                logits_shape,
                target_shape,
            ));
        }
        // max(x, 0) - x*y + log(1 + exp(-|x|))
        let absolute = cx.builder().abs(logits)?;
        let negative = cx.builder().neg(absolute)?;
        let exponential = cx.builder().exp(negative)?;
        let one = cx.constant_scalar(1.0);
        let one_plus = cx.builder().add(exponential, one)?;
        let log_term = cx.builder().log(one_plus)?;
        let positive = cx.builder().relu(logits)?;
        let product = cx.builder().mul(logits, targets)?;
        let first = cx.builder().sub(positive, product)?;
        let loss = cx.builder().add(first, log_term)?;
        mean_all(cx, loss)
    }
}

/// Smooth L1 loss.
#[derive(Clone, Copy, Debug)]
pub struct HuberLoss {
    pub delta: f32,
}

impl Default for HuberLoss {
    fn default() -> Self {
        HuberLoss { delta: 1.0 }
    }
}

impl HuberLoss {
    pub fn new(delta: f32) -> oxide_ml_core::Result<Self> {
        if !(delta.is_finite() && delta > 0.0) {
            return Err(Error::invalid_shape("HuberLoss", "delta must be positive"));
        }
        Ok(HuberLoss { delta })
    }
}

impl Loss for HuberLoss {
    fn name(&self) -> &'static str {
        "HuberLoss"
    }

    fn forward(
        &self,
        cx: &mut ModelContext,
        prediction: ValueId,
        target: ValueId,
    ) -> oxide_ml_core::Result<ValueId> {
        let difference = cx.builder().sub(prediction, target)?;
        let absolute = cx.builder().abs(difference)?;
        let delta = cx.constant_scalar(self.delta);
        let excess = cx.builder().sub(absolute, delta)?;
        let excess = cx.builder().relu(excess)?;
        let square = cx.builder().mul(difference, difference)?;
        let half = cx.constant_scalar(0.5);
        let square = cx.builder().mul(square, half)?;
        let excess_square = cx.builder().mul(excess, excess)?;
        let correction = cx.builder().mul(excess_square, half)?;
        let loss = cx.builder().sub(square, correction)?;
        mean_all(cx, loss)
    }
}

/// Negative log likelihood over one-hot targets.
///
/// For class indices use [`CrossEntropyLoss`], which performs the same
/// computation from integer labels.
#[derive(Clone, Copy, Debug, Default)]
pub struct NLLLoss;

impl NLLLoss {
    pub fn new() -> Self {
        NLLLoss
    }
}

impl Loss for NLLLoss {
    fn name(&self) -> &'static str {
        "NLLLoss"
    }

    fn forward(
        &self,
        cx: &mut ModelContext,
        logits: ValueId,
        targets: ValueId,
    ) -> oxide_ml_core::Result<ValueId> {
        let logits_shape = cx.value_shape(logits)?;
        let target_shape = cx.value_shape(targets)?;
        if logits_shape.rank() != 2 {
            return Err(Error::RankMismatch {
                context: "NLLLoss logits".into(),
                expected: 2,
                got: logits_shape.rank(),
            });
        }
        if target_shape != logits_shape {
            return Err(Error::shape_mismatch(
                "NLLLoss one-hot targets",
                logits_shape,
                target_shape,
            ));
        }
        let log_probs = cx.builder().log_softmax(logits, 1)?;
        let weighted = cx.builder().mul(log_probs, targets)?;
        let summed = cx.builder().reduce_sum(weighted, vec![1], true)?;
        let negative = cx.builder().neg(summed)?;
        mean_all(cx, negative)
    }
}

/// Validate that two operands have the same shape (helper for custom losses).
pub fn require_same_shape(
    cx: &ModelContext,
    lhs: ValueId,
    rhs: ValueId,
    context: &str,
) -> oxide_ml_core::Result<()> {
    let lhs = cx.value_shape(lhs)?;
    let rhs = cx.value_shape(rhs)?;
    if lhs != rhs {
        return Err(Error::shape_mismatch(context, Shape::clone(&lhs), rhs));
    }
    Ok(())
}
