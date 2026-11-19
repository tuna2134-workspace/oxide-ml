//! Autograd adapter for user-registered kernels.

use std::collections::HashMap;
use std::sync::Arc;

use oxide_ml_core::{Error, Result, Scalar, ValueId};
use oxide_ml_graph::{GraphBuilder, Node, Op, OpAttrs};
use oxide_ml_kernel::{CustomBackward, CustomKernelRegistry};

use crate::registry::GradientRule;

/// Dispatches `Op::Custom` backward passes by kernel name.
pub struct CustomKernelRule {
    rules: HashMap<String, CustomBackward>,
}

impl CustomKernelRule {
    pub fn new(rules: HashMap<String, CustomBackward>) -> Self {
        CustomKernelRule { rules }
    }

    pub fn from_registry<T: Scalar>(registry: &CustomKernelRegistry<T>) -> Self {
        CustomKernelRule {
            rules: registry.backward_rules(),
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.rules.contains_key(name)
    }
}

impl GradientRule for CustomKernelRule {
    fn backward(
        &self,
        builder: &mut GraphBuilder,
        node: &Node,
        grad_outputs: &[Option<ValueId>],
    ) -> Result<Vec<Option<ValueId>>> {
        if !matches!(node.op, Op::Custom) {
            return Err(Error::Autograd(format!(
                "CustomKernelRule received non-custom op {}",
                node.op.name()
            )));
        }
        let name = match &node.attrs {
            OpAttrs::Custom { name, .. } => name,
            _ => {
                return Err(Error::Autograd(
                    "Custom op is missing OpAttrs::Custom".to_string(),
                ));
            }
        };
        let rule = self.rules.get(name).ok_or_else(|| {
            Error::Autograd(format!(
                "custom kernel {name:?} has no backward rule; \
                 register one with CustomKernel::with_backward"
            ))
        })?;
        rule(builder, node, grad_outputs)
    }

    fn name(&self) -> &str {
        "custom"
    }
}

impl std::fmt::Debug for CustomKernelRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomKernelRule")
            .field("kernels", &self.rules.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Shared map type used when constructing a rule directly.
pub type CustomBackwardMap = HashMap<String, CustomBackward>;

/// Convenience alias so callers can write `Arc<CustomKernelRule>`.
pub type SharedCustomKernelRule = Arc<CustomKernelRule>;
