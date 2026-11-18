//! Extensible gradient-rule registry.
//!
//! Built-in rules are still implemented in one place (`rules.rs`), but backward
//! generation resolves them through this registry. Custom operations can install
//! a rule before compiling a training graph; a missing rule is reported during
//! that compilation step, never halfway through execution.

use std::collections::HashMap;
use std::sync::Arc;

use oxide_ml_core::{Error, Result, ValueId};
use oxide_ml_graph::{Graph, GraphBuilder, Node, Op};

pub type GradientRuleFn =
    fn(&mut GraphBuilder, &Node, &[Option<ValueId>]) -> Result<Vec<Option<ValueId>>>;

/// One backward rule. Implementations receive the forward node and the
/// upstream gradient for each output, and return one optional gradient per
/// input.
pub trait GradientRule: Send + Sync {
    fn backward(
        &self,
        builder: &mut GraphBuilder,
        node: &Node,
        grad_outputs: &[Option<ValueId>],
    ) -> Result<Vec<Option<ValueId>>>;

    fn name(&self) -> &str;
}

pub struct FunctionRule {
    name: &'static str,
    function: GradientRuleFn,
}

impl FunctionRule {
    pub fn new(name: &'static str, function: GradientRuleFn) -> Self {
        FunctionRule { name, function }
    }
}

impl GradientRule for FunctionRule {
    fn backward(
        &self,
        builder: &mut GraphBuilder,
        node: &Node,
        grad_outputs: &[Option<ValueId>],
    ) -> Result<Vec<Option<ValueId>>> {
        (self.function)(builder, node, grad_outputs)
    }

    fn name(&self) -> &str {
        self.name
    }
}

#[derive(Clone, Default)]
pub struct GradientRegistry {
    rules: HashMap<String, Arc<dyn GradientRule>>,
}

impl std::fmt::Debug for GradientRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientRegistry")
            .field("rules", &self.rules.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl GradientRegistry {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Registry with every built-in rule.
    pub fn builtin() -> Self {
        let mut registry = Self::default();
        for op_name in [
            "add",
            "add_relu",
            "sub",
            "mul",
            "div",
            "neg",
            "matmul",
            "relu",
            "gelu",
            "sigmoid",
            "tanh",
            "softmax",
            "layer_norm",
            "reshape",
            "transpose",
            "broadcast_to",
            "reduce_sum",
            "reduce_mean",
            "conv2d",
            "max_pool2d",
            "avg_pool2d",
            "mse_loss",
            "cross_entropy_loss",
            "constant",
            "constant_scalar",
        ] {
            registry.register(
                op_name,
                Arc::new(FunctionRule::new("oxide-ml.builtin", builtin_rule)),
            );
        }
        registry
    }

    /// Register (or replace) a rule for an operation name.
    pub fn register(&mut self, op_name: &str, rule: Arc<dyn GradientRule>) {
        self.rules.insert(op_name.to_string(), rule);
    }

    pub fn has_rule(&self, op: &Op) -> bool {
        self.rules.contains_key(op.name())
    }

    pub fn resolve(&self, op: &Op) -> Result<Arc<dyn GradientRule>> {
        self.rules.get(op.name()).cloned().ok_or_else(|| {
            Error::Autograd(format!(
                "no gradient rule registered for operation {}",
                op.name()
            ))
        })
    }

    /// Check all operations that can appear in a forward graph before starting
    /// backward construction.
    pub fn ensure_supported(&self, graph: &Graph) -> Result<()> {
        for node in graph.nodes() {
            if !node.op.is_backward() && !self.has_rule(&node.op) {
                return Err(Error::Autograd(format!(
                    "no gradient rule registered for operation {} ({})",
                    node.op.name(),
                    node.id
                )));
            }
        }
        Ok(())
    }
}

fn builtin_rule(
    builder: &mut GraphBuilder,
    node: &Node,
    grad_outputs: &[Option<ValueId>],
) -> Result<Vec<Option<ValueId>>> {
    crate::rules::backward_node(builder, node, grad_outputs)
}
