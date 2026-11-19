//! Kernel selection result for one compiled graph.
//!
//! The plan records the chosen provider per node and why a node has no
//! provider. It is diagnostic and execution-agnostic: the CPU executor still
//! dispatches by operation, while the plan makes selection explicit and lets
//! backend-specific executors consume the same decisions later.

use oxide_ml_core::NodeId;
use oxide_ml_kernel::ProviderKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelPlanEntry {
    pub node: NodeId,
    pub op: String,
    pub provider: Option<String>,
    pub kind: Option<ProviderKind>,
    pub alternatives: Vec<String>,
    /// Present when no provider was selected.
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelPlan {
    pub entries: Vec<KernelPlanEntry>,
    pub architecture: String,
}

impl KernelPlan {
    pub fn selected(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.provider.is_some())
            .count()
    }

    pub fn unselected(&self) -> usize {
        self.entries.len() - self.selected()
    }

    pub fn entry(&self, node: NodeId) -> Option<&KernelPlanEntry> {
        self.entries.iter().find(|entry| entry.node == node)
    }

    pub fn provider_for(&self, node: NodeId) -> Option<&str> {
        self.entry(node).and_then(|entry| entry.provider.as_deref())
    }
}
