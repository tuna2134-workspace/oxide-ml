use std::sync::Arc;
use std::time::Instant;

use oxide_ml_core::{DType, MemoryPlan, Result, Scalar};
use oxide_ml_graph::{Graph, GraphVerifier, Op};
use oxide_ml_kernel::{KernelBackend, KernelCatalog, KernelRequest};

use crate::kernel_plan::{KernelPlan, KernelPlanEntry};
use crate::manager::{CompileReport, PassManager};
use crate::memory_planner::MemoryPlanner;

#[derive(Clone, Debug)]
pub struct PassToggles {
    pub constant_folding: bool,
    pub cse: bool,
    pub fusion: bool,
    pub dce: bool,
}

impl Default for PassToggles {
    fn default() -> Self {
        PassToggles {
            constant_folding: true,
            cse: true,
            fusion: true,
            dce: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompileOptions {
    pub optimize: bool,
    pub memory_planning: bool,
    pub dump_graphs: bool,
    pub passes: PassToggles,
    /// Candidate kernels. `None` uses the generic reference catalog.
    pub kernel_catalog: Option<Arc<KernelCatalog>>,
    /// Target architecture used for provider filtering.
    pub architecture: String,
    /// Fail compilation when a node has no compatible provider.
    pub strict_kernel_selection: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            optimize: true,
            memory_planning: true,
            dump_graphs: false,
            passes: PassToggles::default(),
            kernel_catalog: None,
            architecture: "portable".to_string(),
            strict_kernel_selection: false,
        }
    }
}

#[derive(Debug)]
pub struct CompiledGraph {
    pub graph: Graph,
    pub report: CompileReport,
    pub memory_plan: Option<MemoryPlan>,
    pub kernel_plan: KernelPlan,
}

pub struct Compiler<T: Scalar> {
    backend: Box<dyn KernelBackend<T>>,
    options: CompileOptions,
}

impl<T: Scalar> Compiler<T> {
    pub fn new(backend: Box<dyn KernelBackend<T>>, options: CompileOptions) -> Self {
        Compiler { backend, options }
    }

    pub fn compile(&self, mut graph: Graph) -> Result<CompiledGraph> {
        let started = Instant::now();
        GraphVerifier::verify(&graph)?;
        let mut manager = PassManager::default_passes();
        manager.set_dump(self.options.dump_graphs);
        if !self.options.optimize {
            for name in manager.pass_names() {
                manager.set_enabled(name, false);
            }
        } else {
            manager.set_enabled("constant-folding", self.options.passes.constant_folding);
            manager.set_enabled("cse", self.options.passes.cse);
            manager.set_enabled("fusion", self.options.passes.fusion);
            manager.set_enabled("dce", self.options.passes.dce);
        }
        let report = manager.run(&mut graph, self.backend.as_ref())?;
        GraphVerifier::verify(&graph)?;
        let memory_plan = if self.options.memory_planning {
            Some(MemoryPlanner::default().plan(&graph)?)
        } else {
            None
        };
        let catalog = self
            .options
            .kernel_catalog
            .clone()
            .unwrap_or_else(|| Arc::new(KernelCatalog::builtin_cpu()));
        let mut kernel_plan = KernelPlan {
            entries: Vec::with_capacity(graph.nodes().len()),
            architecture: self.options.architecture.clone(),
        };
        let mut diagnostics = Vec::new();
        for node in graph.nodes() {
            let dtypes: Vec<DType> = if node.inputs.is_empty() {
                node.outputs
                    .iter()
                    .filter_map(|output| graph.value(*output).ok().map(|value| value.dtype))
                    .collect()
            } else {
                node.inputs
                    .iter()
                    .filter_map(|input| graph.value(*input).ok().map(|value| value.dtype))
                    .collect()
            };
            let shapes = node
                .inputs
                .iter()
                .filter_map(|input| graph.value(*input).ok())
                .map(|value| {
                    value
                        .shape
                        .dims()
                        .iter()
                        .map(|dim| dim.static_usize().unwrap_or(0))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let request = KernelRequest {
                op: node.op.name().to_string(),
                dtypes,
                shapes,
                layout: oxide_ml_kernel::MemoryLayout::Contiguous,
                architecture: self.options.architecture.clone(),
                abi_version: oxide_ml_kernel::KERNEL_ABI_VERSION,
                workspace_budget: None,
                allow_custom: true,
            };
            match catalog.select(&request) {
                Ok(selection) => kernel_plan.entries.push(KernelPlanEntry {
                    node: node.id,
                    op: node.op.name().to_string(),
                    provider: Some(selection.provider),
                    kind: Some(selection.kind),
                    alternatives: selection.alternatives,
                    reason: None,
                }),
                Err(error) => {
                    let reason = error.to_string();
                    diagnostics.push(format!("{}: {reason}", node.id));
                    if self.options.strict_kernel_selection {
                        return Err(oxide_ml_core::Error::unsupported(format!(
                            "{}: {reason}",
                            node.id
                        )));
                    }
                    kernel_plan.entries.push(KernelPlanEntry {
                        node: node.id,
                        op: node.op.name().to_string(),
                        provider: None,
                        kind: None,
                        alternatives: Vec::new(),
                        reason: Some(reason),
                    });
                }
            }
        }
        let fusion_count = graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, Op::AddRelu | Op::MulAdd | Op::MatMulBias))
            .count();
        let kernel_count = kernel_plan.selected();
        let applied_passes = report
            .passes
            .iter()
            .filter(|pass| pass.changed)
            .map(|pass| pass.name.to_string())
            .collect::<Vec<_>>();
        let rejected_passes = report
            .passes
            .iter()
            .filter(|pass| !pass.changed)
            .map(|pass| pass.name.to_string())
            .collect::<Vec<_>>();
        let mut report = report;
        report.backend = self.backend.name().to_string();
        report.compilation_time_micros = started.elapsed().as_micros();
        report.fusion_count = fusion_count;
        report.kernel_count = kernel_count;
        report.memory_estimate_bytes = memory_plan
            .as_ref()
            .map(MemoryPlan::peak_bytes)
            .unwrap_or(0);
        report.applied_passes = applied_passes;
        report.rejected_passes = rejected_passes;
        report.diagnostics = diagnostics;
        report.nodes_removed = report.before.nodes.saturating_sub(report.after.nodes);
        report.kernel_plan = kernel_plan.clone();
        Ok(CompiledGraph {
            graph,
            report,
            memory_plan,
            kernel_plan,
        })
    }
}
