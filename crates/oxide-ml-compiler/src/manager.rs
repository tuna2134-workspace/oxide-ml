use std::collections::HashSet;
use std::time::Instant;

use oxide_ml_core::{Result, Scalar};
use oxide_ml_graph::{Graph, GraphVerifier, VerificationReport};
use oxide_ml_kernel::KernelBackend;

use crate::constant_folding::ConstantFolding;
use crate::cse::CommonSubexpressionElimination;
use crate::dce::DeadCodeElimination;
use crate::fusion::OperatorFusion;
use crate::pass::{GraphPass, PassContext, PassResult};

#[derive(Clone, Debug)]
pub struct PassReport {
    pub name: &'static str,
    pub changed: bool,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub duration_micros: u128,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct CompileReport {
    pub before: VerificationReport,
    pub after: VerificationReport,
    pub passes: Vec<PassReport>,
    pub graph_before: Option<String>,
    pub graph_after: Option<String>,
}

impl CompileReport {
    pub fn changed(&self) -> bool {
        self.passes.iter().any(|pass| pass.changed)
    }

    pub fn total_nodes_removed(&self) -> usize {
        self.before.nodes.saturating_sub(self.after.nodes)
    }
}

pub struct PassManager<T: Scalar> {
    passes: Vec<Box<dyn GraphPass<T>>>,
    enabled: HashSet<String>,
    dump: bool,
}

impl<T: Scalar> PassManager<T> {
    pub fn new() -> Self {
        PassManager {
            passes: Vec::new(),
            enabled: HashSet::new(),
            dump: false,
        }
    }

    /// Constant folding -> CSE -> fusion -> DCE.
    pub fn default_passes() -> Self {
        let mut manager = Self::new();
        manager.register(Box::new(ConstantFolding));
        manager.register(Box::new(CommonSubexpressionElimination));
        manager.register(Box::new(OperatorFusion));
        manager.register(Box::new(DeadCodeElimination));
        manager
    }

    pub fn register(&mut self, pass: Box<dyn GraphPass<T>>) {
        self.enabled.insert(pass.name().to_string());
        self.passes.push(pass);
    }

    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> bool {
        if enabled {
            self.enabled.insert(name.to_string());
        } else {
            self.enabled.remove(name);
        }
        self.passes.iter().any(|pass| pass.name() == name)
    }

    pub fn is_enabled(&self, name: &str) -> bool {
        self.enabled.contains(name)
    }

    pub fn set_dump(&mut self, dump: bool) {
        self.dump = dump;
    }

    pub fn pass_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|pass| pass.name()).collect()
    }

    pub fn run(&self, graph: &mut Graph, backend: &dyn KernelBackend<T>) -> Result<CompileReport> {
        let before = GraphVerifier::verify(graph)?;
        let graph_before = self.dump.then(|| graph.dump());
        let context = PassContext { backend };
        let mut reports = Vec::with_capacity(self.passes.len());
        for pass in &self.passes {
            if !self.enabled.contains(pass.name()) {
                continue;
            }
            let nodes_before = graph.nodes().len();
            let start = Instant::now();
            let result: PassResult = pass.run(graph, &context)?;
            let duration_micros = start.elapsed().as_micros();
            GraphVerifier::verify(graph)?;
            reports.push(PassReport {
                name: pass.name(),
                changed: result.changed,
                nodes_before,
                nodes_after: graph.nodes().len(),
                duration_micros,
                notes: result.notes,
            });
        }
        let after = GraphVerifier::verify(graph)?;
        let graph_after = self.dump.then(|| graph.dump());
        Ok(CompileReport {
            before,
            after,
            passes: reports,
            graph_before,
            graph_after,
        })
    }
}

impl<T: Scalar> std::fmt::Debug for PassManager<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassManager")
            .field("passes", &self.pass_names())
            .field("enabled", &self.enabled)
            .field("dump", &self.dump)
            .finish()
    }
}

impl<T: Scalar> Default for PassManager<T> {
    fn default() -> Self {
        Self::new()
    }
}
