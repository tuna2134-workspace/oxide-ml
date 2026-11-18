use oxide_ml_core::{MemoryPlan, Result, Scalar};
use oxide_ml_graph::{Graph, GraphVerifier};
use oxide_ml_kernel::KernelBackend;

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
}

impl Default for CompileOptions {
    fn default() -> Self {
        CompileOptions {
            optimize: true,
            memory_planning: true,
            dump_graphs: false,
            passes: PassToggles::default(),
        }
    }
}

#[derive(Debug)]
pub struct CompiledGraph {
    pub graph: Graph,
    pub report: CompileReport,
    pub memory_plan: Option<MemoryPlan>,
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
        Ok(CompiledGraph {
            graph,
            report,
            memory_plan,
        })
    }
}
