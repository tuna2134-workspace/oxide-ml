//! Phase 5 compiler benchmarks. Run with `cargo bench -p oxide-ml`.
//!
//! This uses a tiny std-only harness so the project does not gain a benchmark
//! dependency. Timings are averaged over a fixed number of iterations.

use std::time::Instant;

use oxide_ml::prelude::*;
use oxide_ml_cpu::CpuBackend;

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    optimize: bool,
    memory_planning: bool,
    only_pass: Option<&'static str>,
}

struct ResultRow {
    scenario: String,
    compile_us: f64,
    nodes: usize,
    non_constant_nodes: usize,
    exec_us: f64,
    peak_bytes: usize,
    allocation_count: usize,
}

fn build_graph(depth: usize, width: usize, batch: usize) -> (Graph, ExecutionInputs<f32>) {
    let mut builder = GraphBuilder::new();
    let x = builder.input("x", Shape::from_static([batch, width]), DType::F32);
    let target = builder.input("y", Shape::from_static([batch, 1]), DType::F32);
    let constant_a = builder
        .constant_tensor(
            ConstantTensor::f32(
                Shape::from_static([batch, width]),
                (0..batch * width)
                    .map(|index| (index % 7) as f32 * 0.01)
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let constant_b = builder
        .constant_tensor(
            ConstantTensor::f32(
                Shape::from_static([batch, width]),
                (0..batch * width)
                    .map(|index| (index % 5) as f32 * 0.02)
                    .collect(),
            )
            .unwrap(),
        )
        .unwrap();
    let constant_sum = builder.add(constant_a, constant_b).unwrap();
    let mut value = builder.add(x, constant_sum).unwrap();
    for layer in 0..depth {
        let weight = builder
            .try_parameter(
                format!("layer{layer}.weight"),
                Shape::from_static([width, width]),
                DType::F32,
                true,
            )
            .unwrap();
        let bias = builder
            .try_parameter(
                format!("layer{layer}.bias"),
                Shape::from_static([width]),
                DType::F32,
                true,
            )
            .unwrap();
        let product = builder.matmul(value, weight).unwrap();
        let sum = builder.add(product, bias).unwrap();
        value = builder.relu(sum).unwrap();
    }
    let final_weight = builder
        .try_parameter(
            "head.weight",
            Shape::from_static([width, 1]),
            DType::F32,
            true,
        )
        .unwrap();
    let final_bias = builder
        .try_parameter("head.bias", Shape::from_static([1]), DType::F32, true)
        .unwrap();
    let product = builder.matmul(value, final_weight).unwrap();
    let prediction = builder.add(product, final_bias).unwrap();
    let loss = builder
        .mse_loss(prediction, target, Reduction::Mean)
        .unwrap();
    let graph = builder.finish(&[loss]).unwrap();
    let mut inputs = ExecutionInputs::<f32>::new();
    inputs.insert(
        graph.inputs()[0],
        DenseTensor::from_vec(
            [batch, width],
            (0..batch * width)
                .map(|index| ((index % 13) as f32 - 6.0) * 0.05)
                .collect(),
        )
        .unwrap(),
    );
    inputs.insert(
        graph.inputs()[1],
        DenseTensor::from_vec([batch, 1], vec![0.5; batch]).unwrap(),
    );
    (graph, inputs)
}

fn compile_scenario(graph: &Graph, scenario: Scenario) -> (Graph, Option<MemoryPlan>, f64) {
    let backend = Box::new(CpuBackend::<f32>::new());
    let start = Instant::now();
    if let Some(only) = scenario.only_pass {
        let mut graph = graph.clone();
        let mut manager = PassManager::<f32>::default_passes();
        for name in manager.pass_names() {
            manager.set_enabled(name, name == only);
        }
        manager.run(&mut graph, backend.as_ref()).unwrap();
        let plan = MemoryPlanner::default().plan(&graph).unwrap();
        (graph, Some(plan), start.elapsed().as_secs_f64() * 1e6)
    } else {
        let options = CompileOptions {
            optimize: scenario.optimize,
            memory_planning: scenario.memory_planning,
            dump_graphs: false,
            ..CompileOptions::default()
        };
        let compiler = Compiler::<f32>::new(backend, options);
        let compiled = compiler.compile(graph.clone()).unwrap();
        (
            compiled.graph,
            compiled.memory_plan,
            start.elapsed().as_secs_f64() * 1e6,
        )
    }
}

fn estimate_value_bytes(graph: &Graph) -> usize {
    graph
        .values()
        .iter()
        .filter(|value| !matches!(value.kind, ValueKind::Input | ValueKind::Parameter(_)))
        .filter_map(|value| {
            value
                .shape
                .numel_static()
                .map(|count| count * value.dtype.size_in_bytes())
        })
        .sum()
}

fn run(
    graph: &Graph,
    inputs: &ExecutionInputs<f32>,
    plan: Option<&MemoryPlan>,
) -> (f64, usize, usize) {
    let params = init_parameters::<f32>(graph, 0x51DE).unwrap();
    let executor = Executor::<f32>::new(graph.clone()).unwrap();
    let iterations = 20;
    let start = Instant::now();
    if let Some(plan) = plan {
        for _ in 0..iterations {
            executor.run_planned(&params, inputs, plan).unwrap();
        }
    } else {
        for _ in 0..iterations {
            executor.run(&params, inputs).unwrap();
        }
    }
    let exec_us = start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
    let (peak, allocs) = match plan {
        Some(plan) => (plan.peak_bytes(), plan.allocation_count()),
        None => (estimate_value_bytes(graph), graph.nodes().len()),
    };
    (exec_us, peak, allocs)
}

fn main() {
    let scenarios = [
        Scenario {
            name: "no-opt/no-plan",
            optimize: false,
            memory_planning: false,
            only_pass: None,
        },
        Scenario {
            name: "full-opt/plan",
            optimize: true,
            memory_planning: true,
            only_pass: None,
        },
        Scenario {
            name: "constant-fold",
            optimize: true,
            memory_planning: true,
            only_pass: Some("constant-folding"),
        },
        Scenario {
            name: "fusion",
            optimize: true,
            memory_planning: true,
            only_pass: Some("fusion"),
        },
    ];
    for (label, depth, width, batch) in [
        (
            "small (3 layers, width 16, batch 8)",
            3usize,
            16usize,
            8usize,
        ),
        ("large (16 layers, width 64, batch 32)", 16, 64, 32),
    ] {
        println!("\n== {label} ==");
        let (base_graph, inputs) = build_graph(depth, width, batch);
        println!(
            "baseline nodes={} estimated_value_bytes={}",
            base_graph.nodes().len(),
            estimate_value_bytes(&base_graph)
        );
        println!(
            "{:<20} {:>10} {:>10} {:>12} {:>10} {:>8} {:>7}",
            "scenario", "compile_us", "nodes", "non_const", "exec_us", "peak_B", "allocs"
        );
        let mut rows = Vec::new();
        for scenario in scenarios {
            let (graph, plan, compile_us) = compile_scenario(&base_graph, scenario);
            let (exec_us, peak_bytes, allocation_count) = run(&graph, &inputs, plan.as_ref());
            let non_constant_nodes = graph
                .nodes()
                .iter()
                .filter(|node| !matches!(node.op, Op::Constant | Op::ConstantScalar))
                .count();
            rows.push(ResultRow {
                scenario: scenario.name.to_string(),
                compile_us,
                nodes: graph.nodes().len(),
                non_constant_nodes,
                exec_us,
                peak_bytes,
                allocation_count,
            });
        }
        for row in rows {
            println!(
                "{:<20} {:>10.1} {:>10} {:>12} {:>10.2} {:>8} {:>7}",
                row.scenario,
                row.compile_us,
                row.nodes,
                row.non_constant_nodes,
                row.exec_us,
                row.peak_bytes,
                row.allocation_count
            );
        }
    }
}
