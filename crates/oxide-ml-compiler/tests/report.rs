//! Compiler report and kernel-plan coverage.

use oxide_ml_compiler::{CompileOptions, Compiler};
use oxide_ml_core::{DType, Shape};
use oxide_ml_graph::{GraphBuilder, Op};
use oxide_ml_kernel::KernelCatalog;

fn compiler(options: CompileOptions) -> Compiler<f32> {
    Compiler::new(Box::new(oxide_ml_cpu::CpuBackend::<f32>::new()), options)
}

fn fused_graph() -> oxide_ml_graph::Graph {
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([4, 8]), DType::F32);
    let y = b.input("y", Shape::from_static([4, 8]), DType::F32);
    let product = b.mul(x, y).unwrap();
    let sum = b.add(product, x).unwrap();
    b.finish(&[sum]).unwrap()
}

#[test]
fn report_contains_passes_fusions_kernels_and_memory() {
    let compiled = compiler(CompileOptions::default())
        .compile(fused_graph())
        .unwrap();
    let report = &compiled.report;
    assert_eq!(report.backend, "cpu-reference");
    assert!(report.applied_passes.iter().any(|name| name == "fusion"));
    assert!(!report.applied_passes.is_empty());
    assert!(report.fusion_count >= 1);
    assert_eq!(
        report.kernel_count,
        compiled.graph.nodes().len(),
        "every node should select the reference provider"
    );
    assert!(report.memory_estimate_bytes > 0);
    assert_eq!(
        compiled.kernel_plan.entries.len(),
        compiled.graph.nodes().len()
    );
    assert_eq!(compiled.kernel_plan.selected(), report.kernel_count);
    for node in compiled.graph.nodes() {
        let provider = compiled.kernel_plan.provider_for(node.id).unwrap();
        assert!(provider.starts_with("reference."));
    }
    assert!(report.nodes_removed >= 1);
    assert!(
        compiled
            .graph
            .nodes()
            .iter()
            .any(|node| matches!(node.op, Op::MulAdd))
    );
}

#[test]
fn specialized_catalog_provider_is_selected() {
    let mut catalog = KernelCatalog::builtin_cpu();
    catalog
        .try_register(oxide_ml_kernel::KernelProvider::new(
            "tiled.mul_add",
            oxide_ml_kernel::ProviderKind::Tiled,
            oxide_ml_kernel::KernelCapabilities::generic("mul_add", [DType::F32]),
        ))
        .unwrap();
    let options = CompileOptions {
        kernel_catalog: Some(std::sync::Arc::new(catalog)),
        ..Default::default()
    };
    let compiled = compiler(options).compile(fused_graph()).unwrap();
    let mul_add = compiled
        .graph
        .nodes()
        .iter()
        .find(|node| matches!(node.op, Op::MulAdd))
        .unwrap();
    assert_eq!(
        compiled.kernel_plan.provider_for(mul_add.id),
        Some("tiled.mul_add")
    );
}

#[test]
fn strict_selection_fails_without_a_provider() {
    let options = CompileOptions {
        kernel_catalog: Some(std::sync::Arc::new(KernelCatalog::new())),
        strict_kernel_selection: true,
        ..Default::default()
    };
    let error = compiler(options).compile(fused_graph()).unwrap_err();
    assert!(error.to_string().contains("no kernel provider"));
}

#[test]
fn non_strict_selection_records_diagnostics() {
    let options = CompileOptions {
        kernel_catalog: Some(std::sync::Arc::new(KernelCatalog::new())),
        ..Default::default()
    };
    let compiled = compiler(options).compile(fused_graph()).unwrap();
    assert_eq!(compiled.report.kernel_count, 0);
    assert!(!compiled.report.diagnostics.is_empty());
    assert_eq!(
        compiled.kernel_plan.unselected(),
        compiled.graph.nodes().len()
    );
}

#[test]
fn unsupported_dtype_is_reported_by_selection() {
    // A graph whose dtype no reference provider advertises (BF16).
    let mut b = GraphBuilder::new();
    let x = b.input("x", Shape::from_static([2, 2]), DType::BF16);
    let y = b.input("y", Shape::from_static([2, 2]), DType::BF16);
    let out = b.add(x, y).unwrap();
    let graph = b.finish(&[out]).unwrap();
    let options = CompileOptions {
        memory_planning: false,
        ..Default::default()
    };
    let compiled = compiler(options).compile(graph).unwrap();
    assert_eq!(compiled.report.kernel_count, 0);
    assert!(
        compiled
            .report
            .diagnostics
            .iter()
            .any(|line| line.contains("no kernel provider"))
    );
}
