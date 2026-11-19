use oxide_ml_core::DType;
use oxide_ml_kernel::{
    AutogradSupport, KernelCapabilities, KernelCatalog, KernelProvider, KernelRequest,
    MemoryLayout, ProviderKind,
};

fn request(op: &str, dtype: DType) -> KernelRequest {
    KernelRequest::new(op)
        .with_dtypes([dtype])
        .with_shapes(vec![vec![4, 8]])
}

#[test]
fn builtin_catalog_selects_generic_reference() {
    let catalog = KernelCatalog::builtin_cpu();
    let selection = catalog.select(&request("add", DType::F32)).unwrap();
    assert_eq!(selection.kind, ProviderKind::Generic);
    assert_eq!(selection.provider, "reference.add");
}

#[test]
fn rejects_unsupported_dtype_and_missing_op() {
    let catalog = KernelCatalog::builtin_cpu();
    assert!(catalog.select(&request("add", DType::BF16)).is_err());
    assert!(catalog.select(&request("not_an_op", DType::F32)).is_err());
}

#[test]
fn prefers_specialized_and_custom_providers() {
    let mut catalog = KernelCatalog::builtin_cpu();
    catalog
        .try_register(KernelProvider::new(
            "tiled.add",
            ProviderKind::Tiled,
            KernelCapabilities::generic("add", [DType::F32]),
        ))
        .unwrap();
    let selection = catalog.select(&request("add", DType::F32)).unwrap();
    assert_eq!(selection.provider, "tiled.add");
    assert!(
        selection
            .alternatives
            .contains(&"reference.add".to_string())
    );

    catalog
        .try_register(KernelProvider::new(
            "my.custom_add",
            ProviderKind::Custom,
            KernelCapabilities::generic("add", [DType::F32]),
        ))
        .unwrap();
    let selection = catalog.select(&request("add", DType::F32)).unwrap();
    assert_eq!(selection.provider, "my.custom_add");

    let no_custom = request("add", DType::F32);
    let mut no_custom = no_custom;
    no_custom.allow_custom = false;
    let selection = catalog.select(&no_custom).unwrap();
    assert_eq!(selection.provider, "tiled.add");
}

#[test]
fn filters_by_rank_architecture_and_workspace() {
    let mut catalog = KernelCatalog::new();
    catalog
        .try_register(KernelProvider::new(
            "sm75.only",
            ProviderKind::Specialized,
            KernelCapabilities::generic("matmul", [DType::F32])
                .with_architectures(vec!["sm_75".to_string()])
                .with_rank(2, 2)
                .with_workspace(1024),
        ))
        .unwrap();
    let mut req = KernelRequest::new("matmul")
        .with_dtypes([DType::F32])
        .with_shapes(vec![vec![4, 8]])
        .with_architecture("sm_75");
    assert!(catalog.select(&req).is_ok());
    req.architecture = "sm_90".to_string();
    assert!(catalog.select(&req).is_err());
    req.architecture = "sm_75".to_string();
    req.shapes = vec![vec![2, 3, 4]];
    assert!(catalog.select(&req).is_err());
    req.shapes = vec![vec![4, 8]];
    req.workspace_budget = Some(512);
    assert!(catalog.select(&req).is_err());
    req.workspace_budget = Some(2048);
    assert!(catalog.select(&req).is_ok());
}

#[test]
fn selection_is_deterministic() {
    let mut catalog = KernelCatalog::new();
    for name in ["b.provider", "a.provider", "c.provider"] {
        catalog
            .try_register(KernelProvider::new(
                name,
                ProviderKind::Generic,
                KernelCapabilities::generic("add", [DType::F32]),
            ))
            .unwrap();
    }
    let first = catalog.select(&request("add", DType::F32)).unwrap();
    for _ in 0..10 {
        assert_eq!(first, catalog.select(&request("add", DType::F32)).unwrap());
    }
    assert_eq!(first.provider, "a.provider");
}

#[test]
fn duplicate_provider_names_are_rejected() {
    let mut catalog = KernelCatalog::new();
    let provider = || {
        KernelProvider::new(
            "same",
            ProviderKind::Generic,
            KernelCapabilities::generic("add", [DType::F32]),
        )
    };
    catalog.try_register(provider()).unwrap();
    assert!(catalog.try_register(provider()).is_err());
    let mut capabilities = KernelCapabilities::generic("add", [DType::F32]);
    capabilities.autograd = AutogradSupport::None;
    capabilities.layouts = vec![MemoryLayout::Contiguous];
}
