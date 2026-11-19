//! Kernel providers and deterministic, capability-based selection.
//!
//! A catalog is a set of candidate implementations for graph operations. Each
//! candidate advertises the dtypes, ranks, layouts, architectures and ABI it
//! supports. Selection filters by capability and then by *specificity* (a
//! specialized or fused kernel is preferred over a generic one), never by an
//! estimated speed. Ties are broken by provider name so selection is stable.

use oxide_ml_core::{DType, Error, Result};

use crate::{AutogradSupport, KERNEL_ABI_VERSION, MemoryLayout, SpecializationKey};

/// Version of provider metadata.
pub const KERNEL_PROVIDER_VERSION: u32 = 1;

/// How specialized a provider is. Higher specificity wins after capability
/// filtering; this is a structural preference, not a performance claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ProviderKind {
    Generic,
    Vectorized,
    Tiled,
    TensorCore,
    Fused,
    Specialized,
    Custom,
}

impl ProviderKind {
    pub const fn specificity(self) -> i64 {
        match self {
            ProviderKind::Generic => 0,
            ProviderKind::Vectorized => 1,
            ProviderKind::Tiled => 2,
            ProviderKind::TensorCore => 3,
            ProviderKind::Fused => 4,
            ProviderKind::Specialized => 5,
            ProviderKind::Custom => 6,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            ProviderKind::Generic => "generic",
            ProviderKind::Vectorized => "vectorized",
            ProviderKind::Tiled => "tiled",
            ProviderKind::TensorCore => "tensor-core",
            ProviderKind::Fused => "fused",
            ProviderKind::Specialized => "specialized",
            ProviderKind::Custom => "custom",
        }
    }
}

/// Capabilities advertised by one provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCapabilities {
    pub op: String,
    pub dtypes: Vec<DType>,
    pub layouts: Vec<MemoryLayout>,
    pub min_rank: Option<usize>,
    pub max_rank: Option<usize>,
    /// Empty means any architecture.
    pub architectures: Vec<String>,
    pub workspace_bytes: usize,
    pub autograd: AutogradSupport,
    pub abi_version: u32,
}

impl KernelCapabilities {
    pub fn generic(op: impl Into<String>, dtypes: impl IntoIterator<Item = DType>) -> Self {
        KernelCapabilities {
            op: op.into(),
            dtypes: dtypes.into_iter().collect(),
            layouts: vec![MemoryLayout::Contiguous],
            min_rank: None,
            max_rank: None,
            architectures: Vec::new(),
            workspace_bytes: 0,
            autograd: AutogradSupport::Explicit,
            abi_version: KERNEL_ABI_VERSION,
        }
    }

    pub fn with_rank(mut self, min_rank: usize, max_rank: usize) -> Self {
        self.min_rank = Some(min_rank);
        self.max_rank = Some(max_rank);
        self
    }

    pub fn with_workspace(mut self, bytes: usize) -> Self {
        self.workspace_bytes = bytes;
        self
    }

    pub fn with_architectures(mut self, architectures: Vec<String>) -> Self {
        self.architectures = architectures;
        self
    }

    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.dtypes.contains(&dtype)
    }

    fn supports_rank(&self, rank: usize) -> bool {
        self.min_rank.map(|min| rank >= min).unwrap_or(true)
            && self.max_rank.map(|max| rank <= max).unwrap_or(true)
    }

    fn supports_architecture(&self, architecture: &str) -> bool {
        self.architectures.is_empty()
            || self
                .architectures
                .iter()
                .any(|candidate| candidate == architecture || candidate == "any")
    }
}

/// One candidate implementation.
#[derive(Clone, Debug)]
pub struct KernelProvider {
    pub name: String,
    pub kind: ProviderKind,
    pub capabilities: KernelCapabilities,
    /// Explicit ordering among equally specific providers (higher first).
    pub priority: i32,
    /// Optional specialization identity (dtype/shape/arch/ABI/constants).
    pub specialization: Option<SpecializationKey>,
}

impl KernelProvider {
    pub fn new(
        name: impl Into<String>,
        kind: ProviderKind,
        capabilities: KernelCapabilities,
    ) -> Self {
        KernelProvider {
            name: name.into(),
            kind,
            capabilities,
            priority: 0,
            specialization: None,
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_specialization(mut self, key: SpecializationKey) -> Self {
        self.specialization = Some(key);
        self
    }

    fn score(&self) -> i64 {
        self.kind.specificity() * 1_000 + i64::from(self.priority)
    }
}

/// What a graph node needs from a provider.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelRequest {
    pub op: String,
    pub dtypes: Vec<DType>,
    pub shapes: Vec<Vec<usize>>,
    pub layout: MemoryLayout,
    pub architecture: String,
    pub abi_version: u32,
    /// Maximum workspace the caller allows. `None` means unbounded.
    pub workspace_budget: Option<usize>,
    pub allow_custom: bool,
}

impl KernelRequest {
    pub fn new(op: impl Into<String>) -> Self {
        KernelRequest {
            op: op.into(),
            dtypes: Vec::new(),
            shapes: Vec::new(),
            layout: MemoryLayout::Contiguous,
            architecture: "portable".to_string(),
            abi_version: KERNEL_ABI_VERSION,
            workspace_budget: None,
            allow_custom: true,
        }
    }

    pub fn with_dtypes(mut self, dtypes: impl IntoIterator<Item = DType>) -> Self {
        self.dtypes = dtypes.into_iter().collect();
        self
    }

    pub fn with_shapes(mut self, shapes: Vec<Vec<usize>>) -> Self {
        self.shapes = shapes;
        self
    }

    pub fn with_architecture(mut self, architecture: impl Into<String>) -> Self {
        self.architecture = architecture.into();
        self
    }

    pub fn with_workspace_budget(mut self, bytes: usize) -> Self {
        self.workspace_budget = Some(bytes);
        self
    }

    fn rank(&self) -> Option<usize> {
        self.shapes.first().map(|shape| shape.len())
    }
}

/// Result of a successful selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelSelection {
    pub provider: String,
    pub kind: ProviderKind,
    pub score: i64,
    /// Other capable providers, ordered by decreasing score.
    pub alternatives: Vec<String>,
}

/// Set of candidate providers.
#[derive(Clone, Debug, Default)]
pub struct KernelCatalog {
    providers: Vec<KernelProvider>,
}

impl KernelCatalog {
    pub fn new() -> Self {
        KernelCatalog {
            providers: Vec::new(),
        }
    }

    /// Catalog with one generic provider per operation name for the given
    /// dtypes. This is the reference CPU baseline.
    pub fn builtin(op_names: &[&str], dtypes: &[DType]) -> Self {
        let mut catalog = Self::new();
        for op in op_names {
            catalog.register(KernelProvider::new(
                format!("reference.{op}"),
                ProviderKind::Generic,
                KernelCapabilities::generic(*op, dtypes.iter().copied()),
            ));
        }
        catalog
    }

    pub fn builtin_cpu() -> Self {
        Self::builtin(
            &[
                "add",
                "add_relu",
                "sub",
                "mul",
                "div",
                "neg",
                "matmul",
                "matmul_bias",
                "relu",
                "gelu",
                "sigmoid",
                "tanh",
                "softmax",
                "log_softmax",
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
                "sqrt",
                "exp",
                "log",
                "abs",
                "slice",
                "embedding",
                "cast",
                "concat",
                "split",
                "mul_add",
                "custom",
                // Backward primitives (used when a catalog covers training graphs).
                "activation_backward",
                "add_relu_backward",
                "softmax_backward",
                "layer_norm_backward",
                "mse_loss_backward",
                "cross_entropy_loss_backward",
                "conv2d_backward",
                "max_pool2d_backward",
                "avg_pool2d_backward",
                "unbroadcast",
                "abs_backward",
                "slice_backward",
                "embedding_backward",
            ],
            &[DType::F32, DType::F64],
        )
    }

    /// Register a provider. Duplicate names are rejected.
    pub fn register(&mut self, provider: KernelProvider) -> &mut Self {
        let name = provider.name.clone();
        if self.providers.iter().all(|existing| existing.name != name) {
            self.providers.push(provider);
        }
        self
    }

    /// Register and surface a duplicate-name error.
    pub fn try_register(&mut self, provider: KernelProvider) -> Result<()> {
        if self
            .providers
            .iter()
            .any(|existing| existing.name == provider.name)
        {
            return Err(Error::invalid_graph(format!(
                "kernel provider {:?} is already registered",
                provider.name
            )));
        }
        self.providers.push(provider);
        Ok(())
    }

    pub fn providers(&self) -> &[KernelProvider] {
        &self.providers
    }

    pub fn len(&self) -> usize {
        self.providers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&KernelProvider> {
        self.providers.iter().find(|provider| provider.name == name)
    }

    fn capable(&self, provider: &KernelProvider, request: &KernelRequest) -> bool {
        if provider.capabilities.op != request.op {
            return false;
        }
        if provider.capabilities.abi_version != request.abi_version {
            return false;
        }
        if provider.kind == ProviderKind::Custom && !request.allow_custom {
            return false;
        }
        if !provider.capabilities.layouts.contains(&request.layout) {
            return false;
        }
        if !request
            .dtypes
            .iter()
            .all(|dtype| provider.capabilities.supports_dtype(*dtype))
        {
            return false;
        }
        if let Some(rank) = request.rank()
            && !provider.capabilities.supports_rank(rank)
        {
            return false;
        }
        if !provider
            .capabilities
            .supports_architecture(&request.architecture)
        {
            return false;
        }
        if let Some(budget) = request.workspace_budget
            && provider.capabilities.workspace_bytes > budget
        {
            return false;
        }
        true
    }

    /// Providers capable of serving the request, best first.
    pub fn candidates(&self, request: &KernelRequest) -> Vec<&KernelProvider> {
        let mut candidates: Vec<&KernelProvider> = self
            .providers
            .iter()
            .filter(|provider| self.capable(provider, request))
            .collect();
        candidates.sort_by(|left, right| {
            right
                .score()
                .cmp(&left.score())
                .then_with(|| left.name.cmp(&right.name))
        });
        candidates
    }

    /// Select one provider deterministically.
    pub fn select(&self, request: &KernelRequest) -> Result<KernelSelection> {
        let candidates = self.candidates(request);
        let Some(best) = candidates.first() else {
            return Err(Error::unsupported(format!(
                "no kernel provider for op {:?} (dtypes {:?}, architecture {}, ABI {})",
                request.op, request.dtypes, request.architecture, request.abi_version
            )));
        };
        Ok(KernelSelection {
            provider: best.name.clone(),
            kind: best.kind,
            score: best.score(),
            alternatives: candidates
                .iter()
                .skip(1)
                .map(|provider| provider.name.clone())
                .collect(),
        })
    }
}
