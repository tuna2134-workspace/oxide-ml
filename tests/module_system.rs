//! Module-system tests: `ModuleList`, enum architectural variants, module
//! collection loops in `#[model]`, and stable parameter paths.

use std::sync::Arc;

use oxide_ml::prelude::*;

// ---------------------------------------------------------------------------
// ModuleList and #[model] loops
// ---------------------------------------------------------------------------

#[derive(Module)]
struct LoopMlp {
    layers: ModuleList<Linear>,
    head: Linear,
}

impl LoopMlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        self.head.forward(hidden)
    }
}

fn loop_mlp() -> LoopMlp {
    LoopMlp {
        layers: ModuleList::from_vec(vec![
            Linear::with_seed(2, 4, true, 11),
            Linear::with_seed(4, 4, true, 12),
        ]),
        head: Linear::with_seed(4, 1, true, 13),
    }
}

#[test]
fn module_list_assigns_indexed_parameter_paths() {
    let model = loop_mlp();
    let mut names = Vec::new();
    model
        .visit_tensors(&ModulePath::root(), &mut |path, tensor, _| {
            names.push((path.as_string(), tensor.dims()));
            Ok(())
        })
        .unwrap();
    assert_eq!(
        names,
        vec![
            ("layers.0.weight".to_string(), vec![4, 2]),
            ("layers.0.bias".to_string(), vec![4]),
            ("layers.1.weight".to_string(), vec![4, 4]),
            ("layers.1.bias".to_string(), vec![4]),
            ("head.weight".to_string(), vec![1, 4]),
            ("head.bias".to_string(), vec![1]),
        ]
    );
    assert_eq!(model.layers.len(), 2);
    assert_eq!(model.layers[0].out_features, 4);
    assert_eq!(model.layers.first().unwrap().in_features, 2);
}

#[test]
fn module_collection_loop_matches_explicit_chain() {
    let model = loop_mlp();
    let input = Tensor::<f32>::from_vec([3, 2], vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6]).unwrap();
    let looped = model.predict(input.clone()).unwrap();

    let sequential = Sequential::new()
        .add(model.layers[0].clone())
        .add(model.layers[1].clone())
        .add(model.head.clone());
    let explicit = sequential.predict(input).unwrap();
    assert_eq!(looped.dims(), vec![3, 1]);
    for (a, b) in looped.data().iter().zip(explicit.data()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

#[derive(Module)]
struct ResidualBlock {
    norm: LayerNorm,
    layers: ModuleList<Linear>,
}

impl ResidualBlock {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let residual = x.clone();
        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        self.norm.forward(hidden + residual)
    }
}

#[derive(Module)]
struct NestedModel {
    blocks: ModuleList<ResidualBlock>,
    head: Linear,
}

impl NestedModel {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let mut hidden = x;
        for block in &self.blocks {
            for layer in &block.layers {
                hidden = layer.forward(hidden);
            }
            hidden = block.norm.forward(hidden);
        }
        self.head.forward(hidden)
    }
}

#[test]
fn nested_module_loops_build_stable_paths() {
    let model = NestedModel {
        blocks: ModuleList::from_vec(vec![
            ResidualBlock {
                norm: LayerNorm::new(4),
                layers: ModuleList::from_vec(vec![Linear::with_seed(4, 4, true, 21)]),
            },
            ResidualBlock {
                norm: LayerNorm::new(4),
                layers: ModuleList::from_vec(vec![
                    Linear::with_seed(4, 4, true, 22),
                    Linear::with_seed(4, 4, true, 23),
                ]),
            },
        ]),
        head: Linear::with_seed(4, 2, true, 24),
    };
    let mut names = Vec::new();
    model
        .visit_tensors(&ModulePath::root(), &mut |path, _, _| {
            names.push(path.as_string());
            Ok(())
        })
        .unwrap();
    assert!(names.contains(&"blocks.0.layers.0.weight".to_string()));
    assert!(names.contains(&"blocks.1.layers.1.bias".to_string()));
    assert!(names.contains(&"blocks.1.norm.gamma".to_string()));

    let input = Tensor::<f32>::from_vec([2, 4], vec![0.5; 8]).unwrap();
    let output = model.predict(input).unwrap();
    assert_eq!(output.dims(), vec![2, 2]);
}

#[derive(Module)]
struct PersistentStack<M: Module> {
    layers: ModuleList<M>,
}

impl<M: Module> PersistentStack<M> {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let mut hidden = x;
        for layer in &self.layers {
            hidden = layer.forward(hidden);
        }
        hidden
    }
}

#[test]
fn generic_modules_hold_module_lists() {
    let stack = PersistentStack {
        layers: ModuleList::from_vec(vec![
            Linear::with_seed(3, 3, true, 31),
            Linear::with_seed(3, 3, true, 32),
        ]),
    };
    let input = Tensor::<f32>::from_vec([1, 3], vec![1.0, 2.0, 3.0]).unwrap();
    let output = stack.predict(input).unwrap();
    assert_eq!(output.dims(), vec![1, 3]);
}

// ---------------------------------------------------------------------------
// Enum architectural variants
// ---------------------------------------------------------------------------

#[derive(Module)]
enum Activation {
    Relu(ReLU),
    Tanh(Tanh),
}

#[derive(Module)]
enum TwoFieldBlock {
    Dense {
        #[module(body)]
        mlp: Linear,
        #[module(skip)]
        label: &'static str,
    },
    Wide {
        #[module(body)]
        wide: Linear,
        router: Linear,
    },
}

#[test]
fn enum_variant_dispatches_forward_and_parameters() {
    let relu_model = Activation::Relu(ReLU);
    let tanh_model = Activation::Tanh(Tanh);
    let mut names = Vec::new();
    relu_model
        .visit_parameters(&ModulePath::root(), &mut ParameterRegistry::new())
        .unwrap();
    relu_model
        .visit_tensors(&ModulePath::root(), &mut |path, _, _| {
            names.push(path.as_string());
            Ok(())
        })
        .unwrap();
    assert!(names.is_empty());

    let descriptor = relu_model.descriptor().unwrap();
    assert_eq!(descriptor.module_type, "Activation");

    let input = Tensor::<f32>::from_vec([4], vec![-1.0, -0.5, 0.5, 1.0]).unwrap();
    let relu_out = oxide_ml::predict_single(&relu_model, input.clone()).unwrap();
    let tanh_out = oxide_ml::predict_single(&tanh_model, input).unwrap();
    assert_eq!(relu_out.data(), &[0.0, 0.0, 0.5, 1.0]);
    assert!((tanh_out.data()[0] - (-1.0f32).tanh()).abs() < 1e-6);
}

#[test]
fn enum_body_marker_selects_the_forward_field() {
    // The dense variant forwards to `mlp`; `label` is skipped entirely.
    let model = TwoFieldBlock::Dense {
        mlp: Linear::with_seed(2, 2, true, 41),
        label: "dense",
    };
    let input = Tensor::<f32>::from_vec([1, 2], vec![0.5, -0.25]).unwrap();
    let output = oxide_ml::predict_single(&model, input).unwrap();
    assert_eq!(output.dims(), vec![1, 2]);

    // The wide variant forwards to `wide`, but still registers the router
    // parameters so checkpoints stay structurally stable across variants.
    let wide = TwoFieldBlock::Wide {
        wide: Linear::with_seed(2, 2, true, 42),
        router: Linear::with_seed(2, 3, true, 43),
    };
    let mut names = Vec::new();
    wide.visit_tensors(&ModulePath::root(), &mut |path, _, _| {
        names.push(path.as_string());
        Ok(())
    })
    .unwrap();
    assert!(names.contains(&"wide.weight".to_string()));
    assert!(names.contains(&"router.weight".to_string()));
    let input = Tensor::<f32>::from_vec([1, 2], vec![0.5, -0.25]).unwrap();
    assert_eq!(
        oxide_ml::predict_single(&wide, input).unwrap().dims(),
        vec![1, 2]
    );
}

// ---------------------------------------------------------------------------
// ModuleList container behaviour
// ---------------------------------------------------------------------------

#[test]
fn module_list_container_operations() {
    let mut list: ModuleList<Linear> = ModuleList::new();
    assert!(list.is_empty());
    list.push(Linear::with_seed(2, 2, false, 1));
    list.insert(0, Linear::with_seed(2, 2, false, 2));
    assert_eq!(list.len(), 2);
    assert_eq!(list.as_slice().len(), 2);
    assert_eq!(list.iter().count(), 2);
    list.iter_mut().for_each(|layer| layer.set_training(false));
    let removed = list.remove(0);
    assert_eq!(removed.in_features, 2);
    assert_eq!(list.into_vec().len(), 1);

    let from_iter: ModuleList<Linear> = (0..3)
        .map(|index| Linear::with_seed(2, 2, false, index))
        .collect();
    assert_eq!(from_iter.len(), 3);
    assert_eq!(from_iter.last().unwrap().out_features, 2);
}

// ---------------------------------------------------------------------------
// Tied parameters
// ---------------------------------------------------------------------------

/// A character-level language-model head tied to the token embedding.
#[derive(Module)]
struct TiedLm {
    embed: Arc<Embedding>,
    norm: LayerNorm,
    head: TiedLinear<Embedding>,
}

impl TiedLm {
    #[model]
    fn forward(&self, ids: Tensor<f32>) -> Tensor<f32> {
        let hidden = self.embed.forward(ids);
        let hidden = self.norm.forward(hidden);
        self.head.forward(hidden)
    }
}

fn tied_lm() -> (TiedLm, Arc<Embedding>) {
    let embed = Arc::new(Embedding::with_seed(16, 8, 51));
    let model = TiedLm {
        embed: embed.clone(),
        norm: LayerNorm::new(8),
        head: TiedLinear::new(embed.clone()),
    };
    (model, embed)
}

#[test]
fn tied_head_shares_one_canonical_parameter() {
    let (model, embed) = tied_lm();
    let descriptor = model.descriptor().unwrap();
    assert_eq!(
        descriptor.parameter_count(),
        3,
        "embed.weight + norm.gamma/beta"
    );

    let mut registry = ParameterRegistry::new();
    model
        .visit_parameters(&ModulePath::root(), &mut registry)
        .unwrap();
    assert_eq!(registry.specs().len(), 3);
    assert_eq!(registry.aliases().len(), 1);
    assert_eq!(
        registry.aliases()[0],
        ("head.weight".to_string(), "embed.weight".to_string())
    );
    assert_eq!(registry.canonical_name("head.weight"), Some("embed.weight"));

    let compiled = CompiledModel::compile(
        &model,
        &[InputSpec::unbatched("input0", [3])],
        ModelCompileOptions::default(),
    )
    .unwrap();
    assert_eq!(compiled.parameter_ids().len(), 3);
    assert_eq!(
        compiled.aliases().get("head.weight").map(String::as_str),
        Some("embed.weight")
    );

    let store = compiled.parameter_store(&model).unwrap();
    assert_eq!(store.len(), 3);
    assert_eq!(
        store.get_by_name("embed.weight").unwrap().data.dims(),
        [16, 8]
    );
    // Exactly one canonical parameter: the alias is not a second store entry.
    assert!(compiled.parameter_ids().contains_key("embed.weight"));
    assert!(!compiled.parameter_ids().contains_key("head.weight"));
    assert_eq!(
        store.get_by_name("embed.weight").unwrap().data.data()[0],
        embed.weight.data()[0]
    );
}

#[test]
fn tied_head_matches_an_independent_copy() {
    let (model, embed) = tied_lm();
    let ids = Tensor::<f32>::from_vec([3], vec![1.0, 5.0, 9.0]).unwrap();
    let tied = model.predict(ids.clone()).unwrap();

    // The same math with a copied weight: equal values, but two parameters.
    let untied = TiedLm {
        embed: embed.clone(),
        norm: model.norm.clone(),
        head: TiedLinear::new(Arc::new(
            Embedding::from_weight(embed.weight.clone()).unwrap(),
        )),
    };
    let copied = untied.predict(ids).unwrap();
    assert_eq!(tied.dims(), vec![3, 16]);
    for (a, b) in tied.data().iter().zip(copied.data()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
    assert_eq!(
        untied.descriptor().unwrap().parameter_count(),
        4,
        "a copied weight is a second independent parameter"
    );
}
