//! Model export/import through safetensors: names, aliases and round trips.

use std::sync::Arc;

use oxide_ml::prelude::*;
use oxide_ml::{load_module_safetensors, safetensors, save_module_safetensors};

#[derive(Module)]
struct Mlp {
    fc1: Linear,
    norm: LayerNorm,
    fc2: Linear,
}

impl Mlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let x = self.fc1.forward(x).relu();
        let x = self.norm.forward(x);
        self.fc2.forward(x)
    }
}

fn model() -> Mlp {
    Mlp {
        fc1: Linear::with_seed(3, 4, true, 11),
        norm: LayerNorm::new(4),
        fc2: Linear::with_seed(4, 2, true, 12),
    }
}

#[test]
fn module_round_trip_preserves_predictions() {
    let path = std::env::temp_dir().join(format!(
        "oxide-model-{}-{}.safetensors",
        std::process::id(),
        1
    ));
    let original = model();
    let report = save_module_safetensors(&original, &path).unwrap();
    assert_eq!(report.aliases.len(), 0);

    let file = safetensors::SafeTensorsFile::open(&path).unwrap();
    assert!(file.contains("fc1.weight"));
    assert!(file.contains("fc2.bias"));
    assert_eq!(
        file.metadata().get("module.type").map(String::as_str),
        Some("Mlp")
    );
    assert_eq!(
        file.len(),
        6,
        "two Linear weight+bias pairs plus LayerNorm gamma/beta"
    );

    let mut restored = model();
    // Corrupt the model, then restore from the file.
    restored.fc1 = Linear::with_seed(3, 4, true, 999);
    let load = load_module_safetensors(&mut restored, &path).unwrap();
    assert!(load.is_exact(), "{load:?}");
    assert_eq!(load.loaded, 6);

    let input = Tensor::<f32>::from_vec([2, 3], vec![0.5, -0.25, 1.0, 0.1, 0.2, -0.3]).unwrap();
    let expected = original.predict(input.clone()).unwrap();
    let actual = restored.predict(input).unwrap();
    assert_eq!(actual.data(), expected.data());
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn tied_parameters_are_written_once_and_resolved_on_load() {
    #[derive(Module)]
    struct Tied {
        embed: Arc<Embedding>,
        head: TiedLinear<Embedding>,
    }

    impl Tied {
        #[model]
        fn forward(&self, ids: Tensor<f32>) -> Tensor<f32> {
            self.head.forward(self.embed.forward(ids))
        }
    }

    let embed = Arc::new(Embedding::with_seed(8, 4, 5));
    let model = Tied {
        embed: embed.clone(),
        head: TiedLinear::new(embed),
    };
    let path = std::env::temp_dir().join(format!(
        "oxide-model-{}-{}.safetensors",
        std::process::id(),
        2
    ));
    let report = save_module_safetensors(&model, &path).unwrap();
    assert_eq!(report.tensors, 1);
    assert_eq!(report.aliases.len(), 1);
    let file = safetensors::SafeTensorsFile::open(&path).unwrap();
    assert!(file.contains("embed.weight"));
    assert!(!file.contains("head.weight"));
    assert_eq!(
        file.metadata().get("alias.head.weight").map(String::as_str),
        Some("embed.weight")
    );

    // Shared storage cannot be written back through `visit_tensors_mut` (the
    // Arc is shared by design); the export metadata records the canonical name
    // so a loader can resolve it.
    struct Owned {
        embed: Embedding,
    }

    impl Module for Owned {
        fn visit_parameters(
            &self,
            path: &ModulePath,
            registry: &mut ParameterRegistry,
        ) -> Result<()> {
            self.embed.visit_parameters(&path.child("embed"), registry)
        }

        fn forward(&self, cx: &mut oxide_ml::ModelContext, input: ValueId) -> Result<Vec<ValueId>> {
            self.embed.forward(cx, input)
        }

        fn visit_tensors(
            &self,
            path: &ModulePath,
            visitor: &mut dyn FnMut(&ModulePath, &DenseTensor<f32>, ParameterRole) -> Result<()>,
        ) -> Result<()> {
            self.embed.visit_tensors(&path.child("embed"), visitor)
        }

        fn visit_tensors_mut(
            &mut self,
            path: &ModulePath,
            visitor: &mut dyn FnMut(
                &ModulePath,
                &mut DenseTensor<f32>,
                ParameterRole,
            ) -> Result<()>,
        ) -> Result<()> {
            self.embed.visit_tensors_mut(&path.child("embed"), visitor)
        }
    }

    let mut owned = Owned {
        embed: Embedding::with_seed(8, 4, 100),
    };
    let loaded = load_module_safetensors(&mut owned, &path).unwrap();
    assert!(loaded.is_exact(), "{loaded:?}");
    assert_eq!(loaded.loaded, 1);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn parameter_store_export_has_stable_names() {
    let store = ParameterStore::<f32>::new();
    let path = std::env::temp_dir().join(format!(
        "oxide-store-{}-{}.safetensors",
        std::process::id(),
        3
    ));
    let report = oxide_ml::save_parameter_store_safetensors(&store, &path);
    assert!(report.is_err(), "empty stores are rejected by the writer");
    let _ = std::fs::remove_file(&path);
}
