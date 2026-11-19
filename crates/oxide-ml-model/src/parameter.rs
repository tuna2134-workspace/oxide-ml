//! Parameter discovery independent from graph construction.
//!
//! A [`ParameterRegistry`] is the result of traversing a module tree. It owns
//! stable, human-readable names and shapes, and it detects duplicate names as
//! well as shared storage (the same tensor reachable through two module paths).
//! Shared storage is reported as an alias instead of being duplicated, so the
//! runtime can bind both names to a single graph parameter and a single
//! optimizer slot.

use std::collections::HashMap;

use oxide_ml_core::{DType, DenseTensor, Error, Result};

use crate::module::ModulePath;

/// How a tensor participates in training.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParameterRole {
    /// Updated by the optimizer.
    Trainable,
    /// Saved and bound, but never updated.
    Frozen,
    /// Non-parameter state such as running statistics. Saved, never updated by
    /// the optimizer.
    Buffer,
}

impl ParameterRole {
    pub const fn is_trainable(self) -> bool {
        matches!(self, ParameterRole::Trainable)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            ParameterRole::Trainable => "trainable",
            ParameterRole::Frozen => "frozen",
            ParameterRole::Buffer => "buffer",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "trainable" => Some(ParameterRole::Trainable),
            "frozen" => Some(ParameterRole::Frozen),
            "buffer" => Some(ParameterRole::Buffer),
            _ => None,
        }
    }
}

/// One discovered parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterSpec {
    /// Full stable name, e.g. `fc1.weight`.
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub role: ParameterRole,
    /// Module that owns the tensor, e.g. `fc1`.
    pub module_path: ModulePath,
}

impl ParameterSpec {
    pub fn new(
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        role: ParameterRole,
        module_path: ModulePath,
    ) -> Self {
        ParameterSpec {
            name: name.into(),
            shape: shape.into(),
            dtype: DType::F32,
            role,
            module_path,
        }
    }

    pub fn trainable(
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        module_path: ModulePath,
    ) -> Self {
        Self::new(name, shape, ParameterRole::Trainable, module_path)
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Collects parameter metadata during a module traversal.
#[derive(Clone, Debug, Default)]
pub struct ParameterRegistry {
    specs: Vec<ParameterSpec>,
    index: HashMap<String, usize>,
    /// Every discovered name (including aliases) -> canonical name.
    canonical: HashMap<String, String>,
    /// Tensor storage address -> canonical parameter name.
    storage: HashMap<usize, String>,
    aliases: Vec<(String, String)>,
    allow_aliases: bool,
}

impl ParameterRegistry {
    pub fn new() -> Self {
        ParameterRegistry {
            allow_aliases: true,
            ..Default::default()
        }
    }

    /// Reject shared storage instead of treating it as an alias.
    pub fn strict() -> Self {
        ParameterRegistry {
            allow_aliases: false,
            ..Default::default()
        }
    }

    pub fn specs(&self) -> &[ParameterSpec] {
        &self.specs
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&ParameterSpec> {
        self.index.get(name).map(|&index| &self.specs[index])
    }

    pub fn canonical_name(&self, name: &str) -> Option<&str> {
        self.canonical.get(name).map(String::as_str)
    }

    /// Alias entries discovered because two module paths share tensor storage.
    pub fn aliases(&self) -> &[(String, String)] {
        &self.aliases
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.specs.iter().map(|spec| spec.name.as_str())
    }

    /// Register a parameter that has no directly owned tensor yet.
    pub fn insert(&mut self, spec: ParameterSpec) -> Result<()> {
        if let Some(existing) = self.index.get(&spec.name) {
            return Err(Error::invalid_graph(format!(
                "parameter {:?} was registered twice (first at module {})",
                spec.name, self.specs[*existing].module_path
            )));
        }
        if spec.shape.contains(&0) {
            return Err(Error::invalid_shape(
                "parameter",
                format!("parameter {:?} has a zero-sized dimension", spec.name),
            ));
        }
        self.index.insert(spec.name.clone(), self.specs.len());
        self.canonical.insert(spec.name.clone(), spec.name.clone());
        self.specs.push(spec);
        Ok(())
    }

    /// Register a concrete tensor discovered during traversal.
    ///
    /// If the same storage address was already registered under a different
    /// name, the new name becomes an alias of the canonical one. Registering the
    /// same name twice is always an error.
    pub fn insert_tensor(
        &mut self,
        path: &ModulePath,
        tensor: &DenseTensor<f32>,
        role: ParameterRole,
    ) -> Result<()> {
        let name = path.as_string();
        if self.canonical.contains_key(&name) {
            return Err(Error::invalid_graph(format!(
                "parameter {name:?} was registered twice"
            )));
        }
        let address = tensor.data().as_ptr() as usize;
        let has_storage = tensor.numel() > 0;
        if has_storage && let Some(canonical) = self.storage.get(&address).cloned() {
            if !self.allow_aliases {
                return Err(Error::invalid_graph(format!(
                    "parameters {canonical:?} and {name:?} share storage; \
                     shared parameters are disabled for this registry"
                )));
            }
            let canonical_spec = self
                .get(&canonical)
                .cloned()
                .ok_or_else(|| Error::invalid_graph("alias points at a missing parameter"))?;
            if canonical_spec.shape != tensor.dims() {
                return Err(Error::invalid_shape(
                    "shared parameter",
                    format!("{canonical:?} and {name:?} share storage but their shapes differ"),
                ));
            }
            self.canonical.insert(name.clone(), canonical.clone());
            self.aliases.push((name, canonical));
            return Ok(());
        }
        let module_path = path.parent().unwrap_or_else(ModulePath::root);
        let spec = ParameterSpec {
            name: name.clone(),
            shape: tensor.dims(),
            dtype: DType::F32,
            role,
            module_path,
        };
        self.insert(spec)?;
        if has_storage {
            self.storage.insert(address, name);
        }
        Ok(())
    }

    /// Names that share storage with another name.
    pub fn alias_names(&self) -> impl Iterator<Item = &str> {
        self.aliases.iter().map(|(alias, _)| alias.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_duplicate_names() {
        let mut registry = ParameterRegistry::new();
        let path = ModulePath::root().child("weight");
        registry
            .insert(ParameterSpec::trainable(
                path.as_string(),
                [4, 4],
                ModulePath::root(),
            ))
            .unwrap();
        let error = registry
            .insert(ParameterSpec::trainable(
                path.as_string(),
                [4, 4],
                ModulePath::root(),
            ))
            .unwrap_err();
        assert!(error.to_string().contains("registered twice"));
    }

    #[test]
    fn shared_storage_becomes_an_alias() {
        let tensor = DenseTensor::<f32>::ones([4, 4]);
        let mut registry = ParameterRegistry::new();
        registry
            .insert_tensor(
                &ModulePath::root().child("a").child("weight"),
                &tensor,
                ParameterRole::Trainable,
            )
            .unwrap();
        registry
            .insert_tensor(
                &ModulePath::root().child("b").child("weight"),
                &tensor,
                ParameterRole::Trainable,
            )
            .unwrap();
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry.canonical_name("b.weight"),
            Some("a.weight"),
            "second registration should alias the first"
        );
        assert_eq!(registry.aliases().len(), 1);
    }

    #[test]
    fn strict_registry_rejects_aliases() {
        let tensor = DenseTensor::<f32>::ones([2]);
        let mut registry = ParameterRegistry::strict();
        registry
            .insert_tensor(
                &ModulePath::root().child("a"),
                &tensor,
                ParameterRole::Trainable,
            )
            .unwrap();
        assert!(
            registry
                .insert_tensor(
                    &ModulePath::root().child("b"),
                    &tensor,
                    ParameterRole::Trainable
                )
                .is_err()
        );
    }
}
