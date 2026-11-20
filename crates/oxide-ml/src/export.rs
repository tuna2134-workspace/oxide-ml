//! Model and parameter-store export to safetensors.
//!
//! The writer itself lives in `oxide-ml-safetensors` and knows nothing about
//! modules. These helpers add the model contract: stable parameter names, tied
//! parameter aliases (the canonical tensor is written once and the alias is
//! recorded in metadata), roles and module metadata.

use std::collections::{BTreeMap, HashMap};

use oxide_ml_core::{DenseTensor, Error, ParameterStore, Result};
use oxide_ml_model::{Module, ModulePath};
use oxide_ml_safetensors::{SafeTensorsFile, SafeTensorsWriter, TensorRecord};

pub const EXPORT_FORMAT: &str = "oxide-ml.safetensors";
pub const EXPORT_VERSION: u32 = 1;

/// Result of writing a model or parameter store.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExportReport {
    pub tensors: usize,
    pub bytes: u64,
    /// `(alias, canonical)` paths that shared storage.
    pub aliases: Vec<(String, String)>,
    /// Parameters skipped because write-back is impossible for them.
    pub skipped: Vec<String>,
}

/// Result of loading a safetensors file into a module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SafetensorsLoadReport {
    pub loaded: usize,
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub aliases: Vec<(String, String)>,
    pub skipped: Vec<String>,
}

impl SafetensorsLoadReport {
    pub fn is_exact(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.skipped.is_empty()
    }
}

/// Write every parameter of `model` to a safetensors file.
///
/// Shared storage is written once under its first canonical path; subsequent
/// paths are recorded as `alias.<name>=<canonical>` metadata. Parameter roles
/// and the module type are stored as metadata as well.
pub fn save_module_safetensors<M: Module + ?Sized>(
    model: &M,
    path: impl AsRef<std::path::Path>,
) -> Result<ExportReport> {
    save_module_safetensors_with_metadata(model, path, &[])
}

/// Like [`save_module_safetensors`] with extra string metadata (class names,
/// input size, training configuration, ...).
pub fn save_module_safetensors_with_metadata<M: Module + ?Sized>(
    model: &M,
    path: impl AsRef<std::path::Path>,
    metadata: &[(&str, &str)],
) -> Result<ExportReport> {
    let mut writer = SafeTensorsWriter::new();
    for (key, value) in metadata {
        if key.starts_with("alias.") || *key == "__metadata__" {
            return Err(Error::invalid_shape(
                "safetensors export",
                format!("reserved metadata key {key:?}"),
            ));
        }
        writer.set_metadata(*key, *value);
    }
    writer.set_metadata("format", EXPORT_FORMAT);
    writer.set_metadata("version", EXPORT_VERSION.to_string());
    writer.set_metadata("module.type", model.module_type());

    let mut canonical: HashMap<(usize, usize), String> = HashMap::new();
    let mut aliases = Vec::new();
    let mut tensors = 0usize;
    let mut bytes = 0u64;
    let mut visited = 0usize;
    model.visit_tensors(&ModulePath::root(), &mut |path, tensor, role| {
        visited += 1;
        let name = path.as_string();
        let key = (tensor.data().as_ptr() as usize, tensor.data().len());
        if let Some(canonical_name) = canonical.get(&key) {
            aliases.push((name.clone(), canonical_name.clone()));
            writer.set_metadata(format!("alias.{name}"), canonical_name.clone());
            return Ok(());
        }
        canonical.insert(key, name.clone());
        writer.set_metadata(format!("role.{name}"), role.as_str().to_string());
        let record = TensorRecord::f32(name, tensor.dims(), tensor.data())?;
        bytes += record.data.len() as u64;
        writer.add_tensor(record)?;
        tensors += 1;
        Ok(())
    })?;
    writer.set_metadata("module.parameters", visited.to_string());
    writer.set_metadata("module.tensors", tensors.to_string());
    writer.write(path)?;
    Ok(ExportReport {
        tensors,
        bytes,
        aliases,
        skipped: Vec::new(),
    })
}

/// Write a parameter store (for example the training-authoritative store) to
/// safetensors. Every store entry is written; no aliasing is applied.
pub fn save_parameter_store_safetensors(
    store: &ParameterStore<f32>,
    path: impl AsRef<std::path::Path>,
) -> Result<ExportReport> {
    let mut writer = SafeTensorsWriter::new();
    writer.set_metadata("format", EXPORT_FORMAT);
    writer.set_metadata("version", EXPORT_VERSION.to_string());
    writer.set_metadata("source", "parameter_store");
    writer.set_metadata("parameters", store.len().to_string());
    let mut bytes = 0u64;
    let mut entries: Vec<_> = store.iter().collect();
    entries.sort_by(|left, right| left.name.cmp(&right.name));
    for parameter in &entries {
        writer.set_metadata(
            format!("role.{}", parameter.name),
            if parameter.trainable {
                "trainable"
            } else {
                "frozen"
            }
            .to_string(),
        );
        let record = TensorRecord::f32(
            parameter.name.clone(),
            parameter.data.dims(),
            parameter.data.data(),
        )?;
        bytes += record.data.len() as u64;
        writer.add_tensor(record)?;
    }
    writer.write(path)?;
    Ok(ExportReport {
        tensors: entries.len(),
        bytes,
        aliases: Vec::new(),
        skipped: Vec::new(),
    })
}

/// Load safetensors values into a module by stable parameter name.
///
/// Aliases recorded by [`save_module_safetensors`] are resolved to the
/// canonical tensor. Missing and unexpected tensors are reported; a shape
/// mismatch is an error.
pub fn load_module_safetensors<M: Module + ?Sized>(
    model: &mut M,
    path: impl AsRef<std::path::Path>,
) -> Result<SafetensorsLoadReport> {
    let file = SafeTensorsFile::open(path)?;
    let alias_map: HashMap<String, String> = file
        .metadata()
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("alias.")
                .map(|_| (key.clone(), value.clone()))
        })
        .map(|(key, value)| (key.trim_start_matches("alias.").to_string(), value))
        .collect();

    let mut report = SafetensorsLoadReport::default();
    let mut seen = Vec::new();
    model.visit_tensors_mut(&ModulePath::root(), &mut |path, tensor, _role| {
        let name = path.as_string();
        let resolved = if file.contains(&name) {
            Some(name.clone())
        } else {
            alias_map.get(&name).cloned()
        };
        let Some(name) = resolved else {
            report.missing.push(name);
            return Ok(());
        };
        seen.push(name.clone());
        let loaded = file.tensor(&name)?;
        if loaded.shape != tensor.dims() {
            return Err(Error::shape_mismatch(
                format!("safetensors tensor {name:?}"),
                tensor.shape().clone(),
                oxide_ml_core::Shape::from_static(&loaded.shape),
            ));
        }
        *tensor = DenseTensor::from_vec(&loaded.shape, loaded.data)?;
        report.loaded += 1;
        Ok(())
    })?;

    for name in file.names() {
        if !seen.iter().any(|seen| seen == name) && !alias_map.contains_key(name) {
            report.unexpected.push(name.to_string());
        }
    }
    report.aliases = alias_map.into_iter().collect();
    Ok(report)
}

/// Export a model to safetensors and verify it can be read back.
pub fn verify_export<M: Module + ?Sized>(
    model: &M,
    path: impl AsRef<std::path::Path>,
) -> Result<ExportReport> {
    let report = save_module_safetensors(model, path.as_ref())?;
    let file = SafeTensorsFile::open(path.as_ref())?;
    if file.len() != report.tensors {
        return Err(Error::Serialization(format!(
            "exported {} tensors but reread {}",
            report.tensors,
            file.len()
        )));
    }
    Ok(report)
}

/// Plain map of parameter name -> tensor for tooling.
pub fn module_tensors<M: Module + ?Sized>(model: &M) -> Result<BTreeMap<String, DenseTensor<f32>>> {
    let mut map = BTreeMap::new();
    model.visit_tensors(&ModulePath::root(), &mut |path, tensor, _| {
        map.insert(path.as_string(), tensor.clone());
        Ok(())
    })?;
    Ok(map)
}
