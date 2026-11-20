//! Safetensors writer.
//!
//! The writer is independent of any model architecture: it serializes named
//! tensors (F32/F16/BF16/I32/I64/U8/BOOL) plus optional string metadata into the
//! standard safetensors layout. Model-specific helpers (parameter traversal,
//! tied-parameter aliases) live in the facade crate.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use oxide_ml_core::{Error, Result};

use crate::StoredDType;

/// One tensor to serialize. `data` is the raw little-endian encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRecord {
    pub name: String,
    pub dtype: StoredDType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl TensorRecord {
    pub fn new(
        name: impl Into<String>,
        dtype: StoredDType,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) -> Result<Self> {
        let record = TensorRecord {
            name: name.into(),
            dtype,
            shape,
            data,
        };
        record.validate()?;
        Ok(record)
    }

    /// F32 tensor from host values.
    pub fn f32(
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<Self> {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        Self::new(name, StoredDType::F32, shape.into(), data)
    }

    /// F16 tensor, converting host f32 values.
    pub fn f16(
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<Self> {
        let mut data = Vec::with_capacity(values.len() * 2);
        for value in values {
            data.extend_from_slice(&half::f16::from_f32(*value).to_bits().to_le_bytes());
        }
        Self::new(name, StoredDType::F16, shape.into(), data)
    }

    /// BF16 tensor, converting host f32 values.
    pub fn bf16(
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<Self> {
        let mut data = Vec::with_capacity(values.len() * 2);
        for value in values {
            data.extend_from_slice(&half::bf16::from_f32(*value).to_bits().to_le_bytes());
        }
        Self::new(name, StoredDType::BF16, shape.into(), data)
    }

    fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn element_size(&self) -> usize {
        match self.dtype {
            StoredDType::F32 | StoredDType::I32 => 4,
            StoredDType::F16 | StoredDType::BF16 => 2,
            StoredDType::I64 => 8,
            StoredDType::U8 | StoredDType::Bool => 1,
        }
    }

    fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(Error::invalid_shape("safetensors", "tensor name is empty"));
        }
        if self.name == "__metadata__" {
            return Err(Error::invalid_shape(
                "safetensors",
                "__metadata__ is reserved",
            ));
        }
        let expected = self
            .numel()
            .checked_mul(self.element_size())
            .ok_or_else(|| Error::invalid_shape("safetensors", "tensor size overflows"))?;
        if self.data.len() != expected {
            return Err(Error::invalid_shape(
                "safetensors",
                format!(
                    "{} expects {expected} bytes for {:?} {:?}, got {}",
                    self.name,
                    self.dtype,
                    self.shape,
                    self.data.len()
                ),
            ));
        }
        Ok(())
    }
}

/// Builder for a safetensors file.
#[derive(Clone, Debug, Default)]
pub struct SafeTensorsWriter {
    metadata: BTreeMap<String, String>,
    tensors: BTreeMap<String, TensorRecord>,
}

impl SafeTensorsWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_metadata(key: impl Into<String>, value: impl Into<String>) -> Self {
        let mut writer = Self::new();
        writer.metadata.insert(key.into(), value.into());
        writer
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Add a tensor. Returns the writer for chaining.
    pub fn tensor(mut self, tensor: TensorRecord) -> Result<Self> {
        self.add_tensor(tensor)?;
        Ok(self)
    }

    pub fn add_tensor(&mut self, tensor: TensorRecord) -> Result<()> {
        tensor.validate()?;
        if self.tensors.contains_key(&tensor.name) {
            return Err(Error::invalid_shape(
                "safetensors",
                format!("duplicate tensor name {:?}", tensor.name),
            ));
        }
        self.tensors.insert(tensor.name.clone(), tensor);
        Ok(())
    }

    pub fn add_f32(
        &mut self,
        name: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        values: &[f32],
    ) -> Result<()> {
        self.add_tensor(TensorRecord::f32(name, shape, values)?)
    }

    /// Serialize to the safetensors byte layout.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        if self.tensors.is_empty() {
            return Err(Error::invalid_shape(
                "safetensors",
                "cannot write a file with no tensors",
            ));
        }
        let mut header = serde_json::Map::new();
        if !self.metadata.is_empty() {
            header.insert(
                "__metadata__".to_string(),
                serde_json::Value::Object(
                    self.metadata
                        .iter()
                        .map(|(key, value)| (key.clone(), serde_json::Value::String(value.clone())))
                        .collect(),
                ),
            );
        }
        // Assign aligned offsets before writing the header, because offsets are
        // part of the header.
        let mut cursor = 0usize;
        let mut offsets = BTreeMap::new();
        for (name, tensor) in &self.tensors {
            let aligned = (cursor + 7) & !7;
            cursor = aligned + tensor.data.len();
            offsets.insert(name.clone(), (aligned, cursor));
        }
        for (name, tensor) in &self.tensors {
            let (begin, end) = offsets[name];
            let mut entry = serde_json::Map::new();
            entry.insert(
                "dtype".to_string(),
                serde_json::Value::String(tensor.dtype.name().to_string()),
            );
            entry.insert(
                "shape".to_string(),
                serde_json::Value::Array(
                    tensor
                        .shape
                        .iter()
                        .map(|dim| serde_json::Value::Number((*dim as u64).into()))
                        .collect(),
                ),
            );
            entry.insert(
                "data_offsets".to_string(),
                serde_json::Value::Array(vec![
                    serde_json::Value::Number((begin as u64).into()),
                    serde_json::Value::Number((end as u64).into()),
                ]),
            );
            header.insert(name.clone(), serde_json::Value::Object(entry));
        }
        let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
            .map_err(|error| Error::Serialization(format!("safetensors header: {error}")))?;
        // The header is padded with spaces to an 8-byte boundary.
        while (8 + header_bytes.len()) % 8 != 0 {
            header_bytes.push(b' ');
        }
        let mut output = Vec::with_capacity(8 + header_bytes.len() + cursor);
        output.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        output.extend_from_slice(&header_bytes);
        let data_start = output.len();
        output.resize(data_start + cursor, 0);
        for (name, tensor) in &self.tensors {
            let (begin, _) = offsets[name];
            let start = data_start + begin;
            output[start..start + tensor.data.len()].copy_from_slice(&tensor.data);
        }
        Ok(output)
    }

    /// Write atomically through a temporary file in the same directory.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let bytes = self.to_bytes()?;
        let temporary = temp_path(path);
        {
            let mut file = std::fs::File::create(&temporary)
                .map_err(|error| Error::Io(format!("create {}: {error}", temporary.display())))?;
            file.write_all(&bytes)
                .map_err(|error| Error::Io(format!("write {}: {error}", temporary.display())))?;
            file.sync_all()
                .map_err(|error| Error::Io(format!("sync {}: {error}", temporary.display())))?;
        }
        std::fs::rename(&temporary, path).map_err(|error| {
            let _ = std::fs::remove_file(&temporary);
            Error::Io(format!("rename {}: {error}", path.display()))
        })
    }
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map_or_else(
        || std::ffi::OsString::from("model.safetensors"),
        |name| name.to_os_string(),
    );
    name.push(format!(".tmp.{}", std::process::id()));
    path.with_file_name(name)
}
