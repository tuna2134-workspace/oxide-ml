//! Minimal safetensors reader that converts F32/F16/BF16 tensors to host `f32`.
//!
//! The file is memory-mapped; tensor data is decoded on demand so a multi-GB
//! checkpoint does not need a second full copy in RAM.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use oxide_ml_core::{DType, Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredDType {
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
    Bool,
}

impl StoredDType {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "F32" => Ok(StoredDType::F32),
            "F16" => Ok(StoredDType::F16),
            "BF16" => Ok(StoredDType::BF16),
            "I64" => Ok(StoredDType::I64),
            "I32" => Ok(StoredDType::I32),
            "U8" => Ok(StoredDType::U8),
            "BOOL" => Ok(StoredDType::Bool),
            other => Err(Error::unsupported(format!(
                "safetensors dtype {other} is not supported"
            ))),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            StoredDType::F32 => "F32",
            StoredDType::F16 => "F16",
            StoredDType::BF16 => "BF16",
            StoredDType::I64 => "I64",
            StoredDType::I32 => "I32",
            StoredDType::U8 => "U8",
            StoredDType::Bool => "BOOL",
        }
    }

    fn element_size(self) -> usize {
        match self {
            StoredDType::F32 | StoredDType::I32 => 4,
            StoredDType::F16 | StoredDType::BF16 => 2,
            StoredDType::I64 => 8,
            StoredDType::U8 | StoredDType::Bool => 1,
        }
    }
}

/// Decoded tensor, always in host `f32` (integer tensors are converted).
#[derive(Clone, Debug)]
pub struct LoadedTensor {
    pub name: String,
    pub dtype: StoredDType,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl LoadedTensor {
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.shape
    }
}

struct Entry {
    dtype: StoredDType,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

/// Memory-mapped safetensors file.
pub struct SafeTensorsFile {
    mmap: Mmap,
    entries: HashMap<String, Entry>,
    metadata: HashMap<String, String>,
}

impl std::fmt::Debug for SafeTensorsFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensorsFile")
            .field("tensors", &self.entries.len())
            .field("bytes", &self.mmap.len())
            .finish()
    }
}

impl SafeTensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .map_err(|error| Error::Io(format!("cannot open {}: {error}", path.display())))?;
        // SAFETY: the mapping is read-only and the file is kept alive by the
        // mapping for the lifetime of `SafeTensorsFile`. No other process is
        // assumed to truncate the checkpoint while it is in use.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|error| Error::Io(format!("mmap {}: {error}", path.display())))?;
        Self::from_mmap(mmap)
    }

    fn from_mmap(mmap: Mmap) -> Result<Self> {
        if mmap.len() < 8 {
            return Err(Error::Serialization("safetensors file is too short".into()));
        }
        let header_len = u64::from_le_bytes(mmap[..8].try_into().expect("8 bytes"));
        let header_len = usize::try_from(header_len)
            .map_err(|_| Error::Serialization("safetensors header is too large".into()))?;
        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Error::Serialization("safetensors header overflow".into()))?;
        if header_end > mmap.len() {
            return Err(Error::Serialization(
                "safetensors header exceeds file size".into(),
            ));
        }
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..header_end])
            .map_err(|error| Error::Serialization(format!("safetensors header: {error}")))?;
        let object = header
            .as_object()
            .ok_or_else(|| Error::Serialization("safetensors header is not an object".into()))?;
        let mut entries = HashMap::new();
        let mut metadata = HashMap::new();
        let data_start = header_end;
        for (name, value) in object {
            if name == "__metadata__" {
                if let Some(map) = value.as_object() {
                    for (key, value) in map {
                        if let Some(text) = value.as_str() {
                            metadata.insert(key.clone(), text.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = value
                .get("dtype")
                .and_then(|value| value.as_str())
                .ok_or_else(|| Error::Serialization(format!("tensor {name}: missing dtype")))?;
            let shape = value
                .get("shape")
                .and_then(|value| value.as_array())
                .ok_or_else(|| Error::Serialization(format!("tensor {name}: missing shape")))?;
            let shape = shape
                .iter()
                .map(|dim| {
                    dim.as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            Error::Serialization(format!("tensor {name}: invalid shape"))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let offsets = value
                .get("data_offsets")
                .and_then(|value| value.as_array())
                .ok_or_else(|| {
                    Error::Serialization(format!("tensor {name}: missing data_offsets"))
                })?;
            if offsets.len() != 2 {
                return Err(Error::Serialization(format!(
                    "tensor {name}: data_offsets must have two entries"
                )));
            }
            let begin = offsets[0]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| Error::Serialization(format!("tensor {name}: invalid offset")))?;
            let end = offsets[1]
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| Error::Serialization(format!("tensor {name}: invalid offset")))?;
            if end < begin || data_start + end > mmap.len() {
                return Err(Error::Serialization(format!(
                    "tensor {name}: data range is outside the file"
                )));
            }
            let dtype = StoredDType::parse(dtype)?;
            let numel: usize = shape.iter().product();
            if numel * dtype.element_size() != end - begin {
                return Err(Error::Serialization(format!(
                    "tensor {name}: shape/dtype do not match the data length"
                )));
            }
            entries.insert(
                name.clone(),
                Entry {
                    dtype,
                    shape,
                    begin: data_start + begin,
                    end: data_start + end,
                },
            );
        }
        Ok(SafeTensorsFile {
            mmap,
            entries,
            metadata,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|entry| entry.shape.as_slice())
    }

    pub fn dtype(&self, name: &str) -> Option<StoredDType> {
        self.entries.get(name).map(|entry| entry.dtype)
    }

    /// Decode one tensor to host `f32`.
    pub fn tensor(&self, name: &str) -> Result<LoadedTensor> {
        let entry = self.entries.get(name).ok_or_else(|| {
            Error::invalid_graph(format!("safetensors tensor {name:?} does not exist"))
        })?;
        let bytes = &self.mmap[entry.begin..entry.end];
        let numel: usize = entry.shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        match entry.dtype {
            StoredDType::F32 => {
                for chunk in bytes.chunks_exact(4) {
                    data.push(f32::from_le_bytes(chunk.try_into().expect("4 bytes")));
                }
            }
            StoredDType::F16 => {
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("2 bytes"));
                    data.push(f16_to_f32(bits));
                }
            }
            StoredDType::BF16 => {
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes(chunk.try_into().expect("2 bytes"));
                    data.push(f32::from_bits((bits as u32) << 16));
                }
            }
            StoredDType::I64 => {
                for chunk in bytes.chunks_exact(8) {
                    let value = i64::from_le_bytes(chunk.try_into().expect("8 bytes"));
                    data.push(value as f32);
                }
            }
            StoredDType::I32 => {
                for chunk in bytes.chunks_exact(4) {
                    let value = i32::from_le_bytes(chunk.try_into().expect("4 bytes"));
                    data.push(value as f32);
                }
            }
            StoredDType::U8 => {
                for value in bytes {
                    data.push(f32::from(*value));
                }
            }
            StoredDType::Bool => {
                for value in bytes {
                    data.push(if *value != 0 { 1.0 } else { 0.0 });
                }
            }
        }
        if data.len() != numel {
            return Err(Error::Serialization(format!(
                "tensor {name:?}: decoded {} values, expected {numel}",
                data.len()
            )));
        }
        Ok(LoadedTensor {
            name: name.to_string(),
            dtype: entry.dtype,
            shape: entry.shape.clone(),
            data,
        })
    }

    /// Decode a tensor or fail with the expected parameter path in the message.
    pub fn require(&self, name: &str) -> Result<LoadedTensor> {
        self.tensor(name)
    }

    /// Map an `oxide_ml_core::DType` for a stored tensor (used by validation).
    pub fn core_dtype(&self, name: &str) -> Option<DType> {
        match self.dtype(name)? {
            StoredDType::F32 => Some(DType::F32),
            StoredDType::F16 => Some(DType::F16),
            StoredDType::BF16 => Some(DType::BF16),
            StoredDType::I64 => Some(DType::I64),
            StoredDType::I32 => Some(DType::I32),
            StoredDType::U8 => Some(DType::U8),
            StoredDType::Bool => Some(DType::Bool),
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let value = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // Subnormal.
            let mut exponent = 127 - 15 + 1;
            let mut mantissa = mantissa;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((mantissa & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_and_f16_decode() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f32::from_bits(0x3f80_0000u32), 1.0);
        let bf16_one = (1.0f32.to_bits() >> 16) as u16;
        assert_eq!(f32::from_bits((bf16_one as u32) << 16), 1.0);
    }
}

#[cfg(test)]
mod file_tests {
    use super::*;
    use safetensors::tensor::{Dtype as SafeDtype, TensorView};

    #[test]
    fn reads_f32_and_bf16_tensors_from_a_real_file() {
        let f32_values = [1.0f32, -2.0];
        let f32_bytes: Vec<u8> = f32_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let bf16_bits: [u16; 2] = [0x3f80, 0xc000];
        let bf16_bytes: Vec<u8> = bf16_bits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let f32_view = TensorView::new(SafeDtype::F32, vec![2], &f32_bytes).unwrap();
        let bf16_view = TensorView::new(SafeDtype::BF16, vec![2], &bf16_bytes).unwrap();
        let serialized =
            safetensors::serialize([("f32", f32_view), ("bf16", bf16_view)], None).unwrap();
        let path = std::env::temp_dir().join(format!(
            "oxide-ml-safetensors-{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &serialized).unwrap();

        let file = SafeTensorsFile::open(&path).unwrap();
        assert_eq!(file.len(), 2);
        assert_eq!(file.shape("f32"), Some(&[2][..]));
        assert_eq!(file.dtype("f32"), Some(StoredDType::F32));
        let f32_tensor = file.tensor("f32").unwrap();
        assert_eq!(f32_tensor.data, vec![1.0, -2.0]);
        let bf16_tensor = file.tensor("bf16").unwrap();
        assert_eq!(bf16_tensor.data, vec![1.0, -2.0]);
        assert_eq!(file.dtype("bf16"), Some(StoredDType::BF16));
        assert!(file.tensor("missing").is_err());
        let _ = std::fs::remove_file(path);
    }
}
