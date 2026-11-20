# Safetensors reader, writer and model export

`oxide-ml-safetensors` owns the file format; `oxide-ml` adds model-level export
and import helpers. Neither depends on a model architecture.

## Crate API

| Item | Purpose |
| --- | --- |
| `SafeTensorsFile::open` | mmap reader; `tensor`, `shape`, `dtype`, `metadata`, `names` |
| `StoredDType` | `F32`, `F16`, `BF16`, `I64`, `I32`, `U8`, `BOOL` |
| `TensorRecord` | one named tensor with raw bytes plus `f32`/`f16`/`bf16` constructors |
| `SafeTensorsWriter` | metadata + named tensors; `to_bytes`, `write` |
| `save_module_safetensors` | traverses a `Module` tree and writes canonical parameters |
| `save_module_safetensors_with_metadata` | the same with extra string metadata |
| `load_module_safetensors` | fills a module by stable parameter name; resolves aliases |
| `save_parameter_store_safetensors` | writes an authoritative `ParameterStore` |
| `module_tensors` | name → tensor map for tooling |

## Writer behaviour

- Tensors are written in deterministic name order; each data section starts on
  an 8-byte boundary and offsets are recorded in the header.
- The header is padded with spaces to an 8-byte boundary, as the format
  requires.
- `shape` must match the byte count; duplicate names and reserved metadata keys
  are rejected.
- Writes go to a temporary file in the same directory and are atomically
  renamed, so a failed export never truncates the previous checkpoint.
- Files are standard safetensors: `safetensors`, PyTorch, NumPy and
  HuggingFace tooling can read them.

## Model export

```rust
use oxide_ml::prelude::*;
use oxide_ml::{save_module_safetensors_with_metadata, load_module_safetensors};

save_module_safetensors_with_metadata(
    &model,
    "model.safetensors",
    &[("model.classes", "cat,dog"), ("model.input_size", "96")],
)?;

let report = load_module_safetensors(&mut model, "model.safetensors")?;
assert!(report.is_exact());
```

Names are the same dot paths used by parameter discovery
(`blocks.0.0.norm.conv.weight`, `head.norm.bn.gamma`, ...). Tied storage is
written once under the canonical name; the other paths are recorded as
`alias.<path>=<canonical>` metadata and resolved on load.

Metadata written by the model helpers:

| Key | Value |
| --- | --- |
| `format` / `version` | `oxide-ml.safetensors` / `1` |
| `module.type` | `Module::module_type` |
| `module.parameters` / `module.tensors` | discovered paths / written tensors |
| `role.<path>` | `trainable`, `frozen` or `buffer` |
| `alias.<path>` | canonical path |

## Loading and limitations

- Loading validates names and shapes; missing/unexpected tensors are reported,
  not silently ignored. `LoadReport::is_exact` covers both.
- Shared storage (`Arc<M>`, tied heads) cannot be written back through
  `visit_tensors_mut`; load the canonical module before tying, or use the
  metadata alias to resolve the canonical tensor. The report counts loaded
  tensors and aliases separately.
- The reader converts F16/BF16 to host f32; the writer accepts F32/F16/BF16 and
  integer payloads.
- No streaming/mmap writer yet: the file is assembled in memory before the
  atomic write. The reader is mmap-based, so multi-GB checkpoints are not copied
  twice on load.

## Tests

- `crates/oxide-ml-safetensors/tests/writer.rs`: layout, dtypes, metadata,
  duplicate rejection, shape validation.
- `tests/safetensors_export.rs`: module round trip, tied alias metadata,
  parameter-store export.
