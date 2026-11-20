//! Reusable model execution state.
//!
//! Model definitions describe *what* to compute; execution state describes
//! *where* recurrent context lives. [`KvCache`] is the framework-owned,
//! backend-independent KV cache used by decoder-style models for prefill and
//! decode. It is deliberately independent of any architecture:
//!
//! - every layer has its own `(kv_heads, head_dim)` shape;
//! - a layer can declare `shared_with` another layer and then reads that
//!   layer's context instead of owning storage (Gemma-style shared KV);
//! - capacity is reserved and grown explicitly, with an error instead of a
//!   silent reallocation on overflow;
//! - state is owned per session, so concurrent sessions cannot alias one
//!   another (`KvCache` is `Send` and has no interior mutability).
//!
//! The cache stores host f32 data. Device residency belongs to the runtime that
//! consumes the cache, not to the model definition.

use oxide_ml_core::{DenseTensor, Error, Result};

/// Shape and sharing rule for one layer's KV storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvLayerConfig {
    pub kv_heads: usize,
    pub head_dim: usize,
    /// When set, this layer reads the KV rows of the referenced layer and does
    /// not own storage. The reference must point to an earlier non-shared
    /// layer.
    pub shared_with: Option<usize>,
}

impl KvLayerConfig {
    pub fn new(kv_heads: usize, head_dim: usize) -> Self {
        KvLayerConfig {
            kv_heads,
            head_dim,
            shared_with: None,
        }
    }

    pub fn shared_with(mut self, source: usize) -> Self {
        self.shared_with = Some(source);
        self
    }

    /// Elements in one token's K (or V) row.
    pub const fn row_size(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    pub const fn owns_storage(&self) -> bool {
        self.shared_with.is_none()
    }
}

/// Full cache layout plus initial row capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheConfig {
    pub capacity: usize,
    pub layers: Vec<KvLayerConfig>,
}

impl KvCacheConfig {
    pub fn new(capacity: usize, layers: Vec<KvLayerConfig>) -> Self {
        KvCacheConfig { capacity, layers }
    }

    /// One uniform layout for every layer.
    pub fn uniform(layers: usize, kv_heads: usize, head_dim: usize, capacity: usize) -> Self {
        KvCacheConfig {
            capacity,
            layers: vec![KvLayerConfig::new(kv_heads, head_dim); layers],
        }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    fn validate(&self) -> Result<()> {
        if self.layers.is_empty() {
            return Err(Error::invalid_shape(
                "KvCache",
                "cache needs at least one layer",
            ));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.kv_heads == 0 || layer.head_dim == 0 {
                return Err(Error::invalid_shape(
                    "KvCache",
                    format!("layer {index} has an empty KV shape"),
                ));
            }
            if let Some(source) = layer.shared_with {
                if source >= index {
                    return Err(Error::invalid_shape(
                        "KvCache",
                        format!(
                            "layer {index} shares KV with layer {source}; the source must be \
                             an earlier layer"
                        ),
                    ));
                }
                if self.layers[source].shared_with.is_some() {
                    return Err(Error::invalid_shape(
                        "KvCache",
                        format!(
                            "layer {index} shares KV with layer {source}, which is itself \
                             shared; point at the owning layer instead"
                        ),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Slots {
    keys: DenseTensor<f32>,
    values: DenseTensor<f32>,
    length: usize,
}

/// Per-session KV cache state.
#[derive(Clone, Debug)]
pub struct KvCache {
    config: KvCacheConfig,
    slots: Vec<Option<Slots>>,
}

impl KvCache {
    pub fn new(config: KvCacheConfig) -> Result<Self> {
        config.validate()?;
        let mut slots = Vec::with_capacity(config.layers.len());
        for layer in &config.layers {
            if layer.owns_storage() {
                let shape = [config.capacity, layer.kv_heads, layer.head_dim];
                slots.push(Some(Slots {
                    keys: DenseTensor::zeros(shape),
                    values: DenseTensor::zeros(shape),
                    length: 0,
                }));
            } else {
                slots.push(None);
            }
        }
        Ok(KvCache { config, slots })
    }

    pub fn config(&self) -> &KvCacheConfig {
        &self.config
    }

    /// Rows reserved per layer.
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Number of cached tokens for one layer (following shared links).
    pub fn len(&self, layer: usize) -> Result<usize> {
        let owner = self.owner(layer)?;
        Ok(self.slots[owner]
            .as_ref()
            .map(|slots| slots.length)
            .unwrap_or(0))
    }

    /// Longest cached sequence across layers.
    pub fn sequence_length(&self) -> usize {
        self.slots
            .iter()
            .filter_map(|slots| slots.as_ref().map(|slots| slots.length))
            .max()
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.sequence_length() == 0
    }

    /// Rows still available before [`reserve`](Self::reserve) is required.
    pub fn remaining(&self) -> usize {
        self.capacity().saturating_sub(self.sequence_length())
    }

    /// Resolve the layer that owns storage for `layer`.
    pub fn owner(&self, layer: usize) -> Result<usize> {
        let entry = self.config.layers.get(layer).ok_or_else(|| {
            Error::invalid_shape("KvCache", format!("layer {layer} is out of range"))
        })?;
        Ok(entry.shared_with.unwrap_or(layer))
    }

    /// Grow the reserved capacity, preserving cached rows.
    pub fn reserve(&mut self, capacity: usize) -> Result<()> {
        if capacity <= self.config.capacity {
            return Ok(());
        }
        for (index, layer) in self.config.layers.iter().enumerate() {
            if !layer.owns_storage() {
                continue;
            }
            let Some(slots) = self.slots[index].as_mut() else {
                continue;
            };
            let shape = [capacity, layer.kv_heads, layer.head_dim];
            let keep = slots.length * layer.row_size();
            let mut keys = DenseTensor::zeros(shape);
            let mut values = DenseTensor::zeros(shape);
            keys.data_mut()[..keep].copy_from_slice(&slots.keys.data()[..keep]);
            values.data_mut()[..keep].copy_from_slice(&slots.values.data()[..keep]);
            slots.keys = keys;
            slots.values = values;
        }
        self.config.capacity = capacity;
        Ok(())
    }

    fn ensure_capacity(&mut self, layer: usize, tokens: usize) -> Result<()> {
        let needed = self.len(layer)? + tokens;
        if needed <= self.config.capacity {
            return Ok(());
        }
        let capacity = needed.next_power_of_two().max(self.config.capacity * 2);
        self.reserve(capacity)
    }

    /// Append one token of K/V to a storage-owning layer.
    pub fn append(&mut self, layer: usize, keys: &[f32], values: &[f32]) -> Result<()> {
        self.append_many(layer, keys, values, 1)
    }

    /// Append `tokens` rows of K/V to a storage-owning layer.
    pub fn append_many(
        &mut self,
        layer: usize,
        keys: &[f32],
        values: &[f32],
        tokens: usize,
    ) -> Result<()> {
        let config = *self.config.layers.get(layer).ok_or_else(|| {
            Error::invalid_shape("KvCache", format!("layer {layer} is out of range"))
        })?;
        if !config.owns_storage() {
            return Err(Error::invalid_graph(format!(
                "layer {layer} shares KV storage with layer {}; append to the owning layer",
                config.shared_with.unwrap_or(layer)
            )));
        }
        let expected = config.row_size() * tokens;
        if keys.len() != expected || values.len() != expected {
            return Err(Error::invalid_shape(
                "KvCache::append",
                format!(
                    "layer {layer} expects {expected} K and V elements for {tokens} token(s), \
                     got {} and {}",
                    keys.len(),
                    values.len()
                ),
            ));
        }
        self.ensure_capacity(layer, tokens)?;
        let slots = self.slots[layer]
            .as_mut()
            .ok_or_else(|| Error::invalid_graph(format!("layer {layer} has no KV storage")))?;
        let start = slots.length * config.row_size();
        slots.keys.data_mut()[start..start + expected].copy_from_slice(keys);
        slots.values.data_mut()[start..start + expected].copy_from_slice(values);
        slots.length += tokens;
        Ok(())
    }

    /// Cached K rows for `layer`, flattened as `[len, kv_heads, head_dim]`.
    pub fn keys(&self, layer: usize) -> Result<&[f32]> {
        let owner = self.owner(layer)?;
        let config = self.config.layers[owner];
        let slots = self.slots[owner]
            .as_ref()
            .ok_or_else(|| Error::invalid_graph(format!("layer {owner} has no KV storage")))?;
        Ok(&slots.keys.data()[..slots.length * config.row_size()])
    }

    /// Cached V rows for `layer`, flattened as `[len, kv_heads, head_dim]`.
    pub fn values(&self, layer: usize) -> Result<&[f32]> {
        let owner = self.owner(layer)?;
        let config = self.config.layers[owner];
        let slots = self.slots[owner]
            .as_ref()
            .ok_or_else(|| Error::invalid_graph(format!("layer {owner} has no KV storage")))?;
        Ok(&slots.values.data()[..slots.length * config.row_size()])
    }

    /// Drop everything but keep the reserved capacity (start a new session).
    pub fn reset(&mut self) {
        for slots in self.slots.iter_mut().flatten() {
            slots.length = 0;
        }
    }

    /// Rewind all layers to at most `length` tokens.
    pub fn truncate(&mut self, length: usize) {
        for slots in self.slots.iter_mut().flatten() {
            slots.length = slots.length.min(length);
        }
    }
}

/// Execution state that a runtime owns per session.
///
/// Implementations may be backend specific; [`KvCache`] is the reference
/// host implementation.
pub trait ModelState: Send {
    /// Return to the empty state while keeping allocated capacity.
    fn reset(&mut self);

    /// Number of tokens currently represented by the state.
    fn sequence_length(&self) -> usize;
}

impl ModelState for KvCache {
    fn reset(&mut self) {
        KvCache::reset(self);
    }

    fn sequence_length(&self) -> usize {
        KvCache::sequence_length(self)
    }
}
