//! Datasets and batch iteration.

use oxide_ml_core::{Error, Result};

use crate::runtime::InputSpec;
use crate::tensor::{Tensor, stack};

/// One training example.
#[derive(Clone, Debug)]
pub struct Sample {
    pub inputs: Vec<Tensor<f32>>,
    pub targets: Vec<Tensor<f32>>,
}

impl Sample {
    pub fn new(inputs: Vec<Tensor<f32>>, targets: Vec<Tensor<f32>>) -> Self {
        Sample { inputs, targets }
    }
}

/// Indexable collection of samples.
///
/// `input_specs` / `target_specs` describe one sample (no batch dimension). The
/// trainer prepends a symbolic batch dimension when it builds the graph.
pub trait Dataset {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn input_specs(&self) -> Vec<InputSpec>;

    fn target_specs(&self) -> Vec<InputSpec>;

    fn get(&self, index: usize) -> Result<Sample>;
}

/// Dataset backed by one tensor per input and one per target.
///
/// Every tensor has a leading sample dimension: an `inputs` tensor of shape
/// `[N, 784]` yields samples of shape `[784]`, and a `targets` tensor of shape
/// `[N]` yields scalar targets.
pub struct TensorDataset {
    inputs: Vec<Tensor<f32>>,
    targets: Vec<Tensor<f32>>,
    len: usize,
}

impl TensorDataset {
    pub fn new(inputs: Vec<Tensor<f32>>, targets: Vec<Tensor<f32>>) -> Result<Self> {
        if inputs.is_empty() {
            return Err(Error::invalid_shape(
                "TensorDataset",
                "at least one input tensor is required",
            ));
        }
        let mut len: Option<usize> = None;
        for (index, tensor) in inputs.iter().enumerate() {
            if tensor.rank() == 0 {
                return Err(Error::invalid_shape(
                    "TensorDataset",
                    format!("input {index} has no sample dimension"),
                ));
            }
            let samples = tensor.dims()[0];
            match len {
                None => len = Some(samples),
                Some(expected) if expected == samples => {}
                Some(expected) => {
                    return Err(Error::invalid_shape(
                        "TensorDataset",
                        format!("input {index} has {samples} samples but input 0 has {expected}"),
                    ));
                }
            }
        }
        for (index, tensor) in targets.iter().enumerate() {
            if tensor.rank() == 0 {
                return Err(Error::invalid_shape(
                    "TensorDataset",
                    format!("target {index} has no sample dimension"),
                ));
            }
            let samples = tensor.dims()[0];
            if samples != len.unwrap_or(0) {
                return Err(Error::invalid_shape(
                    "TensorDataset",
                    format!(
                        "target {index} has {samples} samples but inputs have {}",
                        len.unwrap_or(0)
                    ),
                ));
            }
        }
        Ok(TensorDataset {
            inputs,
            targets,
            len: len.unwrap_or(0),
        })
    }

    /// Convenience constructor for one input / one target.
    pub fn from_tensors(inputs: Tensor<f32>, targets: Tensor<f32>) -> Result<Self> {
        Self::new(vec![inputs], vec![targets])
    }

    pub fn inputs(&self) -> &[Tensor<f32>] {
        &self.inputs
    }

    pub fn targets(&self) -> &[Tensor<f32>] {
        &self.targets
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.len
    }

    fn input_specs(&self) -> Vec<InputSpec> {
        self.inputs
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                InputSpec::unbatched(
                    format!("input{index}"),
                    tensor.dims().into_iter().skip(1).collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    fn target_specs(&self) -> Vec<InputSpec> {
        self.targets
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                InputSpec::unbatched(
                    format!("target{index}"),
                    tensor.dims().into_iter().skip(1).collect::<Vec<_>>(),
                )
            })
            .collect()
    }

    fn get(&self, index: usize) -> Result<Sample> {
        if index >= self.len {
            return Err(Error::execution(format!(
                "sample index {index} out of range for dataset of {}",
                self.len
            )));
        }
        let inputs = self
            .inputs
            .iter()
            .map(|tensor| tensor.row(index))
            .collect::<Result<Vec<_>>>()?;
        let targets = self
            .targets
            .iter()
            .map(|tensor| tensor.row(index))
            .collect::<Result<Vec<_>>>()?;
        Ok(Sample { inputs, targets })
    }
}

impl std::fmt::Debug for TensorDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorDataset")
            .field("samples", &self.len)
            .field("inputs", &self.inputs.len())
            .field("targets", &self.targets.len())
            .finish()
    }
}

/// Deterministic batch iterator.
#[derive(Clone, Debug)]
pub struct DataLoader {
    batch_size: usize,
    shuffle: bool,
    seed: u64,
    drop_last: bool,
}

impl DataLoader {
    pub fn new(batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(Error::invalid_shape("DataLoader", "batch_size must be > 0"));
        }
        Ok(DataLoader {
            batch_size,
            shuffle: false,
            seed: 0,
            drop_last: false,
        })
    }

    pub fn shuffle(mut self, seed: u64) -> Self {
        self.shuffle = true;
        self.seed = seed;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn batches(&self, samples: usize, epoch: u64) -> Vec<Vec<usize>> {
        let mut indices: Vec<usize> = (0..samples).collect();
        if self.shuffle {
            let mut rng = oxide_ml_core::Rng::new(self.seed.wrapping_add(epoch));
            rng.shuffle(&mut indices);
        }
        let mut batches = Vec::new();
        let mut start = 0;
        while start < indices.len() {
            let end = (start + self.batch_size).min(indices.len());
            if self.drop_last && end - start < self.batch_size {
                break;
            }
            batches.push(indices[start..end].to_vec());
            start = end;
        }
        batches
    }
}

/// Graph-ready batch: one stacked tensor per input and per target.
pub type Collated = (Vec<Tensor<f32>>, Vec<Tensor<f32>>);

/// Stack a batch of samples into graph-ready tensors.
pub fn collate(samples: &[Sample]) -> Result<Collated> {
    let first = samples
        .first()
        .ok_or_else(|| Error::execution("cannot collate an empty batch"))?;
    for (index, sample) in samples.iter().enumerate() {
        if sample.inputs.len() != first.inputs.len() {
            return Err(Error::execution(format!(
                "sample {index} has {} inputs but sample 0 has {}",
                sample.inputs.len(),
                first.inputs.len()
            )));
        }
        if sample.targets.len() != first.targets.len() {
            return Err(Error::execution(format!(
                "sample {index} has {} targets but sample 0 has {}",
                sample.targets.len(),
                first.targets.len()
            )));
        }
    }
    let inputs = (0..first.inputs.len())
        .map(|position| {
            let column: Vec<Tensor<f32>> = samples
                .iter()
                .map(|sample| sample.inputs[position].clone())
                .collect();
            stack(&column)
        })
        .collect::<Result<Vec<_>>>()?;
    let targets = (0..first.targets.len())
        .map(|position| {
            let column: Vec<Tensor<f32>> = samples
                .iter()
                .map(|sample| sample.targets[position].clone())
                .collect();
            stack(&column)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((inputs, targets))
}
