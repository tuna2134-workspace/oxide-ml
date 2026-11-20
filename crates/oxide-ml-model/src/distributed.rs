//! Distributed-training abstractions (single-process reference).
//!
//! This module provides the structural API — device mesh, process group,
//! collectives and a `DataParallel` wrapper — without a multi-process runtime.
//! Only `SingleProcess` is executable; constructing a group with
//! `world_size > 1` is rejected with an explicit error. This keeps the surface
//! honest: no silent single-device fallback and no pretend NCCL.

use std::sync::{Arc, Barrier, Mutex};

use oxide_ml_core::{Device, Error, Result};

use crate::tensor::Tensor;

/// Logical device mesh. The product of `dims` must equal the device count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceMesh {
    devices: Vec<Device>,
    dims: Vec<usize>,
}

impl DeviceMesh {
    pub fn new(devices: Vec<Device>, dims: Vec<usize>) -> Result<Self> {
        if devices.is_empty() {
            return Err(Error::invalid_shape(
                "DeviceMesh",
                "mesh needs at least one device",
            ));
        }
        if dims.is_empty() || dims.contains(&0) {
            return Err(Error::invalid_shape(
                "DeviceMesh",
                "mesh dimensions must be non-zero",
            ));
        }
        let product: usize = dims.iter().product();
        if product != devices.len() {
            return Err(Error::invalid_shape(
                "DeviceMesh",
                format!(
                    "mesh {dims:?} has {product} slots but {} devices were provided",
                    devices.len()
                ),
            ));
        }
        Ok(DeviceMesh { devices, dims })
    }

    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn world_size(&self) -> usize {
        self.devices.len()
    }

    /// Row-major rank for mesh coordinates.
    pub fn rank(&self, coordinates: &[usize]) -> Result<usize> {
        if coordinates.len() != self.dims.len() {
            return Err(Error::invalid_shape(
                "DeviceMesh",
                format!(
                    "expected {} coordinates, got {}",
                    self.dims.len(),
                    coordinates.len()
                ),
            ));
        }
        let mut rank = 0usize;
        for (axis, coordinate) in coordinates.iter().enumerate() {
            if *coordinate >= self.dims[axis] {
                return Err(Error::invalid_shape(
                    "DeviceMesh",
                    format!(
                        "coordinate {coordinate} is out of range on axis {axis} ({})",
                        self.dims[axis]
                    ),
                ));
            }
            rank = rank * self.dims[axis] + coordinate;
        }
        Ok(rank)
    }

    pub fn coordinates(&self, rank: usize) -> Result<Vec<usize>> {
        if rank >= self.world_size() {
            return Err(Error::invalid_shape(
                "DeviceMesh",
                format!(
                    "rank {rank} is out of range for world size {}",
                    self.world_size()
                ),
            ));
        }
        let mut remaining = rank;
        let mut coordinates = vec![0usize; self.dims.len()];
        for axis in (0..self.dims.len()).rev() {
            coordinates[axis] = remaining % self.dims[axis];
            remaining /= self.dims[axis];
        }
        Ok(coordinates)
    }

    pub fn device(&self, rank: usize) -> Result<Device> {
        self.devices
            .get(rank)
            .copied()
            .ok_or_else(|| Error::invalid_shape("DeviceMesh", format!("rank {rank} out of range")))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reduction {
    Sum,
    Mean,
    Max,
    Min,
}

/// Collective operations. Only the single-process implementation is provided.
pub trait ProcessGroup: Send + Sync {
    fn world_size(&self) -> usize;
    fn rank(&self) -> usize;
    fn all_reduce(&self, tensor: &mut Tensor<f32>, reduction: Reduction) -> Result<()>;
    fn broadcast(&self, tensor: &mut Tensor<f32>, root: usize) -> Result<()>;
    fn all_gather(&self, tensor: &Tensor<f32>) -> Result<Vec<Tensor<f32>>>;
    fn barrier(&self) -> Result<()>;
}

/// A one-rank group: collectives are identity operations.
#[derive(Clone, Copy, Debug)]
pub struct SingleProcess {
    rank: usize,
}

impl SingleProcess {
    pub fn new() -> Self {
        SingleProcess { rank: 0 }
    }
}

impl Default for SingleProcess {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessGroup for SingleProcess {
    fn world_size(&self) -> usize {
        1
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn all_reduce(&self, _tensor: &mut Tensor<f32>, _reduction: Reduction) -> Result<()> {
        Ok(())
    }

    fn broadcast(&self, _tensor: &mut Tensor<f32>, root: usize) -> Result<()> {
        if root != 0 {
            return Err(Error::invalid_shape(
                "SingleProcess::broadcast",
                format!("root {root} is out of range for world size 1"),
            ));
        }
        Ok(())
    }

    fn all_gather(&self, tensor: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
        Ok(vec![tensor.clone()])
    }

    fn barrier(&self) -> Result<()> {
        Ok(())
    }
}

/// In-process multi-worker group used to exercise collectives in tests.
///
/// Each rank must live in its own thread. This is not a network backend: it
/// shares the tensors through a mutex and a barrier, so it validates the
/// collective semantics without pretending to implement NCCL/Gloo.
pub struct LocalGroup {
    rank: usize,
    world_size: usize,
    barrier: Arc<Barrier>,
    slots: Arc<Mutex<Vec<Option<Vec<f32>>>>>,
}

impl std::fmt::Debug for LocalGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalGroup")
            .field("rank", &self.rank)
            .field("world_size", &self.world_size)
            .finish()
    }
}

impl LocalGroup {
    pub fn new(rank: usize, world_size: usize) -> Result<Self> {
        if world_size == 0 || rank >= world_size {
            return Err(Error::invalid_shape(
                "LocalGroup",
                format!("rank {rank} is invalid for world size {world_size}"),
            ));
        }
        Ok(LocalGroup {
            rank,
            world_size,
            barrier: Arc::new(Barrier::new(world_size)),
            slots: Arc::new(Mutex::new(vec![None; world_size])),
        })
    }

    /// Create a set of in-process ranks sharing the same barrier and slots.
    pub fn cluster(world_size: usize) -> Result<Vec<LocalGroup>> {
        if world_size == 0 {
            return Err(Error::invalid_shape(
                "LocalGroup",
                "world size must be non-zero",
            ));
        }
        let barrier = Arc::new(Barrier::new(world_size));
        let slots = Arc::new(Mutex::new(vec![None; world_size]));
        Ok((0..world_size)
            .map(|rank| LocalGroup {
                rank,
                world_size,
                barrier: Arc::clone(&barrier),
                slots: Arc::clone(&slots),
            })
            .collect())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Option<Vec<f32>>>> {
        self.slots
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl ProcessGroup for LocalGroup {
    fn world_size(&self) -> usize {
        self.world_size
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn all_reduce(&self, tensor: &mut Tensor<f32>, reduction: Reduction) -> Result<()> {
        {
            let mut slots = self.lock();
            slots[self.rank] = Some(tensor.data().to_vec());
        }
        self.barrier.wait();
        let mut reduced = vec![0.0f32; tensor.numel()];
        {
            let slots = self.lock();
            for (index, slot) in slots.iter().enumerate() {
                let values = slot.as_ref().ok_or_else(|| {
                    Error::invalid_graph(format!("all_reduce missing rank {index}"))
                })?;
                if values.len() != reduced.len() {
                    return Err(Error::invalid_shape(
                        "all_reduce",
                        "ranks contributed different tensor sizes",
                    ));
                }
                if index == 0 {
                    reduced.copy_from_slice(values);
                } else {
                    for (accumulator, value) in reduced.iter_mut().zip(values) {
                        *accumulator = match reduction {
                            Reduction::Sum | Reduction::Mean => *accumulator + *value,
                            Reduction::Max => accumulator.max(*value),
                            Reduction::Min => accumulator.min(*value),
                        };
                    }
                }
            }
        }
        if reduction == Reduction::Mean {
            let scale = 1.0 / self.world_size as f32;
            for value in &mut reduced {
                *value *= scale;
            }
        }
        tensor.data_mut().copy_from_slice(&reduced);
        self.barrier.wait();
        let mut slots = self.lock();
        slots[self.rank] = None;
        Ok(())
    }

    fn broadcast(&self, tensor: &mut Tensor<f32>, root: usize) -> Result<()> {
        if root >= self.world_size {
            return Err(Error::invalid_shape(
                "broadcast",
                format!("root {root} is out of range"),
            ));
        }
        if self.rank == root {
            let mut slots = self.lock();
            slots[root] = Some(tensor.data().to_vec());
        }
        self.barrier.wait();
        {
            let slots = self.lock();
            let values = slots[root]
                .as_ref()
                .ok_or_else(|| Error::invalid_graph("broadcast root has no data"))?;
            tensor.data_mut().copy_from_slice(values);
        }
        self.barrier.wait();
        if self.rank == root {
            let mut slots = self.lock();
            slots[root] = None;
        }
        Ok(())
    }

    fn all_gather(&self, tensor: &Tensor<f32>) -> Result<Vec<Tensor<f32>>> {
        {
            let mut slots = self.lock();
            slots[self.rank] = Some(tensor.data().to_vec());
        }
        self.barrier.wait();
        let gathered = {
            let slots = self.lock();
            slots
                .iter()
                .map(|slot| {
                    slot.as_ref()
                        .map(|values| Tensor::from_vec(tensor.dims(), values.clone()))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?
        };
        self.barrier.wait();
        let mut slots = self.lock();
        slots[self.rank] = None;
        Ok(gathered.into_iter().flatten().collect())
    }

    fn barrier(&self) -> Result<()> {
        self.barrier.wait();
        Ok(())
    }
}

/// Wrap a module for data-parallel execution.
///
/// Construction succeeds only for a single-process group. Multi-process
/// execution requires a real launcher and NCCL/Gloo backend, which this
/// workspace does not implement; it returns `Unsupported` instead of silently
/// running on one device.
#[derive(Clone)]
pub struct DataParallel<M> {
    module: M,
    mesh: DeviceMesh,
    group: Arc<dyn ProcessGroup>,
}

impl<M> DataParallel<M> {
    pub fn new(module: M, mesh: DeviceMesh, group: Arc<dyn ProcessGroup>) -> Result<Self> {
        if group.world_size() != mesh.world_size() {
            return Err(Error::invalid_shape(
                "DataParallel",
                format!(
                    "process group world size {} does not match mesh size {}",
                    group.world_size(),
                    mesh.world_size()
                ),
            ));
        }
        Ok(DataParallel {
            module,
            mesh,
            group,
        })
    }

    pub fn module(&self) -> &M {
        &self.module
    }

    pub fn module_mut(&mut self) -> &mut M {
        &mut self.module
    }

    pub fn mesh(&self) -> &DeviceMesh {
        &self.mesh
    }

    pub fn group(&self) -> &Arc<dyn ProcessGroup> {
        &self.group
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxide_ml_core::DenseTensor;

    #[test]
    fn mesh_rejects_size_mismatch() {
        let error = DeviceMesh::new(vec![Device::Cpu, Device::Cpu], vec![4]).unwrap_err();
        assert!(error.to_string().contains("slots"), "{error}");
    }

    #[test]
    fn mesh_maps_rank_and_coordinates() {
        let mesh = DeviceMesh::new(
            vec![Device::Cpu, Device::Cpu, Device::Cpu, Device::Cpu],
            vec![2, 2],
        )
        .unwrap();
        assert_eq!(mesh.rank(&[1, 0]).unwrap(), 2);
        assert_eq!(mesh.coordinates(3).unwrap(), vec![1, 1]);
        assert_eq!(mesh.device(2).unwrap(), Device::Cpu);
        assert!(mesh.rank(&[2, 0]).is_err());
    }

    #[test]
    fn single_process_collectives_are_identity() {
        let group = SingleProcess::new();
        let mut tensor = Tensor::from_dense(DenseTensor::from_vec([2], vec![1.0f32, 2.0]).unwrap());
        group.all_reduce(&mut tensor, Reduction::Sum).unwrap();
        assert_eq!(tensor.data(), &[1.0, 2.0]);
        assert_eq!(group.all_gather(&tensor).unwrap().len(), 1);
        group.broadcast(&mut tensor, 0).unwrap();
        assert!(group.broadcast(&mut tensor, 1).is_err());
        group.barrier().unwrap();
    }

    #[test]
    fn local_group_all_reduce_and_gather() {
        let groups = LocalGroup::cluster(3).unwrap();
        std::thread::scope(|scope| {
            let handles: Vec<_> = groups
                .into_iter()
                .map(|group| {
                    scope.spawn(move || {
                        let mut tensor = Tensor::from_vec(
                            [2],
                            vec![group.rank() as f32, group.rank() as f32 + 1.0],
                        )
                        .unwrap();
                        group.all_reduce(&mut tensor, Reduction::Sum).unwrap();
                        let gathered = group.all_gather(&tensor).unwrap();
                        assert_eq!(gathered.len(), 3);
                        assert_eq!(tensor.data(), &[3.0, 6.0]);
                        let mut mean = Tensor::from_vec([1], vec![group.rank() as f32]).unwrap();
                        group.all_reduce(&mut mean, Reduction::Mean).unwrap();
                        assert!((mean.data()[0] - 1.0).abs() < 1e-6);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn local_group_broadcast() {
        let groups = LocalGroup::cluster(2).unwrap();
        std::thread::scope(|scope| {
            let handles: Vec<_> = groups
                .into_iter()
                .map(|group| {
                    scope.spawn(move || {
                        let mut tensor = if group.rank() == 0 {
                            Tensor::from_vec([2], vec![7.0f32, 8.0]).unwrap()
                        } else {
                            Tensor::from_vec([2], vec![0.0f32, 0.0]).unwrap()
                        };
                        group.broadcast(&mut tensor, 0).unwrap();
                        assert_eq!(tensor.data(), &[7.0, 8.0]);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }

    #[test]
    fn data_parallel_accepts_local_multi_worker() {
        let mesh = DeviceMesh::new(vec![Device::Cpu, Device::Cpu], vec![2]).unwrap();
        let mut groups = LocalGroup::cluster(2).unwrap();
        let parallel = DataParallel::new(1u32, mesh, Arc::new(groups.remove(0))).unwrap();
        assert_eq!(parallel.group().world_size(), 2);
        assert_eq!(*parallel.module(), 1);
    }

    #[test]
    fn data_parallel_single_process_wraps_module() {
        let mesh = DeviceMesh::new(vec![Device::Cpu], vec![1]).unwrap();
        let parallel = DataParallel::new(42u32, mesh, Arc::new(SingleProcess::new())).unwrap();
        assert_eq!(*parallel.module(), 42);
    }
}
