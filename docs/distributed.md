# Distributed abstractions

**Status: single-process and in-process multi-worker collectives are
implemented; multi-process/multi-node execution is NOT IMPLEMENTED.**

The API exists so training code can be written against a mesh and a process
group without pretending that multi-GPU/multi-node execution works.

## API

```rust
use oxide_ml::prelude::*;

let mesh = DeviceMesh::new(vec![Device::Cpu], vec![1])?;
let group = Arc::new(SingleProcess::new());
let parallel = DataParallel::new(model, mesh, group)?;
```

- `DeviceMesh`: devices + logical dimensions, row-major `rank`/`coordinates`,
  size validation.
- `ProcessGroup`: `all_reduce`, `broadcast`, `all_gather`, `barrier`.
- `SingleProcess`: identity collectives (`all_reduce` is a no-op, `all_gather`
  returns one shard, `broadcast(root=0)` is a no-op).
- `LocalGroup`: in-process multi-worker collectives (sum/mean/max/min
  all-reduce, broadcast, all-gather) backed by a shared barrier and slots. Used
  by tests; each rank lives in its own thread.
- `DataParallel<M>`: wraps a module and a group. It rejects `world_size > 1`
  with `Unsupported` instead of silently running on one device.

## Not implemented

- Multi-process launch, rank assignment, rendezvous, and network transport.
- NCCL/Gloo or any real collective backend; `LocalGroup` is in-process only.
- Gradient all-reduce during training (there is no optimizer hook).
- Sharded parameters / tensor parallelism / pipeline parallelism.

These need a process launcher plus `Collective` implementations for CPU and
CUDA; the current module intentionally stops at the structural boundary.
