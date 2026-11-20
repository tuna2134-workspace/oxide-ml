# Benchmark report

Measurements are release builds on the reference CPU workspace. They are
wall-clock host timings. GPU results come from separately recorded A100 runs
(see [verification-report.md](verification-report.md)); CPU and GPU timings were
not collected on the same machine and are not directly comparable.

## Methodology

- `cargo bench -p oxide-ml --bench compiler` — graph compile/execute scenarios
  (small: 3 layers/width 16/batch 8; large: 16 layers/width 64/batch 32).
- `cargo bench -p oxide-ml --bench model` — MLP 784-256-10.
- `cargo bench -p oxide-ml --bench module_system` — construction, parameter
  discovery, module-list iteration, compiled inference over a `ModuleList`
  stack, and KV-cache state operations.
- Timings are averaged over the iterations in the harness; runs vary by machine.
- "Compile" is the optimizer + memory planner; "compiled run" excludes
  parameter upload. Parameter binding is a host clone in the reference backend.

## Compiler

| Graph | Scenario | Compile µs | Nodes | Exec µs | Peak B | Allocs |
|---|---|---:|---:|---:|---:|---:|
| small | no-opt/no-plan | 68.2 | 16 | 55.3 | 6,724 | 16 |
| small | full-opt/plan | 172.0 | 11 | 48.1 | 1,024 | 2 |
| large | no-opt/no-plan | 227.4 | 55 | 5,775.1 | 426,244 | 55 |
| large | full-opt/plan | 879.3 | 37 | 4,618.4 | 16,384 | 2 |

Optimization trades compile time for node count and peak memory; execution-time
deltas at this scale are run-dependent.

## Model API and compile cache

MLP 784-256-10, batch 64:

| Path | Time |
|---|---:|
| compile (cold) | 250 µs |
| compile (cached structural hit) | 10.1 µs |
| compiled inference | 2.8 ms/pass (8-core host) |
| `predict` (cached graph, parameter binding + run) | 6.1 ms/call |
| training step (forward + backward + SGD) | 72–86 ms/batch |

Batch scaling (compiled inference):

| Batch | µs/pass |
|---:|---:|
| 1 | 0.9–1.0 ms |
| 16 | 1.7–1.8 ms |
| 64 | 2.8 ms |
| 256 | 9.1 ms |

The generated model graph compiles to 3 nodes with 2 fusions
(`MatMulBias`, `AddRelu`) and a 128 KiB peak memory plan.

## Interpretation

- The cache removes compilation from the repeated-`predict` path (250 µs →
  10 µs); inference itself is dominated by the matmul and parameter cloning.
  Matmul now splits the contiguous output buffer across up to eight threads
  (deterministic: each output element keeps the same four-accumulator dot
  product), which roughly halves inference time on an 8-core host. This is a
  reference backend, not a tuned BLAS.
- The backward pass is roughly an order of magnitude slower than the forward
  pass; it has not been optimized.
- Small-graph compile overhead is tens to hundreds of microseconds; large-graph
  optimization is the expensive case.

## Gemma 4 E2B (CPU layer streaming)

Colab T4 VM, CPU execution, 12 GB RAM, 9.4 GB BF16 text checkpoint:

| Prompt | Tokens | Observed forward time | Peak RSS |
|---|---:|---:|---:|
| "Hello" | 1 | < 60 s | ≈ 6.2 GB |
| chat template + "Hello" | 26 | ≈ 1–2 min | ≈ 6.2 GB |

Each decoder layer is converted to f32, executed and freed; token/PLE rows are
read from the mmap and the LM head is chunked. Generation adds a full
recompute, so a second token roughly doubles the time. These are observed wall
times from the verification run, not a controlled benchmark.

## Not measured

- GPU kernel latency, GPU event time, multi-stream scaling, batch scaling on
  CUDA, fusion performance on CUDA (session expired; earlier T4 results are in
  the historical phase reports and are not comparable).
- CPU thread-pool/SIMD scaling (not implemented).

## Gemma 4 E2B on CUDA (A100)

End-to-end wall clock for one `Hello` forward, including loading the 9.3 GB
BF16 checkpoint, converting to f32, uploading ~18.6 GB of parameters, compiling
the graph and running: **52.7 s** (A100 40 GB). This is a one-shot check, not a
serving benchmark; CUDA generation has no KV cache yet.


## Gemma 4 E2B CUDA generation — "How to go to Paris?"

A100 40 GB, F32 text weights (4,628,569,379 parameters), greedy generation.
Each step recompiles the graph for the extended sequence and re-uploads the
18.6 GB parameter store (there is no CUDA KV cache yet), so the per-token time
is dominated by compilation/upload, not by the matmuls.

Raw prompt `"How to go to Paris?"` (6 tokens):

| Phase | Time |
|---|---:|
| load checkpoint + BF16→F32 | 19.665 s |
| prefill (compile + forward, 6 tokens) | 29.576 s |
| generated token 1 | 29.733 s |
| generated token 2 | 29.795 s |
| generated token 3 | 29.743 s |
| generated token 4 | 29.816 s |
| **4 generated tokens total** | **119.087 s** (≈29.77 s/token) |
| end-to-end (load + prefill + 4 tokens) | ≈168.3 s |

Chat-formatted prompt (31 tokens; greedy stopped at EOS after 2 tokens):

| Phase | Time |
|---|---:|
| load + F32 conversion | 19.554 s |
| prefill | 30.558 s |
| generated token 1 | 30.697 s |
| generated token 2 | 30.796 s |
| **2 generated tokens total** | **61.493 s** (≈30.75 s/token) |

Evidence: [gemma4-e2b-paris.log](gemma4-e2b-paris.log),
[gemma4-e2b-paris-chat.log](gemma4-e2b-paris-chat.log).

The generated tokens for the raw prompt were formatting tokens
(`<turn|>`/newline) before EOS; this measurement is about latency, not answer
quality. A CUDA KV cache plus resident parameters would remove the per-token
recompile/upload and is the next optimization.

## Module system (redesign)

8 residual blocks of 2 × `Linear(128, 128)` plus a `Linear(128, 128)` head,
assembled with `ModuleList` and a nested `#[model]` loop. Host: reference
workspace, release build.

| Measurement | Result |
|---|---:|
| module construction (8 blocks, 50 parameters) | 657 µs/model |
| `ModuleList` build (16 × 8×8 `Linear`) | 4.0 µs |
| parameter discovery (`visit_parameters`, 50 parameters) | 63.5 µs |
| build + `set_training` propagation | 920 µs/model |
| compiled forward, batch 8 | 10.3 ms/pass |
| graph compile (cold) | 869 µs |
| KV append, 256 tokens × 32 layers (8×128 heads/dim) | 14.5 ms (4.6 GB/s) |
| KV slice lookup, 32 layers | 0.46 µs |
| KV footprint, 256 tokens | 64.0 MiB |

Interpretation:

- Construction is dominated by parameter initialization (RNG), not module
  traversal; parameter discovery is a small fraction of a compile.
- A compiled forward of this shape is matmul-bound in the reference CPU backend;
  the module system adds only graph construction once.
- The KV cache copies host f32 data at 4.6 GB/s, which is memory-bandwidth
  bound. It is designed as the reusable reference state; a device-resident
  cache would upload once and reuse device buffers instead of copying per step.

## CUDA (A100, recorded separately)

- Full Gemma 4 E2B f32 (4.63 B parameters, 18.6 GB) forward after loading and
  upload: **52.7 s** end-to-end for one `Hello` prompt (one-shot check, not a
  serving benchmark).
- `Split`/`LogSoftmax` synthetic parity: `1.192e-7`; Compute Sanitizer
  memcheck 0 errors.
- Tiny MoE Gemma parity: `1.162e-6`; memcheck 0 errors.
- Dense tiny Gemma parity: `1.609e-6`.
- Full E2B generation without a KV cache (compile + run per token): prompt
  `"Hello"` prefill 29.6 s, each subsequent token ≈29.8 s; `"How to go to
  Paris?"` 4 generated tokens in 119.1 s. After the module/tied/state refactor,
  the `"Hello"` prefill is 29.553 s with identical top-5 logits
  ([gemma4-cuda-e2b-refactor.log](gemma4-cuda-e2b-refactor.log)). The A100
  KV-cache gap is the largest remaining performance item.
- Fixed-length resident decode (pad to `prompt + new_tokens`, compile once):
  19.9 s load + 29.3 s compile/upload, then 0.40 s per step on A100
  ([gemma4-e2b-paris-resident.log](gemma4-e2b-paris-resident.log)). It removes
  the 18.6 GB per-token upload but still computes the full sequence each step,
  so it is not a substitute for a KV cache.


## KV-cache comparison — "How to go to Paris?"

Prompt `"How to go to Paris?"` (6 tokens), greedy 4 tokens.

| Path | KV cache | Weights | Time |
|---|---|---|---|
| CPU layer streaming | **yes** (`prefill`/`decode`) | streamed per layer per token | **128.418 s** total (open + prefill + 4 tokens) |
| CUDA, rebuild per token | no | recompiled + 18.6 GB re-upload per token | prefill 29.576 s + 4 × ≈29.8 s = **168.3 s** end-to-end |
| CUDA, fixed-length resident | no (padded decode) | uploaded once | load 19.896 s + compile/upload 29.332 s + 5 runs **2.135 s** ≈ **51.4 s** end-to-end |

The CUDA resident run uses `max_len = 6 + 4 = 10`, pads trailing positions with
token 0 and reads logits at the last real position; causal masking makes the
padding irrelevant. It is **not** a KV cache, but it removes the per-token
recompile/upload and shows where a real CUDA cache would land: **≈0.40 s per
token** once weights are resident.

Evidence: [gemma4-e2b-paris-cpu-cache.log](gemma4-e2b-paris-cpu-cache.log),
[gemma4-e2b-paris-resident.log](gemma4-e2b-paris-resident.log),
[gemma4-e2b-paris.log](gemma4-e2b-paris.log).

A true CUDA KV cache is still unimplemented; the CPU path is the only cache
implementation today.
