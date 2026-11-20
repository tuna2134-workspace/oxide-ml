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


### CUDA KV cache decode (A100)

`"How to go to Paris?"`, 6-token prompt, 4 generated tokens. Prefill runs on
the CPU layer-streaming cache; decode runs on CUDA with resident parameters and
per-layer K/V caches (`gemma_e2b_kvcache`). Each step rebuilds only the graph
(no weight upload).

| Phase | Time |
|---|---:|
| load + BF16→F32 | 20.384 s |
| CPU KV prefill (6 tokens) | 37.846 s |
| CUDA compile + weight upload (once) | 29.693 s |
| generated token 1 (past 6) | 0.290 s |
| generated token 2 (past 7) | 0.294 s |
| generated token 3 (past 8) | 0.295 s |
| generated token 4 (past 9) | 0.295 s |
| **4 generated tokens total** | **1.188 s** (≈0.297 s/token) |
| end-to-end (load + CPU prefill + upload + decode) | ≈89.1 s |

The CPU prefill is the remaining slow part; a GPU prefill with K/V outputs would
remove it. The decode path itself matches the fixed-length resident run
(≈0.40 s/token) and is ~100× faster than the cache-less CUDA loop.

## Declarative transformer components (architecture revision)

`cargo bench -p oxide-ml --bench module_system`, release build, reference CPU
workspace. The transformer is 4 decoder layers of `RmsNorm` +
`MultiHeadAttention` + `GatedMlp` inside a `ModuleList`, driven by a `#[model]`
loop.

| Measurement | Result |
|---|---:|
| model construction (68 parameters) | 931 µs/model |
| parameter discovery | 119 µs |
| graph compile (cold) | 2.07 ms |
| compiled forward, batch 4, sequence 8 | 18.2 ms/pass |
| graph size | 151 nodes |
| `ModuleList` stack (8 blocks, width 128) construction | 657 µs |
| same stack: parameter discovery | 63–65 µs |
| same stack: compiled forward, batch 8 | 9.9–11.1 ms/pass |
| KV append, 256 tokens × 32 layers | 13.7–14.3 ms (4.7–4.9 GB/s) |
| KV slice lookup, 32 layers | 0.33–0.40 µs |

Interpretation:

- Declarative composition adds graph-construction work only; the compiled graph
  is the same SSA IR as a hand-built model (the transformer example is 151
  nodes). Construction is dominated by parameter initialization.
- Forward time is dominated by the reference CPU matmul/attention loops, not by
  module dispatch; the timings vary by several percent between runs on this
  host, so no speed-up is claimed relative to the pre-revision implementation.
- The KV cache copies host f32 data at memory bandwidth and is the reusable
  reference state, not a tuned device cache.

## GPU re-measurement (Tesla T4, current source)

Fresh T4 session with the current declarative/runtime revision. Full log:
[t4-gpu-bench.log](t4-gpu-bench.log), raw JSON:
[phase7f19-benchmark-results-t4.json](phase7f19-benchmark-results-t4.json).

GEMM (`gemm_bench`, microseconds, max error 0):

| Size | CPU | CUDA naive | CUDA tiled |
|---:|---:|---:|---:|
| 64 | 86.0 | 43.5 | 18.6 |
| 128 | 615.6 | 70.2 | 26.3 |
| 256 | 3,663.9 | 228.4 | 87.5 |
| 512 | 29,840 | 1,569.3 | 566.2 |

Previous T4 GEMM numbers (earlier milestone) were 1.32 ms naive / 0.46 ms tiled
at 512; the new run is 1.57 ms / 0.57 ms. The GPU clocks and session differ
(915 MHz SM in this run), so this is a re-measurement, not a regression claim.

`benchmark_check` (30 cases, 100 samples each, GPU mean milliseconds):

| Graph | Normal | Planned | Async planned |
|---|---:|---:|---:|
| small MLP `[2,3]→[4,1]` | 0.123 | 0.106 | 0.131 |
| large MLP `[16,128] 128→256→64→1` | 0.281 | 0.275 | 0.264 |
| matmul `[64,64]×[64,64]` | 0.050 | 0.047 | 0.042 |
| activation `[4096] relu` | 0.043 | 0.044 | 0.037 |
| reduction `[64,64] sum(axis=1)` | 0.046 | 0.046 | 0.041 |
| dynamic batch `[4,3]→[1]` | 0.041 | 0.039 | 0.035 |

Training parity on the same session: forward `7.749e-7`, 20-step SGD loss
difference `2.325e-6`, post-training parameters `2.623e-6`. Compute Sanitizer
memcheck reports 0 errors. `gpu_check` (execution safety, reclamation, backward,
dynamic shapes, training) passes.

The full Gemma 4 E2B f32 model does not fit a 15 GiB T4 (18.6 GB parameters), so
the full-model CUDA number remains the earlier A100 measurement; T4 verifies the
tiny dense/MoE model and the GPU primitives.

## Vision: MobileNetV4 inference and training

Measurements on the reference CPU workspace (release) and a Tesla T4 for the
training step.

| Measurement | Result |
|---|---:|
| MobileNetV4 `conv_small` single-image CPU inference (128×128, including load) | 1.50 s |
| `conv_small` CPU evaluation, 100 images | 118 s (≈1.18 s/image) |
| `nano` model trainable parameters | 136,494 |
| `conv_small` trainable parameters | 890,474 |
| GPU training | Tesla T4, PyTorch mirror; 20 epochs on 6,800 images, best val 0.7042 |
| Export size (`conv_small`) | 3.5 MiB, 173 tensors |
| CPU cats-vs-dogs example (64 train images, 1 epoch) | 75.6 s, 33 BN layers calibrated, 179 tensors exported |

The CPU reference conv kernels are correctness-oriented; inference latency is
dominated by naive direct convolution and is not representative of a tuned CPU
or CUDA implementation. GPU training currently uses the PyTorch mirror because
oxide-ml's CUDA backend does not implement Conv2d/BatchNorm training.

## CUDA training: oxide-ml vs PyTorch

Tesla T4, MobileNetV4 `conv_small`, 6,800 images, 128×128, batch 64, 107 steps
per epoch, AdamW lr 1.5e-3.

| Backend | Epoch time | Per step |
| --- | ---: | ---: |
| oxide-ml CUDA (`im2col` + cuBLAS, new conv/BN/CE kernels) | 125.4 s | 1.17 s |
| PyTorch (cuDNN) | 10.22 s | 0.095 s |

oxide-ml is **12.3× slower** end-to-end. Replacing the reference tiled GEMM with
cuBLAS improved the epoch from 131 s to 125 s, so the remaining gap is the
convolution lowering (im2col memory traffic, per-layer kernel count) and the
unfused BatchNorm decomposition.

Accuracy note: the oxide-ml CUDA runs did not converge in 1–3 epochs
(validation stuck at the 52.5% majority-class rate, loss ≈0.69); the benchmark
measures speed only. The PyTorch-trained checkpoint remains the accurate model
(0.7042).
