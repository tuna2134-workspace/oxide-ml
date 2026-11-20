# Operator support matrix

Status is per implementation and verification, not per API. `YES` means a kernel
exists and is covered by CPU/CUDA tests; `—` means no implementation; `NOT
VERIFIED` means the pieces exist but were not run for that backend. The CUDA
column reflects the kernels in `crates/oxide-ml-cuda/src/kernels.rs` verified on
a T4 or A100. `Slice`, `Embedding`, `Concat`, `Split`, `LogSoftmax` and fused
`MoE` are implemented on CUDA; `SliceBackward` and `EmbeddingBackward` are not
and return `CudaError::Unsupported`.

| Operator | CPU fwd | CPU bwd | CUDA fwd | CUDA bwd | Notes |
|---|---|---|---|---|---|
| Add / Sub / Mul / Div | YES | YES | YES | YES | broadcast + unbroadcast |
| Neg | YES | YES | YES | YES | |
| AddRelu | YES | YES | YES | YES | fused |
| ReLU / GELU / Sigmoid / Tanh | YES | YES | YES | YES | |
| Sqrt / Exp / Log / Abs | YES | YES | YES | — | T4 parity verified; backward for Sqrt/Exp/Log composes forward ops |
| Softmax | YES | YES | YES | — | CUDA forward verified |
| LogSoftmax | YES | YES | YES | — | A100 verified | |
| MatMul (incl. batched) | YES | YES | YES (batch 1) | YES | CUDA rank-2 / rank-3 batch-1 |
| MatMulBias | YES | YES | YES | — | matmul + broadcast add on device |
| MulAdd | YES | YES | YES | — | mul + add on device |
| Reshape | YES | YES | YES | — | |
| Transpose | YES | YES | YES | — | |
| BroadcastTo | YES | YES | YES | — | |
| ReduceSum / ReduceMean | YES | YES | YES | — | |
| Slice / Concat / Split | YES | YES | YES | — | A100 verified |
| Cast (same dtype) | YES | YES | YES | — | device copy; dtype-changing cast rejected |
| Embedding | YES | YES | YES | — | indices are F32 integral; A100 verified |
| LayerNorm | YES | YES | YES | — | CUDA backward not implemented |
| BatchNorm | YES | YES | YES | NOT VERIFIED | T4 forward parity (training stats); running stats not updated |
| GroupNorm / InstanceNorm | YES | YES | — | — | composed from graph ops |
| Conv2d | YES | YES | — | — | |
| Conv1d | YES | YES | — | — | lowered to Conv2d |
| Conv3d | — | — | — | — | explicit unsupported |
| MaxPool2d / AvgPool2d | YES | YES | — | — | |
| MaxPool1d | YES | — | — | — | forward only |
| Attention / MultiHeadAttention | YES | YES | NOT VERIFIED | — | CUDA forward composes supported ops |
| RNN / GRU / LSTM | YES | YES | — | — | last state or `[B,T,H]`; no initial state |
| MSELoss | YES | YES | YES | YES | |
| CrossEntropyLoss | YES | YES | — | — | |
| BCEWithLogits / Huber / NLL | YES | YES | — | — | composed from elementwise ops |
| MoE (fused router + experts) | YES | — | YES | — | inference only; A100 verified |
| Constant / ConstantScalar | YES | n/a | YES | n/a | F32 CUDA |
| Custom (`Op::Custom`) | YES | YES | YES | rule-dependent | T4 verified with `launch_with_access` |
| F16 / BF16 / I8 / I16 / I32 / I64 / U8 / Bool | — | — | — | — | IR/type vocabulary only |
| Dropout | eval only | n/a | eval only | n/a | training returns `Unsupported` |
| AbsBackward | YES | n/a | YES | n/a | T4 parity verified |
| SliceBackward / EmbeddingBackward | YES | n/a | — | n/a | fused backward ops |

## Gradient coverage

Every CPU forward operation above has a gradient rule, exercised by finite
differences (`tests/gradient_check.rs`, `crates/oxide-ml-autograd/tests/*`). A
missing rule is reported when building the training graph. `Cond`/`While` are
not in the IR yet; see [control-flow-ir.md](control-flow-ir.md).

## CUDA verification scope

The T4 run after the compiler-framework milestone covered: model
forward/training parity (`model_check`, now through fused `MatMulBias`/`AddRelu`),
a custom cuda-oxide kernel (`custom_kernel_check`), the new elementwise ops,
`AbsBackward`, BatchNorm forward (`new_ops_check`), the existing `gpu_check`
acceptance suite (40 PASS), and Compute Sanitizer on the new binaries.
Other CUDA acceptance binaries and the operator matrix above were not re-run;
treat `CUDA fwd/bwd` as "kernel exists", not "verified for every shape".
