# Phase 7F-19 Mixed Static / Bounded / Unbounded Graph Report

## 1. Graph

`crates/oxide-ml-cuda/src/bin/mixed_graph_check.rs` builds one graph containing:

- static parameter `weight[2,4]` and bias, used by both branches;
- static input `xs[2,4]` and static intermediate `ys`;
- bounded symbol input `xb[batch,4]`, bounded intermediates `hb`, `hb2` and
  bounded output `yb`;
- `Dim::Dynamic` input `xd[?,4]`, unbounded intermediates `hd`, `yd`;
- allocations through plan slots and through the pool fallback in the same
  execution.

Results: `phase7f19-mixed-graph-results.json` / `.csv`.

## 2. Checks

| Check | Result |
|---|---|
| `plan_bounded` on the mixed graph | **FAIL (API limitation)** |
| Fallback plan with `MemoryPlanner` + `from_memory_plan` | PASS |
| Static + bounded values use plan slots | PASS |
| Unbounded values are unplanned (pool fallback) | PASS |
| Bounded capacity covers batch max 8 | PASS |
| Shared parameter used by bounded and unbounded branches | PASS |
| Both allocation paths coexist in one planned execution | PASS |
| Pool and arena buffer identities do not collide | PASS |
| Buffer generation increments across arena reuse | PASS |
| AccessGuard stats work | PASS |
| Dropped async planned execution reclaims after sync | PASS |
| Dropped async arena execution reclaims; leases returned | PASS |
| Batch 9 rejected with `CudaError::Capacity` before launch | PASS |
| Failed overflow keeps existing arena lease; later batch 4 valid | PASS |

`plan_bounded` rejects the graph because `specialize_graph_with_bindings`
rejects any value containing `Dim::Dynamic`:

```text
CUDA capacity exceeded: shape [?, 4] contains an unbounded dynamic dimension;
bounded specialization requires a symbol or static dimension
```

This is the current API contract: a plan is either bounded-symbol specialized or
uses the generic planner with unplanned dynamic values. The benchmark records
the rejection as FAIL against the literal "plan_bounded on mixed graph"
expectation, and uses the existing public fallback. No executor or safety code
was changed.

The fallback plan had 3 slots, 448 planned bytes, and 2 unplanned values
(`hd`, `yd`). Logical bytes at batch 8 were 1072. Capacity overflow was rejected
before any event allocation changed (event allocations 3 → 3), supporting the
"before kernel launch" claim; the error is raised by
`validate_runtime_capacity` before the node loop.

## 3. Timings (100 samples, batch 8)

| Mode | GPU median ms | Host median ms | Wait median ms |
|---|---:|---:|---:|
| normal | 0.2120 | 0.2102 | 0 (implicit) |
| planned | 0.1989 | 0.1971 | 0 (implicit) |
| arena | 0.2196 | 0.2173 | 0 (implicit) |
| async_planned | 0.1741 | 0.1728 | 0.0063 |
| async_arena | 0.2314 | 0.2288 | 0.0069 |

The async cases show a real separation between submit and wait. The sync cases
include the implicit wait in host time.

## 4. Memory

Snapshot after the measured cases:

- pool reserved 3728 bytes, active 0, pending 0, quarantined 0
- pool allocations 42, reuses 24
- arena capacity 320 bytes, physical reserved 320, active leases 0
- event allocations/reuses are recorded in the JSON

Driver physical memory is not attributed; `driver_used_bytes` is `null`.

## 5. Compile time

`compile_ms` is recorded as `0.0` in this binary: graph construction and plan
creation are performed once before the timed modes and were not separately
timed. No compile-time number is claimed for the mixed graph.

## 6. Verdict

| Criterion | Verdict |
|---|---|
| Static/bounded/unbounded coexist in one graph | PASS |
| Pool fallback for unbounded values | PASS |
| Capacity rejection before launch | PASS |
| AccessGuard / identity / generation / drop reclamation | PASS |
| `plan_bounded` on mixed graph | FAIL (documented API limitation) |
| Timed modes | PASS (5 modes) |
| Overall | **PASS with documented API limitation** |

The benchmark closure criterion is met: the mixed graph executes on the real
T4 with both allocation paths and all safety checks. The `plan_bounded` result
is not hidden; it is an explicit API constraint that the next phase may choose
to relax.
