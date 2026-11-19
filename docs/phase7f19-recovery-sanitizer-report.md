> **Closure note:** the final classification of these results is recorded in
> `phase7f-final-closure.md` and `phase7f-verification-debt.md`. NOT RUN and the
> mixed synccheck application FAIL remain debt.

# Phase 7F-19 Recovery Sanitizer Report

## 1. Environment

- Tesla T4, CC 7.5, UUID `GPU-527ac068-0847-107c-a117-e78d96dacbb8`
- Driver 580.82.07, CUDA 13.0
- `compute-sanitizer --error-exitcode 1`
- Binaries built with `cargo oxide build --arch sm_75 -- --bin <name> --release`

## 2. Results

| Target | memcheck | racecheck | synccheck |
|---|---|---|---|
| `constant_copy_check` | PASS (0) post-fix | PASS (0 hazards) post-fix | PASS (0) post-fix |
| `backward_check_full` | PASS (0) post-fix | PASS (0 hazards) post-fix | PASS (0) post-fix |
| `linear_regression_check` | PASS (0) | NOT RUN (300 s timeout) | PASS (0) |
| `mixed_graph_check` | PASS (0) post-fix | PASS (0 hazards) post-fix | **FAIL (AccessGuard assertion)** |
| `concurrency_graph_check` | PASS (0) post-fix | NOT RUN (300 s timeout) | NOT RUN |
| `shape_matrix_check` | PASS (0) post-fix | PASS (0 hazards) post-fix | PASS (0) post-fix |
| `benchmark_check` | PASS (0) post-fix | PASS (0 hazards) post-fix | PASS (0) post-fix |

### 2.1 Completed results

- Linear regression synccheck completed the full 122-case matrix:
  `ERROR SUMMARY: 0 errors`, `EXIT:0`.
- Concurrency graph memcheck completed the full 177-case matrix:
  `ERROR SUMMARY: 0 errors`, `EXIT:0`.
- Linear regression racecheck and concurrency racecheck/synccheck were killed
  after 300 s because the full matrices are too slow under those tools. Partial
  logs are kept; they are NOT RUN, not PASS.

## 3. Mixed graph synccheck failure detail

Sanitizer itself reported:

```text
========= ERROR SUMMARY: 0 errors
EXIT:11
```

The binary's `AccessGuard stats work` check expects
`same_stream_elided=1`. Under synccheck instrumentation the observed value was
one of:

```text
same_stream_elided=0 cross_stream=0 pending_writer=1   (one run, EXIT:11)
same_stream_elided=2 cross_stream=0 pending_writer=1   (another run, EXIT:0)
```

The sanitizer itself reported 0 errors in both runs. The application-level
timing-sensitive counter assertion failed because the observed elision count
depends on host scheduling. This is recorded as **FAIL**, not a clean PASS, and
the cause is not claimed to be resolved. A follow-up should replace the exact
count assertion with a scheduling-independent check (counter readability plus
the presence of the pending writer) and then re-run synccheck.

## 4. Interpretation

- A sanitizer PASS means the tool completed the binary with exit 0 and reported
  no errors.
- A sanitizer NOT RUN is not PASS.
- The mixed synccheck result has 0 sanitizer errors but a non-zero application
  exit; it is not counted as a clean PASS.
- No performance number from a sanitizer run is used in the benchmark reports.

## 5. Logs

- `p7f19-recovery-const-{memcheck,racecheck,synccheck}.log`
- `p7f19-recovery-backward-synccheck.log`
- `p7f19-recovery-linear-memcheck.log`, `p7f19-recovery-linear-synccheck.log`
- `p7f19-recovery-mixed-{memcheck,racecheck,synccheck}.log`
- `p7f19-recovery-shape-{memcheck,racecheck,synccheck}.log`
- `p7f19-recovery-benchmark-memcheck.log`
- `p7f19-recovery-concurrency-memcheck.log`
- `p7f19-recovery-cuda-clippy.log`, `p7f19-recovery-cuda-test.log`

Partial logs for timed-out racechecks are kept remotely; they are not counted as
PASS.
