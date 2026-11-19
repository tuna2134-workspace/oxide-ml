# Phase 7F CUDA Adapter Fault Injection — Report

> **Status note (2026-11-19):** real driver fault injection remains SOURCE ONLY /
> NOT RUN and is tracked as VD-01 in `phase7f-verification-debt.md`.

## 境界

実 CUDA Driver に任意障害を注入することはしない。既存の安全層を
Testable Boundary として使用する。

| Fault | Boundary | Test |
|---|---|---|
| Event Create | oxide-ml-event-safety Mock | PASS |
| Event Record | event-safety + access_failure | PASS |
| Event Query | event-safety Mock | PASS |
| Event Synchronize | event-safety Mock | PASS |
| Event Destroy | event-safety Mock | PASS |
| Stream Wait Event | access_failure Mock Harness | PASS |
| Kernel Launch | access_failure Mock Harness | PASS |
| Context Binding | execution-safety Mock | PASS |
| Device Lost | execution-safety Mock | PASS |
| Buffer Allocation | execution-safety / pool | PASS |
| Buffer Release | event-safety / execution-safety | PASS |

## Timing

1. Launch 前 failure: state / buffer を記録しない。
2. Launch 自体の failure: EventLease 未記録、buffer invalidate。
3. Launch 後 Event Record failure: Event Failed → EventPool quarantine、
   buffer invalidate → Buffer quarantine。
4. Execution 中 Query failure: complete と扱わない。
5. Completion Wait failure: Execution Failed、resource 保持。
6. Drop 時 failure: `finish_or_return` で resource 保持、panic しない。

## CPU Tests

- event-safety: 22 PASS
- execution-safety: 35 PASS（Access Failure Mock 8 含む）

## 実 CUDA Driver Fault

**SOURCE ONLY / NOT RUN.** 実 Driver の Event/Stream/Kernel failure は
意図的に発生させていない。T4 では正常経路 Regression のみ PASS。

## 既知の制約

- Production CUDA に汎用 FaultInjector trait は導入していない。
  CPU Mock は安全層の状態遷移を検証し、CUDA 実装は同じ順序・同じ quarantine
  経路を使用する。