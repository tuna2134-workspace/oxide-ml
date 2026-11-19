# Control-flow IR — design and status

**Status: NOT IMPLEMENTED (design only).**

`#[model]` control flow (`if`, `match`, `for`, `while`, `loop`) still executes on
the host while the graph is being built. This is correct for conditions that
depend only on configuration (`self.training`, static shapes, hyperparameters),
and it is what the current tests cover. Tensor-dependent control flow
(`if x > 0`, data-dependent loops) cannot be expressed in the IR yet, so it is
rejected by ordinary Rust type checking (a graph value is not a `bool`).

This document specifies the intended design so the representation can be added
without changing the rest of the compiler.

## Representation

```text
Graph
  regions: Vec<Region>
  nodes:   Vec<Node>          // SSA nodes; region membership is explicit

Region
  id: RegionId
  kind: Root | Then | Else | Cond | Body
  parent: Option<RegionId>
  inputs: Vec<ValueId>        // branch / loop-carried arguments
  outputs: Vec<ValueId>       // yields (region results)
  nodes: Vec<NodeId>          // nodes owned by the region, topological order

Node
  regions: Vec<RegionId>      // Cond: [then, else]; While: [cond, body]
```

- `Op::Cond { outputs: N }`: inputs `[predicate]`; selects `then` or `else`;
  each region yields `N` values.
- `Op::While { carried: N }`: inputs are the initial carried values; the
  `cond` region yields one scalar, the `body` region yields `N` values.
- Values remain global SSA. A region may capture values defined outside it;
  values defined inside a region are visible only to that region and its
  descendants. Region yields are the only way values escape.

## Verifier rules

1. Region ids are dense and match their index.
2. A node belongs to at most one region; nested regions reference a parent.
3. `Cond`/`While` have exactly the required regions and arities.
4. Region inputs and captured values are defined before the control node;
   region outputs are defined inside the region.
5. A region's node list is a topologically ordered subset of the graph nodes.
6. No value defined in an inactive region is used outside it.
7. `Cond` predicate and `While` condition are scalar values.

## Execution

The CPU executor becomes a structured interpreter:

- maintain an active-region stack and a value environment;
- at `Cond`, evaluate the predicate (scalar tensor), push the selected region,
  execute its nodes, pop, bind the region yields to the `Cond` outputs;
- at `While`, evaluate the `cond` region, run `body`, copy body yields into the
  carried environment, repeat with an iteration guard;
- skip nodes whose region is not active.

CUDA execution needs the same structure; device-side control flow would lower
`Cond` to predicated branches and `While` to a loop with a device-side condition,
or the graph could be partitioned into host-launched segments.

## Autograd

- `Cond`: route `dy` to the taken branch only (the other branch receives zeros).
- `While`: reverse loop with a saved iteration count, or recomputation.
- Both require the backward graph to contain regions and the verifier to check
  region-local gradient accumulation. Until this exists, `GradientRegistry`
  must reject `Cond`/`While` with an explicit `Unsupported` error.

## Compiler / cache impact

- Canonical hashing must encode region structure, kinds, inputs, outputs and
  nested regions, and the op tags.
- Passes must operate per region (or reject region graphs until implemented).
- The compile cache key must include the canonical hash, which then covers the
  control-flow structure automatically.

## Work required

1. IR types + serialization (`IR_VERSION` bump, v1 migration).
2. Builder API: `region(...)`, `cond(...)`, `while_loop(...)` with a region
   stack; `#[model]` lowering for `if_tensor!`/`while_tensor!` (plain `if`
   stays host-side because the macro has no type information).
3. Verifier rules above.
4. CPU interpreter + tests (diamond, nested branches, bounded loop).
5. Autograd for `Cond` (branch routing) and `While` (reverse loop).
6. Canonical encoding and pass updates.
7. CUDA lowering/partitioning.

Until these land, this file documents the gap rather than claiming support.
