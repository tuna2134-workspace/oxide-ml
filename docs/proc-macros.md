# Proc macros

`oxide-ml` exposes three macros from the facade:

| Macro | Role |
| --- | --- |
| `#[derive(Module)]` | Implements the model-API `Module` contract: discovery, traversal, forward dispatch. |
| `#[model]` | Rewrites a forward method into graph-building code and generates `predict`. |
| `graph!` | Declarative graph construction over `GraphBuilder`. |

All three expand to ordinary Rust and leave type checking to `rustc`.
`oxide_ml::prelude` re-exports `Module`, `model`, and `graph`.

## `#[derive(Module)]`

```rust
use oxide_ml::prelude::*;

#[derive(Module)]
struct Mlp {
    fc1: Linear,
    #[module(rename = "norm")]
    layer_norm: LayerNorm,
    #[module(skip)]
    seed: u64,
    #[param(role = "frozen")]
    scale: DenseTensor<f32>,
}
```

Structs with named fields and enums are supported. Tuple structs, unit structs,
and unions are rejected at compile time, as is a struct with no module or
parameter fields. Every non-skipped, non-`#[param]` field must implement
`Module`. Generic structs and enums are supported: the derive appends
`FieldType: Module` predicates to the where clause.

Field attributes:

| Attribute | Effect |
| --- | --- |
| *(none)* | Treated as a child module. Its fields contribute to the path. |
| `#[module(skip)]` | Excluded from parameters, traversal, forward calls, and training-mode propagation. |
| `#[module(rename = "x")]` | Uses `x` instead of the Rust field name as the path segment. |
| `#[param]` | Registers a `DenseTensor<f32>` field as a parameter. |
| `#[param(role = "trainable"\|"frozen"\|"buffer")]` | Sets the parameter role. |
| `#[module(body)]` | Enum variants only: the field `forward` dispatches to. |
| `#[module(skip)]` + `#[param]` | Compile error. |

Container options are written as `#[oxide(...)]`:

| Option | Effect |
| --- | --- |
| `crate = ::some_path` (alias `root`) | Facade path used by generated code. Defaults to `::oxide_ml`. |
| `forward_multi = method_name` | Method the generated `forward` delegates to. Default `forward_multi`. |
| `name = "..."` | Value returned by `module_type()`. Defaults to the struct name. |
| `forward`, `predict` | Accepted by the shared option parser; they do not change derive output. |

Parameter names are dot-joined paths from the root, for example `fc1.weight`,
`norm.gamma`, or `0.weight` inside a `Sequential`. The model type name is not
part of the path. Duplicate names are rejected. If two module paths reach the
same tensor storage, the second path becomes an alias of the first canonical
parameter, and both names resolve to one graph parameter.

The generated implementation provides `module_type`, `visit_parameters`,
`forward` (delegating to `forward_multi(&[input])`), `forward_multi`,
`set_training` (forwarded to child modules), `visit_tensors`,
`visit_tensors_mut`, `describe`, and the default `state_config`. It does not
override `is_training`; the trait default is `true`, while containers such as
`Sequential` track their own mode.

### Enum architectural variants

`#[derive(Module)]` also accepts enums. Each variant is an *alternative
architecture*; the variant name is **not** a path segment, so every variant's
fields share the same parameter paths. A checkpoint or optimizer state therefore
stays valid when a variant changes.

```rust
#[derive(Module)]
enum Activation {
    Relu(ReLU),
    Tanh(Tanh),
}

#[derive(Module)]
enum FeedForward {
    Dense { #[module(body)] mlp: GemmaMlp },
    Sparse {
        #[module(body)] moe: GemmaMoE,
        router: Linear, // registered, not called by the derived forward
    },
}
```

The derived `forward`/`forward_multi` dispatches to exactly one module field per
variant: the only module field, or the field marked `#[module(body)]`. Two or
more module fields without a `body` marker are a compile error that names the
required attribute; so is a unit variant (use `Option<M>`). Parameters,
skipped fields, generics and `#[module(rename)]` work inside variants exactly as
they do for structs.

## `#[model]`

```rust
impl Mlp {
    #[model]
    fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
        let hidden = relu(self.fc1.forward(x));
        let hidden = self.norm.forward(hidden);
        self.fc2.forward(hidden)
    }
}
```

The annotated method must have `&self` (not `&mut self`), at least one input
parameter with a simple identifier, and a declared return type. It must not be
`async` or `unsafe`. The annotated body is not emitted as written; the macro
generates graph-building methods with a `&mut ModelContext` and `ValueId`
inputs, plus a `predict` helper that keeps the declared signature.

Rewrite rules:

| Source syntax | Expansion |
| --- | --- |
| `self.fc.forward(x)` | `module_forward(cx, "fc", &self.fc, x)` |
| `self.block.fc.forward(x)` | The same, with path scope `"block.fc"`. |
| `self.attn.forward_multi(q, k, v)` | `module_forward_multi(...)`, wrapped in `single_output` when the model declares one output. |
| `relu(x)`, `x.relu()` | `ops::relu(cx, x)` |
| `add(a, b)`, `a + b` | `ops::add(cx, a, b)` (also `sub`, `mul`, `div`) |
| `-x` | `ops::neg(cx, x)` |
| `matmul(a, b)` / `matmul_t(a, b, ta, tb)` | Matching `ops` calls. |
| `apply_kernel(x, "name")` | `ops::apply_kernel`; output shape equals input shape. |
| `apply_kernel_shaped(x, "name", [dims])` | `ops::apply_kernel_shaped`; explicit static output shape. |
| Any other call | Left as ordinary Rust. |

A `for` loop over a module collection is rewritten so the loop variable keeps
its parameter path:

```rust
#[model]
fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
    let mut hidden = x;
    for layer in &self.layers {              // ModuleList<Linear>, Vec<Linear>, ...
        hidden = layer.forward(hidden);
    }
    hidden
}
```

Supported iterables are `&self.layers`, `self.layers`, `self.layers.iter()`,
`self.layers.iter().enumerate()` (pattern `for (index, layer) in ...`), and the
same shapes rooted at a module bound by an enclosing module loop, so nested
`for block in &self.blocks { for layer in &block.layers { ... } }` works. The
generated loop appends the running index to the parameter path. A loop is only
rewritten when its body calls `forward`/`forward_multi` on the loop variable, so
ordinary `for value in &self.config.heads` loops keep their element binding.
Loops over a dynamic number of layers are unrolled at graph-construction time.

The tensor operations recognized by the macro are `relu`, `gelu`, `sigmoid`,
`tanh`, `exp`, `log`, `sqrt`, `abs`, `neg`, `add`, `sub`, `mul`, `div`,
`matmul`, `matmul_t`, `reshape`, `transpose`, `transpose_last`,
`broadcast_to`, `mean`, `sum`, `softmax`, `log_softmax`, `layer_norm`,
`mse_loss`, `cross_entropy_loss`, `scale`, `cast`, `apply_kernel`, and
`apply_kernel_shaped`. Each rewritten call evaluates its arguments into
temporaries, so nested calls do not borrow one `&mut ModelContext` twice.

Supported statements and expressions include `let`, ordinary expressions,
`if`/`else`, `for`, `while`, `loop`, `match`, `return`, blocks, literal
constants, and module calls. Every value used as a tensor must be a `ValueId` or
a scalar accepted by the operation.

Rejected with a compile error:

| Construct | Reason |
| --- | --- |
| `async fn` / `unsafe fn` | The rewrite has no async or unsafe execution model. |
| Missing `&self`, or `&mut self` | Graph building must not mutate the model. |
| Closures, `.await`, async blocks | Not representable in the generated graph code. |
| Non-identifier parameters such as `(a, b)` | The generated method needs named inputs. |
| `.forward()` on a local not bound by a module loop | Only `self.<field>` chains and module-collection loop variables are recognized. |
| `self.forward(...)` | Would recurse; call a module field instead. |
| A body without a tail output expression | The macro must know what to return. |
| `return;` without a value | The forward method must return an output. |
| A tuple variable for a multi-output return | Return a literal tuple of expressions. |

Generated methods:

| Method | When | Signature |
| --- | --- | --- |
| `forward` | Exactly one input | `fn forward(&self, cx: &mut ModelContext, input: ValueId) -> Result<Vec<ValueId>>` |
| `forward_multi` | Always | `fn forward_multi(&self, cx: &mut ModelContext, inputs: &[ValueId]) -> Result<Vec<ValueId>>` |
| `predict` | Unless `predict = false` | Declared argument types, returning `Result<T>`, or `Result<(T, U)>` for a tuple return. |

`#[model(...)]` accepts `crate = ...` and `predict = false`.
The derive expects the graph methods to be named `forward` and `forward_multi`;
use `#[oxide(forward_multi = "...")]` when the `#[model]` method has another
name. A hand-written `Module` implementation can call any method name.

`predict` is generated for the annotated forward method unless `predict = false`;
it compiles the model afresh for each call. `name` is accepted for shared option
parsing, but `#[model]` does not emit `module_type()`.

## `graph!`

```rust
use oxide_ml::prelude::*;

let graph = graph! {
    input x: [4, 8];
    param w: [8, 3];
    param b: [3] frozen;
    let y = matmul(x, w);
    let y = add(y, b);
    let y = relu(y);
    output y;
}?;
assert!(GraphVerifier::verify(&graph).is_ok());
```

`graph!` expands to a closure returning `Result<Graph>`. Its statements are
`input name: [dims];`, `param name: [dims];`, `param name: [dims] trainable;`,
`param name: [dims] frozen;`, `let name = op(args);`, `output name;`, and
`outputs [a, b];`. It requires at least one output.

| Operation | Arguments |
| --- | --- |
| `add`, `sub`, `mul`, `div`, `matmul` | `(lhs, rhs)` |
| `neg`, `relu`, `gelu`, `sigmoid`, `tanh`, `sqrt`, `exp`, `log`, `abs` | `(value)` |
| `softmax`, `log_softmax` | `(value, axis)` |
| `reshape` | `(value, dims)` |
| `transpose` | `(value, [axes])` |
| `mean`, `sum` | `(value, [axes])` or `(value, [axes], keep_dims)` |
| `slice` | `(value, axis, start, len)` |
| `embedding` | `(weight, indices)` |
| `mse_loss`, `cross_entropy_loss` | `(prediction, target)`; reduction is `Mean`. |
| `layer_norm` | `(value, gamma, beta, shape, eps)`; `shape` is a `Vec<usize>` such as `vec![8]`. |

`graph!` is facade-only: it refers to `::oxide_ml::__private` and has no crate
override. It has no constants: every value is an input, a parameter, or the
result of a listed operation. It is a statement list, not general Rust, so no
`if`, loops, or closures are accepted inside it.

## Limitations

- `#[model]` recognizes `.forward()`/`.forward_multi()` on `self.<field>`
  chains and on loop variables over module collections. A module created in a
  local variable and forwarded directly is rejected: store it in a field or
  iterate the field.
- Unlisted calls are not rewritten; they must be ordinary Rust that does not
  need the `ModelContext`.
- `graph!` cannot be used from a crate that renames the facade dependency, and
  cannot express constants or control flow.
- Generated code reserves identifiers beginning with `__oxide_`; do not use such
  names for locals, fields, or graph values.
- The model API is F32-only, so `cast` only succeeds when the dtype is unchanged.

## Compile-fail tests

`tests/compile_fail.rs` drives `trybuild` over `tests/ui/*.rs` and compares each
diagnostic with its `.stderr` snapshot: tuple structs, unit and ambiguous enum
variants, `#[module(skip)]` + `#[param]`, local `.forward()`, closure bodies and
missing output expressions. Update snapshots deliberately with
`TRYBUILD=overwrite cargo test -p oxide-ml --test compile_fail`.
