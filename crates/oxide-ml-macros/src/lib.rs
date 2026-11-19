//! Proc macros for oxide-ml.
//!
//! Three entry points, with deliberately separate responsibilities:
//!
//! - `#[derive(Module)]` implements the module/parameter contract.
//! - `#[model]` rewrites a forward method into graph-building code.
//! - `graph!` is a small declarative graph DSL over `GraphBuilder`.
//!
//! The macros never try to replicate Rust's type checker. They translate syntax
//! into ordinary Rust calls, and the resulting code is type-checked by rustc.
//! Unsupported syntax is rejected with a diagnostic pointing at the source.

use proc_macro::TokenStream;

mod attr;
mod derive;
mod graph;
mod model;

/// Derive the `Module` trait: parameter discovery, traversal and forward
/// dispatch.
///
/// ```ignore
/// #[derive(Module)]
/// struct Mlp {
///     fc1: Linear,
///     norm: LayerNorm,
///     fc2: Linear,
/// }
/// ```
///
/// Fields implement `Module` by default. Use `#[module(skip)]` for plain
/// configuration fields, and `#[param]` / `#[param(role = "frozen")]` to make a
/// `DenseTensor<f32>` field a parameter.
#[proc_macro_derive(Module, attributes(oxide, module, param))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    derive::derive(input)
}

/// Translate a forward method into a graph-building function.
///
/// The declared signature is kept for the generated `predict` method, while the
/// generated graph method takes a `&mut ModelContext` and graph values. Method
/// calls on `self.<field>`, tensor operations, and `+ - * /` are rewritten into
/// `oxide_ml::ops` calls.
#[proc_macro_attribute]
pub fn model(attr: TokenStream, item: TokenStream) -> TokenStream {
    model::expand(attr, item)
}

/// Declarative graph construction.
///
/// ```ignore
/// let graph = graph! {
///     input x: [4, 8];
///     param w: [8, 3];
///     param b: [3];
///     let y = matmul(x, w);
///     let y = add(y, b);
///     let y = relu(y);
///     output y;
/// }?;
/// ```
#[proc_macro]
pub fn graph(input: TokenStream) -> TokenStream {
    graph::expand(input)
}
