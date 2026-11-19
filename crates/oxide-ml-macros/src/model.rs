//! `#[model]`: rewrite a forward method into graph-building code.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::visit_mut::VisitMut;
use syn::{
    Block, Error, Expr, ExprBinary, ExprCall, ExprMethodCall, FnArg, ImplItemFn, Member, Pat,
    ReturnType, Stmt, Type, parse_macro_input, parse_quote, spanned::Spanned,
};

use crate::attr::Options;

/// Operations that map to `oxide_ml_model::ops` free functions.
const TENSOR_OPS: &[&str] = &[
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "abs",
    "neg",
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "matmul_t",
    "reshape",
    "transpose",
    "transpose_last",
    "broadcast_to",
    "mean",
    "sum",
    "softmax",
    "log_softmax",
    "layer_norm",
    "mse_loss",
    "cross_entropy_loss",
    "scale",
    "cast",
    "apply_kernel",
    "apply_kernel_shaped",
];

pub fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let options = match Options::parse_attribute(attr.into()) {
        Ok(options) => options,
        Err(error) => return error.to_compile_error().into(),
    };
    let method = parse_macro_input!(item as ImplItemFn);
    match expand_method(method, options) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand_method(method: ImplItemFn, options: Options) -> syn::Result<TokenStream2> {
    let root = &options.root;
    let receiver = method.sig.receiver().ok_or_else(|| {
        Error::new_spanned(&method.sig, "#[model] requires a method with `&self`")
    })?;
    if receiver.reference.is_none() || receiver.mutability.is_some() {
        return Err(Error::new_spanned(
            receiver,
            "#[model] requires `&self`; graph building must not mutate the model",
        ));
    }
    if method.sig.asyncness.is_some() || method.sig.unsafety.is_some() {
        return Err(Error::new_spanned(
            &method.sig,
            "#[model] does not support `async` or `unsafe` forward functions",
        ));
    }

    let mut input_idents = Vec::new();
    let mut input_patterns = Vec::new();
    let mut input_types = Vec::new();
    for argument in &method.sig.inputs {
        let FnArg::Typed(typed) = argument else {
            continue;
        };
        let Pat::Ident(pattern) = typed.pat.as_ref() else {
            return Err(Error::new_spanned(
                &typed.pat,
                "#[model] requires simple identifier parameters (`x: Tensor<f32>`)",
            ));
        };
        input_idents.push(pattern.ident.clone());
        input_patterns.push(pattern.clone());
        input_types.push((*typed.ty).clone());
    }
    let arity = input_idents.len();
    if arity == 0 {
        return Err(Error::new_spanned(
            &method.sig,
            "#[model] requires at least one tensor input",
        ));
    }

    let return_type = match &method.sig.output {
        ReturnType::Type(_, ty) => (**ty).clone(),
        ReturnType::Default => {
            return Err(Error::new_spanned(
                &method.sig,
                "#[model] forward must return a tensor or a tuple of tensors",
            ));
        }
    };
    let output_arity = tuple_arity(&return_type);

    let mut block = method.block.clone();
    {
        let mut rewriter = Rewriter {
            root: root.clone(),
            output_arity,
            errors: Vec::new(),
        };
        rewriter.visit_block_mut(&mut block);
        if let Some(error) = combine_errors(rewriter.errors) {
            return Err(error);
        }
    }
    wrap_tail(&mut block, output_arity)?;

    let forward_name = method.sig.ident.clone();
    let forward_multi_name = quote::format_ident!("{}_multi", forward_name);
    let vis = method.vis.clone();
    let cx = quote::format_ident!("__oxide_cx");
    let inputs = quote::format_ident!("__oxide_inputs");
    let indexes = (0..arity).map(syn::Index::from);

    let single_forward = if arity == 1 {
        quote! {
            #vis fn #forward_name(
                &self,
                #cx: &mut #root::__private::ModelContext,
                #(#input_patterns: #root::__private::ValueId),*
            ) -> #root::__private::Result<::std::vec::Vec<#root::__private::ValueId>> {
                self.#forward_multi_name(#cx, &[#(#input_idents),*])
            }
        }
    } else {
        TokenStream2::new()
    };

    let graph_methods = quote! {
        #single_forward

        #vis fn #forward_multi_name(
            &self,
            #cx: &mut #root::__private::ModelContext,
            #inputs: &[#root::__private::ValueId],
        ) -> #root::__private::Result<::std::vec::Vec<#root::__private::ValueId>> {
            if #inputs.len() != #arity {
                return ::std::result::Result::Err(
                    #root::__private::Error::invalid_graph(::std::format!(
                        "model {} expects {} inputs, got {}",
                        stringify!(#forward_name),
                        #arity,
                        #inputs.len(),
                    )),
                );
            }
            #(let #input_patterns = #inputs[#indexes];)*
            #block
        }
    };

    let predict_method = if options.predict {
        let unwrap = predict_unwrap(root, output_arity);
        quote! {
            #vis fn predict(
                &self,
                #(#input_idents: #input_types),*
            ) -> ::std::result::Result<#return_type, #root::__private::Error> {
                let __oxide_outputs = #root::__private::predict_many(
                    self,
                    ::std::vec![#(#input_idents),*],
                )?;
                #unwrap
            }
        }
    } else {
        TokenStream2::new()
    };

    Ok(quote! {
        #graph_methods
        #predict_method
    })
}

fn tuple_arity(ty: &Type) -> usize {
    match ty {
        Type::Tuple(tuple) => tuple.elems.len(),
        Type::Paren(paren) => tuple_arity(&paren.elem),
        Type::Group(group) => tuple_arity(&group.elem),
        _ => 1,
    }
}

fn predict_unwrap(root: &syn::Path, arity: usize) -> TokenStream2 {
    if arity == 1 {
        quote! {
            let mut __oxide_iter = __oxide_outputs.into_iter();
            let __oxide_first = __oxide_iter.next().ok_or_else(|| {
                #root::__private::Error::invalid_graph("model produced no output")
            })?;
            ::std::result::Result::Ok(__oxide_first)
        }
    } else {
        let values = (0..arity).map(|_| {
            quote! {
                __oxide_iter.next().ok_or_else(|| {
                    #root::__private::Error::invalid_graph("model produced too few outputs")
                })?
            }
        });
        quote! {
            let mut __oxide_iter = __oxide_outputs.into_iter();
            ::std::result::Result::Ok((#(#values),*))
        }
    }
}

fn combine_errors(errors: Vec<Error>) -> Option<Error> {
    let mut iter = errors.into_iter();
    let mut combined = iter.next()?;
    for error in iter {
        combined.combine(error);
    }
    Some(combined)
}

fn wrap_tail(block: &mut Block, arity: usize) -> syn::Result<()> {
    let Some(last) = block.stmts.last_mut() else {
        return Err(Error::new(
            proc_macro2::Span::call_site(),
            "#[model] forward function is empty",
        ));
    };
    match last {
        Stmt::Expr(Expr::Return(_), _) => Ok(()),
        Stmt::Expr(expr, None) => {
            let wrapped = wrap_outputs(expr, arity)?;
            *expr = wrapped;
            Ok(())
        }
        Stmt::Expr(expr, Some(_)) => Err(Error::new_spanned(
            expr,
            "#[model] forward must end with an expression (remove the trailing `;` \
             and return the output tensor)",
        )),
        other => Err(Error::new_spanned(
            other,
            "#[model] forward must end with an output expression",
        )),
    }
}

fn wrap_outputs(expr: &Expr, arity: usize) -> syn::Result<Expr> {
    if arity == 1 {
        return Ok(parse_quote! { ::std::result::Result::Ok(::std::vec![#expr]) });
    }
    match expr {
        Expr::Tuple(tuple) if tuple.elems.len() == arity => {
            let elems: Vec<&Expr> = tuple.elems.iter().collect();
            Ok(parse_quote! { ::std::result::Result::Ok(::std::vec![#(#elems),*]) })
        }
        other => Err(Error::new_spanned(
            other,
            format!(
                "#[model] declares {arity} outputs; return a tuple of {arity} expressions \
                 (a tuple variable is not supported)"
            ),
        )),
    }
}

struct Rewriter {
    root: syn::Path,
    output_arity: usize,
    errors: Vec<Error>,
}

impl Rewriter {
    fn error(&mut self, tokens: impl ToTokens, message: impl std::fmt::Display) {
        self.errors.push(Error::new_spanned(tokens, message));
    }

    fn parse_expr(&mut self, tokens: TokenStream2, span: proc_macro2::Span) -> Expr {
        match syn::parse2::<Expr>(tokens.clone()) {
            Ok(expr) => expr,
            Err(error) => {
                self.errors.push(Error::new(
                    span,
                    format!("internal #[model] rewrite failed: {error} ({tokens})"),
                ));
                parse_quote! { unreachable!() }
            }
        }
    }

    fn ops_call(
        &mut self,
        name: &syn::Ident,
        args: Vec<TokenStream2>,
        span: proc_macro2::Span,
    ) -> Expr {
        let root = &self.root;
        // Evaluate every argument into a temporary first. Nested rewrites need
        // `&mut ModelContext`, and two temporaries inside one call expression
        // would otherwise borrow the context twice.
        let mut bindings = Vec::new();
        let mut temporaries = Vec::new();
        for (index, argument) in args.into_iter().enumerate() {
            let temporary = quote::format_ident!(
                "__oxide_arg{}",
                index,
                span = proc_macro2::Span::call_site()
            );
            bindings.push(quote! { let #temporary = #argument; });
            temporaries.push(temporary);
        }
        let tokens = quote! {
            {
                #(#bindings)*
                #root::__private::ops::#name(__oxide_cx, #(#temporaries),*)?
            }
        };
        self.parse_expr(tokens, span)
    }
}

impl VisitMut for Rewriter {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        // User-written `?` around an operation: the rewrite already adds one.
        if let Expr::Try(try_expr) = expr {
            self.visit_expr_mut(&mut try_expr.expr);
            if ends_with_try(&try_expr.expr) {
                let inner = (*try_expr.expr).clone();
                *expr = inner;
            }
            return;
        }
        if let Expr::Closure(closure) = expr {
            self.error(
                closure,
                "#[model] does not support closures; write the operation explicitly",
            );
            return;
        }
        if let Expr::Await(awaited) = expr {
            self.error(awaited, "#[model] does not support `.await`");
            return;
        }
        if let Expr::Async(asynchronous) = expr {
            self.error(asynchronous, "#[model] does not support `async` blocks");
            return;
        }

        syn::visit_mut::visit_expr_mut(self, expr);

        let replacement = match &*expr {
            Expr::Call(call) => self.rewrite_call(call),
            Expr::MethodCall(method_call) => self.rewrite_method_call(method_call),
            Expr::Binary(binary) => self.rewrite_binary(binary),
            Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
                // A negative literal is a constant argument (e.g. `softmax(-1)`),
                // not a tensor negation operation.
                if matches!(unary.expr.as_ref(), Expr::Lit(_)) {
                    None
                } else {
                    let inner = unary.expr.to_token_stream();
                    let span = unary.span();
                    Some(self.ops_call(&quote::format_ident!("neg"), vec![inner], span))
                }
            }
            Expr::Return(return_expr) => match &return_expr.expr {
                Some(inner) => match wrap_outputs(inner, self.output_arity) {
                    Ok(wrapped) => Some(parse_quote! { return #wrapped }),
                    Err(error) => {
                        self.errors.push(error);
                        None
                    }
                },
                None => {
                    self.error(
                        return_expr,
                        "#[model] forward cannot use a bare `return`; return an output tensor",
                    );
                    None
                }
            },
            _ => None,
        };
        if let Some(replacement) = replacement {
            *expr = replacement;
        }
    }
}

impl Rewriter {
    fn rewrite_call(&mut self, call: &ExprCall) -> Option<Expr> {
        let Expr::Path(path) = call.func.as_ref() else {
            return None;
        };
        let ident = path.path.get_ident()?;
        if !TENSOR_OPS.contains(&ident.to_string().as_str()) {
            return None;
        }
        let args = call.args.iter().map(ToTokens::to_token_stream).collect();
        let span = call.span();
        Some(self.ops_call(&ident.clone(), args, span))
    }

    fn rewrite_method_call(&mut self, method_call: &ExprMethodCall) -> Option<Expr> {
        let method = method_call.method.clone();
        let method_name = method.to_string();
        let span = method_call.span();

        if let Some(segments) = self_path(&method_call.receiver) {
            if method_name == "forward" || method_name == "forward_multi" {
                if segments.is_empty() {
                    self.error(
                        method_call,
                        "calling `.forward()` on `self` inside #[model] would recurse; \
                         call a module field instead (`self.fc.forward(x)`)",
                    );
                    return None;
                }
                let receiver = method_call.receiver.to_token_stream();
                let path = segments
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(".");
                let root = &self.root;
                let tokens = if method_name == "forward" {
                    if method_call.args.len() != 1 {
                        self.error(
                            method_call,
                            "module `.forward()` expects exactly one input; \
                             use `.forward_multi(...)` for multiple inputs",
                        );
                        return None;
                    }
                    let argument = method_call.args[0].to_token_stream();
                    quote! {
                        {
                            let __oxide_input = #argument;
                            #root::__private::module_forward(
                                __oxide_cx, #path, &#receiver, __oxide_input,
                            )?
                        }
                    }
                } else {
                    let arguments: Vec<TokenStream2> = method_call
                        .args
                        .iter()
                        .map(ToTokens::to_token_stream)
                        .collect();
                    let mut bindings = Vec::new();
                    let mut temporaries = Vec::new();
                    for (index, argument) in arguments.into_iter().enumerate() {
                        let temporary = quote::format_ident!(
                            "__oxide_arg{}",
                            index,
                            span = proc_macro2::Span::call_site()
                        );
                        bindings.push(quote! { let #temporary = #argument; });
                        temporaries.push(temporary);
                    }
                    if self.output_arity == 1 {
                        quote! {
                            {
                                #(#bindings)*
                                #root::__private::single_output(
                                    #root::__private::module_forward_multi(
                                        __oxide_cx, #path, &#receiver, &[#(#temporaries),*],
                                    )?,
                                )?
                            }
                        }
                    } else {
                        quote! {
                            {
                                #(#bindings)*
                                #root::__private::module_forward_multi(
                                    __oxide_cx, #path, &#receiver, &[#(#temporaries),*],
                                )?
                            }
                        }
                    }
                };
                return Some(self.parse_expr(tokens, span));
            }

            // `self.config.some_method()` is ordinary Rust and is left alone.
            if TENSOR_OPS.contains(&method_name.as_str()) {
                self.error(
                    method_call,
                    format!(
                        "`self.{}.{method_name}()` is ambiguous inside #[model]; \
                         call the operation on a graph value instead",
                        segments
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(".")
                    ),
                );
            }
            return None;
        }

        if TENSOR_OPS.contains(&method_name.as_str()) {
            let receiver = method_call.receiver.to_token_stream();
            let mut args = vec![receiver];
            args.extend(method_call.args.iter().map(ToTokens::to_token_stream));
            return Some(self.ops_call(&method, args, span));
        }

        if method_name == "forward" {
            self.error(
                method_call,
                "calling `.forward()` on a local value is not supported inside #[model]; \
                 store the module in a struct field and call `self.<field>.forward(x)`",
            );
        }
        None
    }

    fn rewrite_binary(&mut self, binary: &ExprBinary) -> Option<Expr> {
        let operation = match binary.op {
            syn::BinOp::Add(_) => "add",
            syn::BinOp::Sub(_) => "sub",
            syn::BinOp::Mul(_) => "mul",
            syn::BinOp::Div(_) => "div",
            _ => return None,
        };
        let lhs = binary.left.to_token_stream();
        let rhs = binary.right.to_token_stream();
        let span = binary.span();
        Some(self.ops_call(&quote::format_ident!("{operation}"), vec![lhs, rhs], span))
    }
}

fn ends_with_try(expr: &Expr) -> bool {
    match expr {
        Expr::Try(_) => true,
        Expr::Block(block) => match block.block.stmts.last() {
            Some(Stmt::Expr(tail, None)) => ends_with_try(tail),
            _ => false,
        },
        Expr::Paren(paren) => ends_with_try(&paren.expr),
        Expr::Group(group) => ends_with_try(&group.expr),
        _ => false,
    }
}

fn self_path(expr: &Expr) -> Option<Vec<syn::Ident>> {
    match expr {
        Expr::Path(path) if path.path.is_ident("self") => Some(Vec::new()),
        Expr::Field(field) => {
            let mut base = self_path(&field.base)?;
            match &field.member {
                Member::Named(ident) => {
                    base.push(ident.clone());
                    Some(base)
                }
                Member::Unnamed(_) => None,
            }
        }
        Expr::Paren(paren) => self_path(&paren.expr),
        Expr::Group(group) => self_path(&group.expr),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn expand(source: proc_macro2::TokenStream) -> syn::Result<TokenStream2> {
        let method: ImplItemFn = syn::parse2(source)?;
        expand_method(method, Options::default())
    }

    fn expands(source: proc_macro2::TokenStream) -> bool {
        expand(source).is_ok()
    }

    fn error_message(source: proc_macro2::TokenStream) -> String {
        expand(source).unwrap_err().to_string()
    }

    #[test]
    fn rewrites_module_and_tensor_calls() {
        assert!(expands(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                let x = self.fc1.forward(x);
                let x = relu(x);
                self.fc2.forward(x + x)
            }
        }));
    }

    #[test]
    fn rejects_async_forward() {
        let message = error_message(quote! {
            async fn forward(&self, x: Tensor<f32>) -> Tensor<f32> { x }
        });
        assert!(message.contains("async"), "{message}");
    }

    #[test]
    fn rejects_missing_receiver() {
        let message = error_message(quote! {
            fn forward(x: Tensor<f32>) -> Tensor<f32> { x }
        });
        assert!(message.contains("&self"), "{message}");
    }

    #[test]
    fn rejects_closures() {
        let message = error_message(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                let f = |value| value;
                f(x)
            }
        });
        assert!(message.contains("closures"), "{message}");
    }

    #[test]
    fn rejects_missing_tail_expression() {
        let message = error_message(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                let y = relu(x);
            }
        });
        assert!(message.contains("output expression"), "{message}");
    }

    #[test]
    fn rejects_forward_on_locals() {
        let message = error_message(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                let layer = self.fc1;
                layer.forward(x)
            }
        });
        assert!(message.contains("local value"), "{message}");
    }

    #[test]
    fn rejects_non_identifier_parameters() {
        let message = error_message(quote! {
            fn forward(&self, (a, b): (Tensor<f32>, Tensor<f32>)) -> Tensor<f32> { a }
        });
        assert!(message.contains("identifier"), "{message}");
    }

    #[test]
    fn keeps_negative_literal_arguments() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> { x.softmax(-1) }
        })
        .unwrap()
        .to_string();
        assert!(expanded.contains("softmax"), "{expanded}");
        assert!(!expanded.contains("neg"), "{expanded}");
    }

    #[test]
    fn accepts_multi_input_and_tuple_output() {
        assert!(expands(quote! {
            fn forward(&self, a: Tensor<f32>, b: Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
                (relu(a), tanh(b))
            }
        }));
    }

    #[test]
    fn accepts_return_statements() {
        assert!(expands(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                if self.training {
                    return relu(x);
                }
                x
            }
        }));
    }

    #[test]
    fn predict_is_not_generated_when_disabled() {
        let method: ImplItemFn = parse_quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> { x }
        };
        let options = Options {
            predict: false,
            ..Default::default()
        };
        let tokens = expand_method(method, options).unwrap();
        assert!(!tokens.to_string().contains("predict"));
    }
}
