//! `#[model]`: rewrite a forward method into graph-building code.

use std::collections::HashMap;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{ToTokens, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{
    Block, Error, Expr, ExprBinary, ExprCall, ExprForLoop, ExprMethodCall, FnArg, ImplItemFn,
    Member, Pat, ReturnType, Stmt, Type, parse_macro_input, parse_quote, spanned::Spanned,
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
    "parameter",
    "split_heads",
    "merge_heads",
    "apply_rotary",
    "grouped_attention",
    "grouped_attention_masked",
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

    struct ModelInput {
        ident: syn::Ident,
        pattern: Pat,
        ty: syn::Type,
        optional: bool,
    }

    let mut model_inputs = Vec::new();
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
        let optional = tensor_parameter_kind(&typed.ty).ok_or_else(|| {
            Error::new_spanned(
                &typed.ty,
                "#[model] parameters must be `Tensor<f32>` or `Option<Tensor<f32>>`",
            )
        })?;
        model_inputs.push(ModelInput {
            ident: pattern.ident.clone(),
            pattern: typed.pat.as_ref().clone(),
            ty: (*typed.ty).clone(),
            optional,
        });
    }
    let input_idents: Vec<syn::Ident> = model_inputs
        .iter()
        .map(|input| input.ident.clone())
        .collect();
    let input_patterns: Vec<Pat> = model_inputs
        .iter()
        .map(|input| input.pattern.clone())
        .collect();
    let input_types: Vec<syn::Type> = model_inputs.iter().map(|input| input.ty.clone()).collect();
    let has_optional = model_inputs.iter().any(|input| input.optional);
    let arity = model_inputs.len();
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
            locals: HashMap::new(),
            loop_index: 0,
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

    let single_forward = if arity == 1 && !has_optional {
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

    let slots_bindings = model_inputs.iter().enumerate().map(|(index, input)| {
        let ident = &input.ident;
        let position = syn::Index::from(index);
        if input.optional {
            quote! { let #ident = #inputs[#position]; }
        } else {
            quote! {
                let #ident = #inputs[#position].ok_or_else(|| {
                    #root::__private::Error::invalid_graph(::std::format!(
                        "model {} requires input {} ({})",
                        stringify!(#forward_name),
                        #position,
                        stringify!(#ident),
                    ))
                })?;
            }
        }
    });

    let slots_method = if has_optional {
        quote! {
            #[allow(clippy::clone_on_copy, clippy::redundant_clone)]
            #vis fn forward_slots(
                &self,
                #cx: &mut #root::__private::ModelContext,
                #inputs: &[::std::option::Option<#root::__private::ValueId>],
            ) -> #root::__private::Result<::std::vec::Vec<#root::__private::ValueId>> {
                if #inputs.len() != #arity {
                    return ::std::result::Result::Err(
                        #root::__private::Error::invalid_graph(::std::format!(
                            "model {} expects {} input slots, got {}",
                            stringify!(#forward_name),
                            #arity,
                            #inputs.len(),
                        )),
                    );
                }
                #(#slots_bindings)*
                #block
            }
        }
    } else {
        TokenStream2::new()
    };

    let multi_method = if has_optional {
        quote! {
            #[allow(clippy::clone_on_copy, clippy::redundant_clone)]
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
                let __oxide_slots: ::std::vec::Vec<::std::option::Option<#root::__private::ValueId>> =
                    #inputs.iter().map(|value| ::std::option::Option::Some(*value)).collect();
                self.forward_slots(#cx, &__oxide_slots)
            }
        }
    } else {
        quote! {
            #[allow(clippy::clone_on_copy, clippy::redundant_clone)]
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
        }
    };

    let graph_methods = quote! {
        #single_forward
        #slots_method
        #multi_method
    };

    let predict_method = if options.predict && !has_optional {
        let unwrap = predict_unwrap(root, output_arity);
        quote! {
            #[allow(clippy::clone_on_copy, clippy::redundant_clone)]
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

/// `Some(true)` for `Option<Tensor<..>>`, `Some(false)` for `Tensor<..>`.
fn tensor_parameter_kind(ty: &syn::Type) -> Option<bool> {
    match ty {
        syn::Type::Paren(paren) => tensor_parameter_kind(&paren.elem),
        syn::Type::Group(group) => tensor_parameter_kind(&group.elem),
        syn::Type::Path(path) => {
            let segment = path.path.segments.last()?;
            match segment.ident.to_string().as_str() {
                "Tensor" => Some(false),
                "Option" => {
                    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
                        return None;
                    };
                    let inner = args.args.iter().find_map(|arg| match arg {
                        syn::GenericArgument::Type(ty) => Some(ty),
                        _ => None,
                    })?;
                    match tensor_parameter_kind(inner)? {
                        false => Some(true),
                        true => None,
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
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

/// A loop-bound module whose parameter path is only known while building the
/// graph (for example `for layer in &self.layers`).
struct LocalModule {
    /// Rust expression evaluating to the module's path string.
    path: TokenStream2,
}

struct LoopSpec {
    /// Expression yielding the collection (e.g. `self.layers`).
    collection: TokenStream2,
    /// Rust expression evaluating to the parent path (a `&str`).
    base_path: TokenStream2,
    /// Loop variable bound to each module.
    module: syn::Ident,
    /// Index variable used in the module path.
    index: syn::Ident,
    /// Whether the user pattern already destructures `(index, module)`.
    tuple_pattern: bool,
}

struct Rewriter {
    root: syn::Path,
    output_arity: usize,
    errors: Vec<Error>,
    locals: HashMap<String, LocalModule>,
    loop_index: usize,
}

impl Rewriter {
    fn error(&mut self, tokens: impl ToTokens, message: impl std::fmt::Display) {
        self.errors.push(Error::new_spanned(tokens, message));
    }

    /// Detect a `for` loop over a module collection and rewrite its body so
    /// `.forward()` calls on the loop variable resolve their parameter paths.
    ///
    /// Supported iterables: `&self.layers`, `self.layers`, `self.layers.iter()`,
    /// `self.layers.iter().enumerate()`, and the same shapes rooted at a module
    /// bound by an enclosing module loop. Unrelated loops are left untouched.
    fn rewrite_module_loop(&mut self, for_loop: &mut ExprForLoop) -> bool {
        let Some(spec) = self.loop_spec(for_loop) else {
            return false;
        };
        let spec = match spec {
            Ok(spec) => spec,
            Err(error) => {
                self.errors.push(error);
                return true;
            }
        };
        if !uses_module_forward(&for_loop.body, &spec.module) {
            return false;
        }
        let path_var = quote::format_ident!("__oxide_module_path_{}", self.loop_index);
        let key = spec.module.to_string();
        let previous = self.locals.insert(
            key.clone(),
            LocalModule {
                path: quote!(#path_var),
            },
        );
        self.visit_block_mut(&mut for_loop.body);
        match previous {
            Some(previous) => {
                self.locals.insert(key, previous);
            }
            None => {
                self.locals.remove(&key);
            }
        }

        let collection = spec.collection;
        *for_loop.expr = parse_quote!(#collection.iter().enumerate());
        if !spec.tuple_pattern {
            let pattern = for_loop.pat.clone();
            let index = spec.index.clone();
            *for_loop.pat = parse_quote!((#index, #pattern));
        }
        let base_path = spec.base_path;
        let index = spec.index;
        let statement: Stmt = parse_quote!(
            let #path_var: ::std::string::String =
                ::std::format!("{}.{}", #base_path, #index);
        );
        for_loop.body.stmts.insert(0, statement);
        true
    }

    fn loop_spec(&mut self, for_loop: &ExprForLoop) -> Option<syn::Result<LoopSpec>> {
        self.loop_index += 1;
        let mut expression: &Expr = &for_loop.expr;
        while let Expr::Paren(paren) = expression {
            expression = &paren.expr;
        }
        if let Expr::MethodCall(call) = expression {
            if call.method == "enumerate" {
                let mut receiver: &Expr = &call.receiver;
                while let Expr::Paren(paren) = receiver {
                    receiver = &paren.expr;
                }
                let Expr::MethodCall(inner) = receiver else {
                    return None;
                };
                if inner.method != "iter" {
                    return None;
                }
                expression = &inner.receiver;
            } else if call.method == "iter" {
                expression = &call.receiver;
            }
        } else if let Expr::Reference(reference) = expression {
            expression = &reference.expr;
        }
        while let Expr::Paren(paren) = expression {
            expression = &paren.expr;
        }

        let (collection, base_path) = if let Some(segments) = self_path(expression) {
            if segments.is_empty() {
                return None;
            }
            let joined = segments.join(".");
            (quote!(#expression), quote!(#joined))
        } else if let Some((root, suffix)) = local_path(expression) {
            let binding = self.locals.get(&root.to_string())?.path.clone();
            let path = if suffix.is_empty() {
                quote!(#binding)
            } else {
                let suffix = suffix.join(".");
                quote!(::std::format!("{}.{}", #binding, #suffix))
            };
            (quote!(#expression), path)
        } else {
            return None;
        };

        let (module, index, tuple_pattern) = match &*for_loop.pat {
            Pat::Ident(pattern) => {
                let index = quote::format_ident!("__oxide_loop_index_{}", self.loop_index);
                (pattern.ident.clone(), index, false)
            }
            Pat::Tuple(tuple) if tuple.elems.len() == 2 => {
                let mut idents = Vec::new();
                for element in &tuple.elems {
                    match element {
                        Pat::Ident(pattern) => idents.push(pattern.ident.clone()),
                        other => {
                            return Some(Err(Error::new_spanned(
                                other,
                                "#[model] module loops accept `for layer in ...` or \
                                 `for (index, layer) in ...`",
                            )));
                        }
                    }
                }
                (idents[1].clone(), idents[0].clone(), true)
            }
            other => {
                return Some(Err(Error::new_spanned(
                    other,
                    "#[model] module loops accept `for layer in ...` or \
                     `for (index, layer) in ...`",
                )));
            }
        };

        Some(Ok(LoopSpec {
            collection,
            base_path,
            module,
            index,
            tuple_pattern,
        }))
    }

    /// Rewrite a module `forward`/`forward_multi` call into a graph-building
    /// call. `path_ref` evaluates to the parameter path and `receiver_ref` is a
    /// reference to the module.
    fn module_call(
        &mut self,
        method_call: &ExprMethodCall,
        path_ref: TokenStream2,
        receiver_ref: TokenStream2,
        outputs: Option<usize>,
        span: proc_macro2::Span,
    ) -> Option<Expr> {
        let method_name = method_call.method.to_string();
        let root = &self.root;
        let arguments: Vec<TokenStream2> = method_call
            .args
            .iter()
            .map(ToTokens::to_token_stream)
            .collect();
        if arguments.is_empty() {
            self.error(
                method_call,
                "module calls need at least one input tensor (`self.fc.forward(x)`)",
            );
            return None;
        }
        if let Some(count) = outputs
            && !(2..=4).contains(&count)
        {
            self.error(
                method_call,
                format!(
                    "#[model] can destructure 2, 3 or 4 module outputs into a tuple, not {count}; \
                     call `.forward_multi(...)` and index the result for other arities"
                ),
            );
            return None;
        }

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

        if method_name == "forward_slots" {
            let call = quote! {
                #root::__private::module_forward_slots(
                    __oxide_cx, #path_ref, #receiver_ref, &[#(#temporaries),*],
                )?
            };
            let tokens = match outputs {
                Some(2) => quote! {{ #(#bindings)* #root::__private::output_pair(#call)? }},
                Some(3) => quote! {{ #(#bindings)* #root::__private::output_triple(#call)? }},
                Some(4) => quote! {{ #(#bindings)* #root::__private::output_quad(#call)? }},
                Some(_) => unreachable!("validated above"),
                None => quote! {{ #(#bindings)* #root::__private::single_output(#call)? }},
            };
            return Some(self.parse_expr(tokens, span));
        }

        // A plain single-input `.forward(x)` uses the module's single-input
        // entry point. Everything else goes through `forward_multi`, which the
        // trait derives from `forward` for single-input modules.
        let single_output_forward =
            method_name == "forward" && temporaries.len() == 1 && outputs.is_none();
        let call = if single_output_forward {
            let argument = temporaries[0].clone();
            quote! {
                #root::__private::module_forward(
                    __oxide_cx, #path_ref, #receiver_ref, #argument,
                )?
            }
        } else {
            quote! {
                #root::__private::module_forward_multi(
                    __oxide_cx, #path_ref, #receiver_ref, &[#(#temporaries),*],
                )?
            }
        };

        let tokens = match outputs {
            Some(2) => quote! {
                {
                    #(#bindings)*
                    #root::__private::output_pair(#call)?
                }
            },
            Some(3) => quote! {
                {
                    #(#bindings)*
                    #root::__private::output_triple(#call)?
                }
            },
            Some(4) => quote! {
                {
                    #(#bindings)*
                    #root::__private::output_quad(#call)?
                }
            },
            Some(_) => unreachable!("validated above"),
            None if single_output_forward => quote! {
                {
                    #(#bindings)*
                    #call
                }
            },
            None => quote! {
                {
                    #(#bindings)*
                    #root::__private::single_output(#call)?
                }
            },
        };
        Some(self.parse_expr(tokens, span))
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
    fn visit_stmt_mut(&mut self, stmt: &mut Stmt) {
        // Multi-output destructuring: `let (hidden, weights) = self.attn.forward(x, mask);`
        // The arity comes from the tuple pattern, so nested module calls in the
        // arguments keep their own single-output semantics.
        if let Stmt::Local(local) = stmt {
            let arity = match &local.pat {
                Pat::Tuple(tuple) => tuple.elems.len(),
                _ => 0,
            };
            if arity >= 2
                && let Some(init) = local.init.as_mut()
                && let Expr::MethodCall(call) = init.expr.as_mut()
                && (call.method == "forward"
                    || call.method == "forward_multi"
                    || call.method == "forward_slots")
            {
                for argument in &mut call.args {
                    self.visit_expr_mut(argument);
                }
                if let Some((path_ref, receiver_ref)) = self.module_target(call) {
                    let span = call.span();
                    if let Some(expr) =
                        self.module_call(call, path_ref, receiver_ref, Some(arity), span)
                    {
                        *init.expr = expr;
                    }
                    return;
                }
            }
        }
        syn::visit_mut::visit_stmt_mut(self, stmt);
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::ForLoop(for_loop) = expr {
            if self.rewrite_module_loop(for_loop) {
                return;
            }
        }
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

    fn module_target(
        &mut self,
        method_call: &ExprMethodCall,
    ) -> Option<(TokenStream2, TokenStream2)> {
        let method_name = method_call.method.to_string();
        if method_name != "forward"
            && method_name != "forward_multi"
            && method_name != "forward_slots"
        {
            return None;
        }
        if let Some(segments) = self_path(&method_call.receiver) {
            if segments.is_empty() {
                self.error(
                    method_call,
                    "calling `.forward()` on `self` inside #[model] would recurse; \
                     call a module field instead (`self.fc.forward(x)`)",
                );
                return None;
            }
            let receiver = method_call.receiver.to_token_stream();
            let path = segments.join(".");
            return Some((quote!(#path), quote!(&#receiver)));
        }
        if let Some((root, suffix)) = local_path(&method_call.receiver)
            && let Some(local) = self.locals.get(&root.to_string())
        {
            let binding = local.path.clone();
            let receiver = method_call.receiver.to_token_stream();
            let (path_ref, receiver_ref) = if suffix.is_empty() {
                (quote!(&(#binding)), quote!(#receiver))
            } else {
                let suffix = suffix.join(".");
                (
                    quote!(&(::std::format!("{}.{}", #binding, #suffix))),
                    quote!(&#receiver),
                )
            };
            return Some((path_ref, receiver_ref));
        }
        None
    }

    fn rewrite_method_call(&mut self, method_call: &ExprMethodCall) -> Option<Expr> {
        let method = method_call.method.clone();
        let method_name = method.to_string();
        let span = method_call.span();
        let is_forward = method_name == "forward" || method_name == "forward_multi";

        if let Some((path_ref, receiver_ref)) = self.module_target(method_call) {
            return self.module_call(method_call, path_ref, receiver_ref, None, span);
        }

        if let Some(segments) = self_path(&method_call.receiver) {
            // `self.config.some_method()` is ordinary Rust and is left alone,
            // but a tensor operation through a field would be ambiguous.
            if TENSOR_OPS.contains(&method_name.as_str()) {
                self.error(
                    method_call,
                    format!(
                        "`self.{}.{method_name}()` is ambiguous inside #[model]; \
                         call the operation on a graph value instead",
                        segments.join(".")
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

        if is_forward {
            self.error(
                method_call,
                "calling `.forward()` on a local value is not supported inside #[model] \
                 unless the local is bound by a loop over a module collection; store the \
                 module in a struct field or iterate over `self.<field>`",
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

/// Whether a loop body calls `forward`/`forward_multi` on the loop variable.
///
/// Ordinary `for` loops over configuration fields must keep their element
/// binding, so a loop is only rewritten when it actually forwards a module.
fn uses_module_forward(body: &Block, name: &syn::Ident) -> bool {
    struct ForwardUse<'a> {
        name: &'a syn::Ident,
        found: bool,
    }

    impl<'ast> Visit<'ast> for ForwardUse<'_> {
        fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
            if (call.method == "forward"
                || call.method == "forward_multi"
                || call.method == "forward_slots")
                && let Some((root, _)) = local_path(&call.receiver)
                && root == *self.name
            {
                self.found = true;
            }
            syn::visit::visit_expr_method_call(self, call);
        }

        fn visit_expr_for_loop(&mut self, nested: &'ast ExprForLoop) {
            // A nested module loop iterates `&block.layers`, so the outer loop
            // must be rewritten even though it does not forward directly.
            if let Some((root, _)) = iterable_local_root(&nested.expr)
                && root == *self.name
            {
                self.found = true;
            }
            syn::visit::visit_expr_for_loop(self, nested);
        }
    }

    let mut visitor = ForwardUse { name, found: false };
    visitor.visit_block(body);
    visitor.found
}

/// Root local identifier of a loop iterable (`&block.layers` -> `block`).
fn iterable_local_root(expr: &Expr) -> Option<(syn::Ident, Vec<String>)> {
    let mut expression = expr;
    loop {
        match expression {
            Expr::Paren(paren) => expression = &paren.expr,
            Expr::Group(group) => expression = &group.expr,
            Expr::Reference(reference) => expression = &reference.expr,
            Expr::MethodCall(call) if call.method == "iter" || call.method == "enumerate" => {
                expression = &call.receiver;
            }
            _ => break,
        }
    }
    local_path(expression)
}

/// Resolve `layer.attn` to a root local identifier plus path segments.
///
/// Segments are strings so tuple fields (`layer.0`) and named fields share the
/// same parameter-path rules.
fn local_path(expr: &Expr) -> Option<(syn::Ident, Vec<String>)> {
    match expr {
        Expr::Path(path) => {
            let ident = path.path.get_ident()?;
            Some((ident.clone(), Vec::new()))
        }
        Expr::Field(field) => {
            let (root, mut segments) = local_path(&field.base)?;
            segments.push(member_segment(&field.member)?);
            Some((root, segments))
        }
        Expr::Paren(paren) => local_path(&paren.expr),
        Expr::Group(group) => local_path(&group.expr),
        _ => None,
    }
}

fn self_path(expr: &Expr) -> Option<Vec<String>> {
    match expr {
        Expr::Path(path) if path.path.is_ident("self") => Some(Vec::new()),
        Expr::Field(field) => {
            let mut base = self_path(&field.base)?;
            base.push(member_segment(&field.member)?);
            Some(base)
        }
        Expr::Paren(paren) => self_path(&paren.expr),
        Expr::Group(group) => self_path(&group.expr),
        _ => None,
    }
}

/// Path segment for a named or tuple field.
fn member_segment(member: &Member) -> Option<String> {
    match member {
        Member::Named(ident) => Some(ident.to_string()),
        Member::Unnamed(index) => Some(index.index.to_string()),
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
    fn rewrites_module_collection_loops() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                let mut hidden = x;
                for layer in &self.layers {
                    hidden = layer.forward(hidden);
                }
                hidden
            }
        })
        .unwrap()
        .to_string();
        assert!(expanded.contains("enumerate"), "{expanded}");
        assert!(expanded.contains("module_forward"), "{expanded}");
        assert!(expanded.contains("__oxide_module_path_1"), "{expanded}");
    }

    #[test]
    fn rewrites_tuple_module_loops() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                let mut hidden = x;
                for (index, layer) in self.layers.iter().enumerate() {
                    hidden = layer.forward(hidden);
                }
                hidden
            }
        })
        .unwrap()
        .to_string();
        assert!(expanded.contains("module_forward"), "{expanded}");
        assert!(!expanded.contains("__oxide_loop_index"), "{expanded}");
    }

    #[test]
    fn rewrites_nested_module_loops() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                let mut hidden = x;
                for block in &self.blocks {
                    for layer in &block.layers {
                        hidden = layer.forward(hidden);
                    }
                }
                hidden
            }
        })
        .unwrap()
        .to_string();
        assert!(expanded.contains("__oxide_module_path_2"), "{expanded}");
    }

    #[test]
    fn leaves_plain_configuration_loops_alone() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                let mut total = 0usize;
                for value in &self.config.heads {
                    total += *value;
                }
                let _ = total;
                x
            }
        })
        .unwrap()
        .to_string();
        assert!(!expanded.contains("enumerate"), "{expanded}");
        assert!(!expanded.contains("__oxide_module_path"), "{expanded}");
    }

    #[test]
    fn rejects_module_loops_with_complex_patterns() {
        let message = error_message(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                for Layer { .. } in &self.layers {
                    x;
                }
                x
            }
        });
        assert!(message.contains("module loops"), "{message}");
    }

    #[test]
    fn rewrites_multi_input_module_calls() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>, mask: Tensor<f32>) -> Tensor<f32> {
                self.attention.forward(hidden, mask)
            }
        })
        .unwrap()
        .to_string();
        assert!(expanded.contains("module_forward_multi"), "{expanded}");
        assert!(!expanded.contains("module_forward ("), "{expanded}");
    }

    #[test]
    fn rewrites_tuple_destructuring_of_module_outputs() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
                let (hidden, weights) = self.attention.forward(x);
                (hidden, weights)
            }
        })
        .unwrap()
        .to_string();
        assert!(expanded.contains("output_pair"), "{expanded}");
    }

    #[test]
    fn forward_multi_returns_a_single_tensor_when_not_destructured() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                self.attention.forward_multi(x, x)
            }
        })
        .unwrap()
        .to_string();
        assert!(expanded.contains("single_output"), "{expanded}");
    }

    #[test]
    fn tuple_fields_are_module_paths() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                self.0.forward(x)
            }
        })
        .unwrap()
        .to_string();
        assert!(expanded.contains("module_forward"), "{expanded}");
    }

    #[test]
    fn rejects_five_output_destructuring() {
        let message = error_message(quote! {
            fn forward(&self, x: Tensor<f32>) -> Tensor<f32> {
                let (a, b, c, d, e) = self.attention.forward(x);
                a
            }
        });
        assert!(message.contains("2, 3 or 4"), "{message}");
    }

    #[test]
    fn rewrites_forward_slots_with_tuple_outputs() {
        let expanded = expand(quote! {
            fn forward(&self, x: Tensor<f32>) -> (Tensor<f32>, Tensor<f32>, Tensor<f32>) {
                let (a, b, c) = self.attn.forward_slots(Some(x), None, None);
                (a, b, c)
            }
        })
        .unwrap()
        .to_string();
        assert!(expanded.contains("module_forward_slots"), "{expanded}");
        assert!(expanded.contains("output_triple"), "{expanded}");
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
