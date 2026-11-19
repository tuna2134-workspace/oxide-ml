//! `graph!` declarative graph construction.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Error, Expr, Ident, Result, Token, bracketed, parenthesized, parse_macro_input};

pub fn expand(input: TokenStream) -> TokenStream {
    let spec = parse_macro_input!(input as GraphSpec);
    match spec.expand() {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

enum Item {
    Input {
        name: Ident,
        dims: Vec<usize>,
    },
    Param {
        name: Ident,
        dims: Vec<usize>,
        trainable: bool,
    },
    Let {
        name: Ident,
        op: Ident,
        args: Vec<Expr>,
    },
}

struct GraphSpec {
    items: Vec<Item>,
    outputs: Vec<Ident>,
}

impl Parse for GraphSpec {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut items = Vec::new();
        let mut outputs = Vec::new();
        while !input.is_empty() {
            if input.peek(Token![let]) {
                input.parse::<Token![let]>()?;
                let name: Ident = input.parse()?;
                input.parse::<Token![=]>()?;
                let op: Ident = input.parse()?;
                let content;
                parenthesized!(content in input);
                let args = Punctuated::<Expr, Token![,]>::parse_terminated(&content)?
                    .into_iter()
                    .collect();
                input.parse::<Token![;]>()?;
                items.push(Item::Let { name, op, args });
                continue;
            }

            let keyword: Ident = input.parse()?;
            match keyword.to_string().as_str() {
                "input" => {
                    let name: Ident = input.parse()?;
                    input.parse::<Token![:]>()?;
                    let content;
                    bracketed!(content in input);
                    let dims = parse_dims(&content)?;
                    input.parse::<Token![;]>()?;
                    items.push(Item::Input { name, dims });
                }
                "param" => {
                    let name: Ident = input.parse()?;
                    input.parse::<Token![:]>()?;
                    let content;
                    bracketed!(content in input);
                    let dims = parse_dims(&content)?;
                    let mut trainable = true;
                    if input.peek(Ident) {
                        let modifier: Ident = input.parse()?;
                        match modifier.to_string().as_str() {
                            "trainable" => trainable = true,
                            "frozen" => trainable = false,
                            other => {
                                return Err(Error::new_spanned(
                                    modifier,
                                    format!(
                                        "unknown parameter modifier {other:?}; \
                                         expected `trainable` or `frozen`"
                                    ),
                                ));
                            }
                        }
                    }
                    input.parse::<Token![;]>()?;
                    items.push(Item::Param {
                        name,
                        dims,
                        trainable,
                    });
                }
                "output" => {
                    outputs.push(input.parse()?);
                    input.parse::<Token![;]>()?;
                }
                "outputs" => {
                    let content;
                    bracketed!(content in input);
                    let names = Punctuated::<Ident, Token![,]>::parse_terminated(&content)?;
                    outputs.extend(names);
                    input.parse::<Token![;]>()?;
                }
                other => {
                    return Err(Error::new_spanned(
                        keyword,
                        format!(
                            "unknown graph! statement {other:?}; expected \
                             `input`, `param`, `let`, `output` or `outputs`"
                        ),
                    ));
                }
            }
        }
        if outputs.is_empty() {
            return Err(Error::new(
                proc_macro2::Span::call_site(),
                "graph! requires at least one `output`",
            ));
        }
        Ok(GraphSpec { items, outputs })
    }
}

fn parse_dims(input: ParseStream<'_>) -> Result<Vec<usize>> {
    let dims = Punctuated::<syn::LitInt, Token![,]>::parse_terminated(input)?;
    dims.into_iter()
        .map(|literal| literal.base10_parse::<usize>())
        .collect()
}

impl GraphSpec {
    fn expand(self) -> Result<TokenStream2> {
        let mut statements = Vec::new();
        for item in &self.items {
            match item {
                Item::Input { name, dims } => {
                    let literal = name.to_string();
                    statements.push(quote! {
                        let #name = __oxide_builder.input(
                            #literal,
                            __oxide_shape(&[#(#dims),*]),
                            __oxide_dtype(),
                        );
                    });
                }
                Item::Param {
                    name,
                    dims,
                    trainable,
                } => {
                    let literal = name.to_string();
                    statements.push(quote! {
                        let #name = __oxide_builder.try_parameter(
                            #literal,
                            __oxide_shape(&[#(#dims),*]),
                            __oxide_dtype(),
                            #trainable,
                        )?;
                    });
                }
                Item::Let { name, op, args } => {
                    let call = builder_call(op, args)?;
                    statements.push(quote! {
                        let #name = #call;
                    });
                }
            }
        }
        let outputs = self.outputs.iter().map(|name| quote! { #name });
        Ok(quote! {
            (|| -> ::oxide_ml::__private::Result<::oxide_ml::__private::Graph> {
                fn __oxide_shape(dims: &[usize]) -> ::oxide_ml::__private::Shape {
                    ::oxide_ml::__private::Shape::from_static(dims)
                }
                fn __oxide_dtype() -> ::oxide_ml::__private::DType {
                    ::oxide_ml::__private::DType::F32
                }
                let mut __oxide_builder = ::oxide_ml::__private::GraphBuilder::new();
                #(#statements)*
                __oxide_builder.finish(&[#(#outputs),*])
            })()
        })
    }
}

fn arg<'a>(args: &'a [Expr], index: usize, op: &Ident) -> Result<&'a Expr> {
    args.get(index).ok_or_else(|| {
        Error::new_spanned(
            op,
            format!(
                "graph! operation {:?} expects at least {} argument(s)",
                op.to_string(),
                index + 1
            ),
        )
    })
}

fn array_to_vec(expr: &Expr, op: &Ident) -> Result<TokenStream2> {
    match expr {
        Expr::Array(array) => {
            let elems: Vec<&Expr> = array.elems.iter().collect();
            Ok(quote! { ::std::vec![#(#elems),*] })
        }
        other => Err(Error::new_spanned(
            other,
            format!(
                "graph! operation {:?} expects an array literal such as `[1, 0]`",
                op.to_string()
            ),
        )),
    }
}

fn builder_call(op: &Ident, args: &[Expr]) -> Result<TokenStream2> {
    let name = op.to_string();
    let result = match name.as_str() {
        "neg" | "relu" | "gelu" | "sigmoid" | "tanh" | "sqrt" | "exp" | "log" | "abs" => {
            let value = arg(args, 0, op)?;
            quote! { __oxide_builder.#op(#value)? }
        }
        "add" | "sub" | "mul" | "div" | "matmul" => {
            let lhs = arg(args, 0, op)?;
            let rhs = arg(args, 1, op)?;
            quote! { __oxide_builder.#op(#lhs, #rhs)? }
        }
        "softmax" => {
            let value = arg(args, 0, op)?;
            let axis = arg(args, 1, op)?;
            quote! { __oxide_builder.softmax(#value, #axis)? }
        }
        "log_softmax" => {
            let value = arg(args, 0, op)?;
            let axis = arg(args, 1, op)?;
            quote! { __oxide_builder.log_softmax(#value, #axis)? }
        }
        "reshape" => {
            let value = arg(args, 0, op)?;
            let dims = arg(args, 1, op)?;
            quote! { __oxide_builder.reshape_dims(#value, #dims)? }
        }
        "transpose" => {
            let value = arg(args, 0, op)?;
            let axes = array_to_vec(arg(args, 1, op)?, op)?;
            quote! { __oxide_builder.transpose(#value, #axes)? }
        }
        "mean" | "sum" => {
            let value = arg(args, 0, op)?;
            let axes = array_to_vec(arg(args, 1, op)?, op)?;
            let keep = args.get(2);
            let method = if name == "mean" {
                quote! { reduce_mean }
            } else {
                quote! { reduce_sum }
            };
            match keep {
                Some(keep) => quote! { __oxide_builder.#method(#value, #axes, #keep)? },
                None => quote! { __oxide_builder.#method(#value, #axes, false)? },
            }
        }
        "slice" => {
            let value = arg(args, 0, op)?;
            let axis = arg(args, 1, op)?;
            let start = arg(args, 2, op)?;
            let len = arg(args, 3, op)?;
            quote! { __oxide_builder.slice(#value, #axis, #start, #len)? }
        }
        "embedding" => {
            let weight = arg(args, 0, op)?;
            let indices = arg(args, 1, op)?;
            quote! { __oxide_builder.embedding(#weight, #indices)? }
        }
        "mse_loss" => {
            let prediction = arg(args, 0, op)?;
            let target = arg(args, 1, op)?;
            quote! {
                __oxide_builder.mse_loss(
                    #prediction,
                    #target,
                    ::oxide_ml::__private::Reduction::Mean,
                )?
            }
        }
        "cross_entropy_loss" => {
            let prediction = arg(args, 0, op)?;
            let target = arg(args, 1, op)?;
            quote! {
                __oxide_builder.cross_entropy_loss(
                    #prediction,
                    #target,
                    ::oxide_ml::__private::Reduction::Mean,
                )?
            }
        }
        "layer_norm" => {
            let value = arg(args, 0, op)?;
            let gamma = arg(args, 1, op)?;
            let beta = arg(args, 2, op)?;
            let shape = arg(args, 3, op)?;
            let eps = arg(args, 4, op)?;
            quote! { __oxide_builder.layer_norm(#value, #gamma, #beta, #shape, #eps)? }
        }
        other => {
            return Err(Error::new_spanned(
                op,
                format!(
                    "unsupported graph! operation {other:?}; supported operations are \
                     add, sub, mul, div, matmul, neg, relu, gelu, sigmoid, tanh, sqrt, exp, \
                     log, abs, softmax, log_softmax, reshape, transpose, mean, sum, slice, \
                     embedding, mse_loss, cross_entropy_loss, layer_norm"
                ),
            ));
        }
    };
    Ok(result)
}
