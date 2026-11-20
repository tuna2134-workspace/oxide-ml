//! `#[derive(Module)]`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DeriveInput, Error, Fields, Ident, Result, parse_macro_input, parse_quote};

use crate::attr::{FieldOptions, Options};

struct ModuleField {
    /// Field access after `self.`, e.g. `fc1` or `0`.
    access: TokenStream2,
    ty: syn::Type,
    segment: String,
}

struct ParamField {
    access: TokenStream2,
    segment: String,
    role: Ident,
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand(input: DeriveInput) -> Result<TokenStream2> {
    let options = Options::parse_oxide_attrs(&input.attrs)?;
    let root = &options.root;
    let name = &input.ident;
    let module_name = options
        .module_name
        .clone()
        .unwrap_or_else(|| name.to_string());

    let entries: Vec<(syn::Member, &syn::Field)> = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => named
                .named
                .iter()
                .map(|field| {
                    let ident = field
                        .ident
                        .clone()
                        .ok_or_else(|| Error::new_spanned(field, "expected a named field"))?;
                    Ok((syn::Member::Named(ident), field))
                })
                .collect::<Result<Vec<_>>>()?,
            Fields::Unnamed(unnamed) => unnamed
                .unnamed
                .iter()
                .enumerate()
                .map(|(index, field)| (syn::Member::Unnamed(syn::Index::from(index)), field))
                .collect(),
            Fields::Unit => {
                return Err(Error::new_spanned(
                    name,
                    "#[derive(Module)] requires a struct with fields; \
                     implement Module manually for unit structs",
                ));
            }
        },
        Data::Enum(data) => return expand_enum(&input, options, data),
        Data::Union(_) => {
            return Err(Error::new_spanned(
                name,
                "#[derive(Module)] cannot be applied to unions",
            ));
        }
    };

    let mut module_fields = Vec::new();
    let mut param_fields = Vec::new();
    for (member, field) in entries {
        let segment_name = match &member {
            syn::Member::Named(ident) => ident.to_string(),
            syn::Member::Unnamed(index) => index.index.to_string(),
        };
        let field_options = FieldOptions::parse(&field.attrs)?;
        if field_options.skip {
            continue;
        }
        let segment = field_options.rename.clone().unwrap_or(segment_name);
        let access = quote!(#member);
        if field_options.param {
            let role = field_options.role.ident();
            param_fields.push(ParamField {
                access,
                segment,
                role,
            });
        } else {
            module_fields.push(ModuleField {
                access,
                ty: field.ty.clone(),
                segment,
            });
        }
    }

    if module_fields.is_empty() && param_fields.is_empty() {
        return Err(Error::new_spanned(
            name,
            "module has no fields; add modules, #[param] tensors or implement Module manually",
        ));
    }

    // Extend the where-clause with `Field: Module` bounds so generic modules
    // type-check without the user spelling every bound.
    let mut generics = input.generics.clone();
    if !generics.params.is_empty() {
        let where_clause = generics.make_where_clause();
        for field in &module_fields {
            let ty = &field.ty;
            where_clause
                .predicates
                .push(parse_quote!(#ty: #root::__private::Module));
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let forward_multi_name = &options.forward_multi;

    let child_discovery = module_fields.iter().map(|field| {
        let access = &field.access;
        let segment = &field.segment;
        quote! {
            #root::__private::Module::visit_parameters(
                &self.#access,
                &__path.child(#segment),
                __registry,
            )?;
        }
    });
    let param_discovery = param_fields.iter().map(|field| {
        let access = &field.access;
        let segment = &field.segment;
        let role = &field.role;
        quote! {
            __registry.insert_tensor(
                &__path.child(#segment),
                &self.#access,
                #root::__private::ParameterRole::#role,
            )?;
        }
    });

    let child_set_training = module_fields.iter().map(|field| {
        let access = &field.access;
        quote! {
            #root::__private::Module::set_training(&mut self.#access, __training);
        }
    });

    let child_visit_tensors = module_fields.iter().map(|field| {
        let access = &field.access;
        let segment = &field.segment;
        quote! {
            #root::__private::Module::visit_tensors(
                &self.#access,
                &__path.child(#segment),
                __visitor,
            )?;
        }
    });
    let param_visit_tensors = param_fields.iter().map(|field| {
        let access = &field.access;
        let segment = &field.segment;
        let role = &field.role;
        quote! {
            __visitor(
                &__path.child(#segment),
                &self.#access,
                #root::__private::ParameterRole::#role,
            )?;
        }
    });

    let child_visit_tensors_mut = module_fields.iter().map(|field| {
        let access = &field.access;
        let segment = &field.segment;
        quote! {
            #root::__private::Module::visit_tensors_mut(
                &mut self.#access,
                &__path.child(#segment),
                __visitor,
            )?;
        }
    });
    let param_visit_tensors_mut = param_fields.iter().map(|field| {
        let access = &field.access;
        let segment = &field.segment;
        let role = &field.role;
        quote! {
            __visitor(
                &__path.child(#segment),
                &mut self.#access,
                #root::__private::ParameterRole::#role,
            )?;
        }
    });

    let child_describe = module_fields.iter().map(|field| {
        let access = &field.access;
        let segment = &field.segment;
        quote! {
            #root::__private::Module::describe(
                &self.#access,
                &__path.child(#segment),
                &mut __node,
            );
        }
    });

    let expanded = quote! {
        #[automatically_derived]
        impl #impl_generics #root::__private::Module for #name #ty_generics #where_clause {
            fn module_type(&self) -> &'static str {
                #module_name
            }

            fn visit_parameters(
                &self,
                __path: &#root::__private::ModulePath,
                __registry: &mut #root::__private::ParameterRegistry,
            ) -> #root::__private::Result<()> {
                #(#child_discovery)*
                #(#param_discovery)*
                Ok(())
            }

            fn forward(
                &self,
                __cx: &mut #root::__private::ModelContext,
                __input: #root::__private::ValueId,
            ) -> #root::__private::Result<::std::vec::Vec<#root::__private::ValueId>> {
                Self::#forward_multi_name(self, __cx, &[__input])
            }

            fn forward_multi(
                &self,
                __cx: &mut #root::__private::ModelContext,
                __inputs: &[#root::__private::ValueId],
            ) -> #root::__private::Result<::std::vec::Vec<#root::__private::ValueId>> {
                Self::#forward_multi_name(self, __cx, __inputs)
            }

            fn set_training(&mut self, __training: bool) {
                #(#child_set_training)*
            }

            fn visit_tensors(
                &self,
                __path: &#root::__private::ModulePath,
                __visitor: &mut dyn FnMut(
                    &#root::__private::ModulePath,
                    &#root::__private::DenseTensor<f32>,
                    #root::__private::ParameterRole,
                ) -> #root::__private::Result<()>,
            ) -> #root::__private::Result<()> {
                #(#child_visit_tensors)*
                #(#param_visit_tensors)*
                Ok(())
            }

            fn visit_tensors_mut(
                &mut self,
                __path: &#root::__private::ModulePath,
                __visitor: &mut dyn FnMut(
                    &#root::__private::ModulePath,
                    &mut #root::__private::DenseTensor<f32>,
                    #root::__private::ParameterRole,
                ) -> #root::__private::Result<()>,
            ) -> #root::__private::Result<()> {
                #(#child_visit_tensors_mut)*
                #(#param_visit_tensors_mut)*
                Ok(())
            }

            fn describe(
                &self,
                __path: &#root::__private::ModulePath,
                __out: &mut #root::__private::ModuleDescriptor,
            ) {
                let mut __node = #root::__private::ModuleDescriptor::new(#module_name, __path.clone());
                #(#child_describe)*
                __out.children.push(__node);
            }
        }
    };
    Ok(expanded)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EnumFieldKind {
    Module,
    Param,
    Skipped,
}

struct EnumField {
    binding: Ident,
    segment: String,
    role: Ident,
    kind: EnumFieldKind,
    body: bool,
    ty: syn::Type,
}

struct EnumVariant {
    pattern: TokenStream2,
    body: Ident,
    body_segment: String,
    fields: Vec<EnumField>,
}

/// `#[derive(Module)]` for enums.
///
/// Each variant is an *alternative architecture*: its fields are registered at
/// the same paths regardless of the active variant (the variant name is not a
/// path segment), so a checkpoint or optimizer state survives a variant switch.
/// The derived `forward` delegates to exactly one module field per variant,
/// either the only module field or the one marked `#[module(body)]`.
fn expand_enum(
    input: &DeriveInput,
    options: Options,
    data: &syn::DataEnum,
) -> Result<TokenStream2> {
    let root = &options.root;
    let name = &input.ident;
    let module_name = options
        .module_name
        .clone()
        .unwrap_or_else(|| name.to_string());
    if options.forward_multi != "forward_multi" {
        return Err(Error::new_spanned(
            name,
            "#[derive(Module)] on an enum always dispatches `forward`/`forward_multi` to the \
             variant body field; remove the `forward_multi` option and call the body module \
             directly, or implement Module manually",
        ));
    }
    if data.variants.is_empty() {
        return Err(Error::new_spanned(
            name,
            "#[derive(Module)] cannot be applied to an enum with no variants",
        ));
    }

    let mut variants = Vec::new();
    for (variant_index, variant) in data.variants.iter().enumerate() {
        let variant_name = &variant.ident;
        let mut fields = Vec::new();
        let mut pattern_fields = Vec::new();
        let mut tuples = Vec::new();
        match &variant.fields {
            Fields::Named(named) => {
                for (field_index, field) in named.named.iter().enumerate() {
                    let ident = field.ident.clone().expect("named field");
                    let field_options = FieldOptions::parse(&field.attrs)?;
                    let binding = quote::format_ident!(
                        "__field_{}_{}",
                        variant_index,
                        field_index,
                        span = proc_macro2::Span::call_site()
                    );
                    pattern_fields.push(quote! { #ident: #binding });
                    fields.push(EnumField {
                        binding,
                        segment: field_options
                            .rename
                            .clone()
                            .unwrap_or_else(|| ident.to_string()),
                        role: field_options.role.ident(),
                        kind: enum_field_kind(&field_options)?,
                        body: field_options.body,
                        ty: field.ty.clone(),
                    });
                }
            }
            Fields::Unnamed(unnamed) => {
                for (field_index, field) in unnamed.unnamed.iter().enumerate() {
                    let field_options = FieldOptions::parse(&field.attrs)?;
                    let binding = quote::format_ident!(
                        "__field_{}_{}",
                        variant_index,
                        field_index,
                        span = proc_macro2::Span::call_site()
                    );
                    tuples.push(binding.clone());
                    fields.push(EnumField {
                        binding,
                        segment: field_options
                            .rename
                            .clone()
                            .unwrap_or_else(|| field_index.to_string()),
                        role: field_options.role.ident(),
                        kind: enum_field_kind(&field_options)?,
                        body: field_options.body,
                        ty: field.ty.clone(),
                    });
                }
            }
            Fields::Unit => {
                return Err(Error::new_spanned(
                    variant_name,
                    "#[derive(Module)] does not support unit enum variants; use an `Option<M>` \
                     field or a variant with a module body",
                ));
            }
        }

        let body = select_body(variant_name, &fields)?;
        let body_segment = fields
            .iter()
            .find(|field| field.binding == body)
            .map(|field| field.segment.clone())
            .ok_or_else(|| {
                Error::new_spanned(
                    variant_name,
                    "internal #[derive(Module)] error: body field has no path segment",
                )
            })?;
        let pattern = if pattern_fields.is_empty() {
            quote! { Self::#variant_name(#(#tuples),*) }
        } else {
            quote! { Self::#variant_name { #(#pattern_fields),* } }
        };
        variants.push(EnumVariant {
            pattern,
            body,
            body_segment,
            fields,
        });
    }

    // Generic enums receive `FieldType: Module` bounds for every module field.
    let mut generics = input.generics.clone();
    if !generics.params.is_empty() {
        let where_clause = generics.make_where_clause();
        for variant in &variants {
            for field in &variant.fields {
                if field.kind == EnumFieldKind::Module {
                    let ty = &field.ty;
                    where_clause
                        .predicates
                        .push(parse_quote!(#ty: #root::__private::Module));
                }
            }
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let visit_parameters = variants.iter().map(|variant| {
        let pattern = &variant.pattern;
        let statements = variant
            .fields
            .iter()
            .filter(|field| field.kind != EnumFieldKind::Skipped)
            .map(|field| {
                let binding = &field.binding;
                let segment = &field.segment;
                match field.kind {
                    EnumFieldKind::Param => {
                        let role = &field.role;
                        quote! {
                            __registry.insert_tensor(
                                &__path.child(#segment),
                                #binding,
                                #root::__private::ParameterRole::#role,
                            )?;
                        }
                    }
                    _ => quote! {
                        #root::__private::Module::visit_parameters(
                            #binding,
                            &__path.child(#segment),
                            __registry,
                        )?;
                    },
                }
            });
        quote! { #pattern => { #(#statements)* ::std::result::Result::Ok(()) } }
    });

    let forward = variants.iter().map(|variant| {
        let pattern = &variant.pattern;
        let body = &variant.body;
        let segment = &variant.body_segment;
        quote! {
            #pattern => {
                let __body = #root::__private::module_forward(
                    __cx, #segment, #body, __input,
                )?;
                ::std::result::Result::Ok(::std::vec![__body])
            }
        }
    });
    let forward_multi = variants.iter().map(|variant| {
        let pattern = &variant.pattern;
        let body = &variant.body;
        let segment = &variant.body_segment;
        quote! {
            #pattern => #root::__private::module_forward_multi(
                __cx, #segment, #body, __inputs,
            )
        }
    });
    let set_training = variants.iter().map(|variant| {
        let pattern = &variant.pattern;
        let statements = variant
            .fields
            .iter()
            .filter(|field| field.kind == EnumFieldKind::Module)
            .map(|field| {
                let binding = &field.binding;
                quote! { #root::__private::Module::set_training(#binding, __training); }
            });
        quote! { #pattern => { #(#statements)* } }
    });
    let is_training = variants.iter().map(|variant| {
        let pattern = &variant.pattern;
        let body = &variant.body;
        quote! { #pattern => #root::__private::Module::is_training(#body) }
    });
    let visit_tensors = variants.iter().map(|variant| {
        let pattern = &variant.pattern;
        let statements = variant
            .fields
            .iter()
            .filter(|field| field.kind != EnumFieldKind::Skipped)
            .map(|field| {
                let binding = &field.binding;
                let segment = &field.segment;
                match field.kind {
                    EnumFieldKind::Param => {
                        let role = &field.role;
                        quote! {
                            __visitor(
                                &__path.child(#segment),
                                #binding,
                                #root::__private::ParameterRole::#role,
                            )?;
                        }
                    }
                    _ => quote! {
                        #root::__private::Module::visit_tensors(
                            #binding,
                            &__path.child(#segment),
                            __visitor,
                        )?;
                    },
                }
            });
        quote! { #pattern => { #(#statements)* ::std::result::Result::Ok(()) } }
    });
    let visit_tensors_mut = variants.iter().map(|variant| {
        let pattern = &variant.pattern;
        let statements = variant
            .fields
            .iter()
            .filter(|field| field.kind != EnumFieldKind::Skipped)
            .map(|field| {
                let binding = &field.binding;
                let segment = &field.segment;
                match field.kind {
                    EnumFieldKind::Param => {
                        let role = &field.role;
                        quote! {
                            __visitor(
                                &__path.child(#segment),
                                #binding,
                                #root::__private::ParameterRole::#role,
                            )?;
                        }
                    }
                    _ => quote! {
                        #root::__private::Module::visit_tensors_mut(
                            #binding,
                            &__path.child(#segment),
                            __visitor,
                        )?;
                    },
                }
            });
        quote! { #pattern => { #(#statements)* ::std::result::Result::Ok(()) } }
    });
    let describe = variants.iter().map(|variant| {
        let pattern = &variant.pattern;
        let statements = variant
            .fields
            .iter()
            .filter(|field| field.kind == EnumFieldKind::Module)
            .map(|field| {
                let binding = &field.binding;
                let segment = &field.segment;
                quote! {
                    #root::__private::Module::describe(
                        #binding,
                        &__path.child(#segment),
                        &mut __node,
                    );
                }
            });
        quote! { #pattern => { #(#statements)* } }
    });

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics #root::__private::Module for #name #ty_generics #where_clause {
            fn module_type(&self) -> &'static str {
                #module_name
            }

            fn visit_parameters(
                &self,
                __path: &#root::__private::ModulePath,
                __registry: &mut #root::__private::ParameterRegistry,
            ) -> #root::__private::Result<()> {
                match self {
                    #(#visit_parameters),*
                }
            }

            fn forward(
                &self,
                __cx: &mut #root::__private::ModelContext,
                __input: #root::__private::ValueId,
            ) -> #root::__private::Result<::std::vec::Vec<#root::__private::ValueId>> {
                match self {
                    #(#forward),*
                }
            }

            fn forward_multi(
                &self,
                __cx: &mut #root::__private::ModelContext,
                __inputs: &[#root::__private::ValueId],
            ) -> #root::__private::Result<::std::vec::Vec<#root::__private::ValueId>> {
                match self {
                    #(#forward_multi),*
                }
            }

            fn set_training(&mut self, __training: bool) {
                match self {
                    #(#set_training),*
                }
            }

            fn is_training(&self) -> bool {
                match self {
                    #(#is_training),*
                }
            }

            fn visit_tensors(
                &self,
                __path: &#root::__private::ModulePath,
                __visitor: &mut dyn FnMut(
                    &#root::__private::ModulePath,
                    &#root::__private::DenseTensor<f32>,
                    #root::__private::ParameterRole,
                ) -> #root::__private::Result<()>,
            ) -> #root::__private::Result<()> {
                match self {
                    #(#visit_tensors),*
                }
            }

            fn visit_tensors_mut(
                &mut self,
                __path: &#root::__private::ModulePath,
                __visitor: &mut dyn FnMut(
                    &#root::__private::ModulePath,
                    &mut #root::__private::DenseTensor<f32>,
                    #root::__private::ParameterRole,
                ) -> #root::__private::Result<()>,
            ) -> #root::__private::Result<()> {
                match self {
                    #(#visit_tensors_mut),*
                }
            }

            fn describe(
                &self,
                __path: &#root::__private::ModulePath,
                __out: &mut #root::__private::ModuleDescriptor,
            ) {
                let mut __node =
                    #root::__private::ModuleDescriptor::new(#module_name, __path.clone());
                match self {
                    #(#describe),*
                }
                __out.children.push(__node);
            }
        }
    })
}

fn enum_field_kind(options: &FieldOptions) -> Result<EnumFieldKind> {
    if options.skip {
        return Ok(EnumFieldKind::Skipped);
    }
    if options.param {
        return Ok(EnumFieldKind::Param);
    }
    Ok(EnumFieldKind::Module)
}

fn select_body(variant: &Ident, fields: &[EnumField]) -> Result<Ident> {
    if let Some(invalid) = fields
        .iter()
        .find(|field| field.body && field.kind != EnumFieldKind::Module)
    {
        return Err(Error::new_spanned(
            &invalid.binding,
            "#[module(body)] must mark a module field, not a parameter or skipped field",
        ));
    }
    let marked: Vec<&EnumField> = fields
        .iter()
        .filter(|field| field.kind == EnumFieldKind::Module && field.body)
        .collect();
    if marked.len() > 1 {
        return Err(Error::new_spanned(
            variant,
            "a module variant may mark at most one field with #[module(body)]",
        ));
    }
    if let Some(field) = marked.first() {
        return Ok(field.binding.clone());
    }
    let modules: Vec<&EnumField> = fields
        .iter()
        .filter(|field| field.kind == EnumFieldKind::Module)
        .collect();
    match modules.as_slice() {
        [only] => Ok(only.binding.clone()),
        [] => Err(Error::new_spanned(
            variant,
            "module variant has no module field to forward to; add a module field or \
             implement the enum's Module forward manually",
        )),
        many => Err(Error::new_spanned(
            variant,
            format!(
                "module variant has {} module fields; mark the forward target with \
                 `#[module(body)]` or implement the enum's Module forward manually",
                many.len()
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    fn expand(source: DeriveInput) -> Result<TokenStream2> {
        super::expand(source)
    }

    #[test]
    fn derives_for_named_fields() {
        let tokens = expand(parse_quote! {
            struct Mlp {
                fc1: Linear,
                #[module(skip)]
                seed: u64,
                #[param]
                scale: DenseTensor<f32>,
            }
        })
        .unwrap()
        .to_string();
        assert!(tokens.contains("visit_parameters"));
        assert!(tokens.contains("visit_tensors_mut"));
        assert!(tokens.contains("scale"));
        assert!(tokens.contains("fc1"));
    }

    #[test]
    fn derives_for_tuple_structs() {
        let tokens = expand(parse_quote! { struct Wrapper(Linear, #[module(skip)] u64); })
            .unwrap()
            .to_string();
        assert!(tokens.contains("visit_parameters"));
        assert!(tokens.contains("self . 0"), "{tokens}");
        assert!(!tokens.contains("self . 1"), "{tokens}");
    }

    #[test]
    fn rejects_unit_structs() {
        let error = expand(parse_quote! { struct Unit; }).unwrap_err();
        assert!(error.to_string().contains("requires a struct with fields"));
    }

    #[test]
    fn rejects_empty_modules() {
        let error = expand(parse_quote! { struct Empty {} }).unwrap_err();
        assert!(error.to_string().contains("no fields"));
    }

    #[test]
    fn supports_generics() {
        let tokens = expand(parse_quote! {
            struct Wrapper<M: Module> {
                inner: M,
                #[module(skip)]
                marker: u8,
            }
        })
        .unwrap()
        .to_string();
        assert!(tokens.contains("Module"));
    }

    #[test]
    fn derives_enum_variants() {
        let tokens = expand(parse_quote! {
            enum Kind {
                Sliding {
                    rope: Rope,
                    #[module(skip)]
                    scale: f32,
                },
                Full(TinyAttention),
            }
        })
        .unwrap()
        .to_string();
        assert!(tokens.contains("match self"));
        assert!(tokens.contains("rope"));
        assert!(tokens.contains("forward_multi"));
    }

    #[test]
    fn enum_body_marker_selects_forward_field() {
        let tokens = expand(parse_quote! {
            enum Block {
                Dense {
                    #[module(body)]
                    mlp: Mlp,
                    #[module(skip)]
                    config: u64,
                },
                Sparse {
                    #[module(body)]
                    moe: Moe,
                    router: Router,
                },
            }
        })
        .unwrap()
        .to_string();
        assert!(tokens.contains("mlp"));
        assert!(tokens.contains("moe"));
        assert!(tokens.contains("router"));
    }

    #[test]
    fn rejects_enum_without_unique_body() {
        let error = expand(parse_quote! {
            enum Ambiguous {
                Two { a: A, b: B },
            }
        })
        .unwrap_err();
        assert!(error.to_string().contains("module(body)"), "{error}");
    }

    #[test]
    fn rejects_enum_unit_variant() {
        let error = expand(parse_quote! {
            enum Optional { Absent, Present(Inner) }
        })
        .unwrap_err();
        assert!(error.to_string().contains("unit enum variant"), "{error}");
    }

    #[test]
    fn rejects_empty_enum() {
        let error = expand(parse_quote! { enum Never {} }).unwrap_err();
        assert!(error.to_string().contains("no variants"), "{error}");
    }

    #[test]
    fn enum_with_parameters_and_generics() {
        let tokens = expand(parse_quote! {
            enum Generic<M: Module> {
                Wrapped(M),
                Scaled {
                    inner: M,
                    #[param(role = "frozen")]
                    scale: DenseTensor<f32>,
                },
            }
        })
        .unwrap()
        .to_string();
        assert!(tokens.contains("Frozen"));
        assert!(tokens.contains("Wrapped"));
        assert!(tokens.contains("Scaled"));
    }

    #[test]
    fn rejects_skip_and_param_together() {
        let error = expand(parse_quote! {
            struct Bad {
                #[module(skip)]
                #[param]
                value: DenseTensor<f32>,
            }
        })
        .unwrap_err();
        assert!(error.to_string().contains("cannot be combined"));
    }
}
