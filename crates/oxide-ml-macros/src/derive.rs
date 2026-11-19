//! `#[derive(Module)]`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Data, DeriveInput, Error, Fields, Ident, Result, parse_macro_input, parse_quote};

use crate::attr::{FieldOptions, Options};

struct ModuleField {
    ident: Ident,
    ty: syn::Type,
    segment: String,
}

struct ParamField {
    ident: Ident,
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

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(named) => &named.named,
            Fields::Unit => {
                return Err(Error::new_spanned(
                    name,
                    "#[derive(Module)] requires a struct with named fields; \
                     implement Module manually for tuple or unit structs",
                ));
            }
            Fields::Unnamed(_) => {
                return Err(Error::new_spanned(
                    name,
                    "#[derive(Module)] requires named fields; \
                     implement Module manually for tuple structs",
                ));
            }
        },
        Data::Enum(_) | Data::Union(_) => {
            return Err(Error::new_spanned(
                name,
                "#[derive(Module)] can only be applied to structs",
            ));
        }
    };

    let mut module_fields = Vec::new();
    let mut param_fields = Vec::new();
    for field in fields {
        let ident = field
            .ident
            .clone()
            .ok_or_else(|| Error::new_spanned(field, "expected a named field"))?;
        let field_options = FieldOptions::parse(&field.attrs)?;
        if field_options.skip {
            continue;
        }
        let segment = field_options
            .rename
            .clone()
            .unwrap_or_else(|| ident.to_string());
        if field_options.param {
            let role = field_options.role.ident();
            param_fields.push(ParamField {
                ident,
                segment,
                role,
            });
        } else {
            module_fields.push(ModuleField {
                ident,
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
        let ident = &field.ident;
        let segment = &field.segment;
        quote! {
            #root::__private::Module::visit_parameters(
                &self.#ident,
                &__path.child(#segment),
                __registry,
            )?;
        }
    });
    let param_discovery = param_fields.iter().map(|field| {
        let ident = &field.ident;
        let segment = &field.segment;
        let role = &field.role;
        quote! {
            __registry.insert_tensor(
                &__path.child(#segment),
                &self.#ident,
                #root::__private::ParameterRole::#role,
            )?;
        }
    });

    let child_set_training = module_fields.iter().map(|field| {
        let ident = &field.ident;
        quote! {
            #root::__private::Module::set_training(&mut self.#ident, __training);
        }
    });

    let child_visit_tensors = module_fields.iter().map(|field| {
        let ident = &field.ident;
        let segment = &field.segment;
        quote! {
            #root::__private::Module::visit_tensors(
                &self.#ident,
                &__path.child(#segment),
                __visitor,
            )?;
        }
    });
    let param_visit_tensors = param_fields.iter().map(|field| {
        let ident = &field.ident;
        let segment = &field.segment;
        let role = &field.role;
        quote! {
            __visitor(
                &__path.child(#segment),
                &self.#ident,
                #root::__private::ParameterRole::#role,
            )?;
        }
    });

    let child_visit_tensors_mut = module_fields.iter().map(|field| {
        let ident = &field.ident;
        let segment = &field.segment;
        quote! {
            #root::__private::Module::visit_tensors_mut(
                &mut self.#ident,
                &__path.child(#segment),
                __visitor,
            )?;
        }
    });
    let param_visit_tensors_mut = param_fields.iter().map(|field| {
        let ident = &field.ident;
        let segment = &field.segment;
        let role = &field.role;
        quote! {
            __visitor(
                &__path.child(#segment),
                &mut self.#ident,
                #root::__private::ParameterRole::#role,
            )?;
        }
    });

    let child_describe = module_fields.iter().map(|field| {
        let ident = &field.ident;
        let segment = &field.segment;
        quote! {
            #root::__private::Module::describe(
                &self.#ident,
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
    fn rejects_tuple_structs() {
        let error = expand(parse_quote! { struct Wrapper(Linear); }).unwrap_err();
        assert!(error.to_string().contains("named fields"));
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
