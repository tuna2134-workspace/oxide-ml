//! Attribute parsing shared by the derive and attribute macros.

use syn::parse_quote;
use syn::{Attribute, Error, Ident, LitStr, Path, Result};

/// Options read from `#[oxide(...)]` on the type and `#[model(...)]`.
#[derive(Clone)]
pub struct Options {
    /// Path to the facade crate, e.g. `::oxide_ml`.
    pub root: Path,
    /// Name of the generated single-input graph method.
    pub forward: Ident,
    /// Name of the generated multi-input graph method.
    pub forward_multi: Ident,
    /// Override for `module_type()`.
    pub module_name: Option<String>,
    /// Generate the `predict` convenience method.
    pub predict: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            root: parse_quote!(::oxide_ml),
            forward: parse_quote!(forward),
            forward_multi: parse_quote!(forward_multi),
            module_name: None,
            predict: true,
        }
    }
}

impl Options {
    pub fn parse_oxide_attrs(attrs: &[Attribute]) -> Result<Self> {
        let mut options = Options::default();
        for attr in attrs {
            if !attr.path().is_ident("oxide") {
                continue;
            }
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("crate") || meta.path.is_ident("root") {
                    options.root = meta.value()?.parse()?;
                    return Ok(());
                }
                if meta.path.is_ident("forward") {
                    options.forward = meta.value()?.parse()?;
                    return Ok(());
                }
                if meta.path.is_ident("forward_multi") {
                    options.forward_multi = meta.value()?.parse()?;
                    return Ok(());
                }
                if meta.path.is_ident("name") {
                    options.module_name = Some(meta.value()?.parse::<LitStr>()?.value());
                    return Ok(());
                }
                if meta.path.is_ident("predict") {
                    options.predict = parse_bool(&meta)?;
                    return Ok(());
                }
                Err(meta.error(
                    "unknown #[oxide] option; expected `crate`, `forward`, `forward_multi`, `name` or `predict`",
                ))
            })?;
        }
        Ok(options)
    }

    /// Parse the arguments of `#[model(...)]`.
    pub fn parse_attribute(tokens: proc_macro2::TokenStream) -> Result<Self> {
        let mut options = Options::default();
        if tokens.is_empty() {
            return Ok(options);
        }
        let parser = syn::meta::parser(|meta| {
            if meta.path.is_ident("crate") || meta.path.is_ident("root") {
                options.root = meta.value()?.parse()?;
                return Ok(());
            }
            if meta.path.is_ident("predict") {
                options.predict = parse_bool(&meta)?;
                return Ok(());
            }
            if meta.path.is_ident("name") {
                options.module_name = Some(meta.value()?.parse::<LitStr>()?.value());
                return Ok(());
            }
            Err(meta.error("unknown #[model] option; expected `crate`, `predict` or `name`"))
        });
        syn::parse::Parser::parse2(parser, tokens)?;
        Ok(options)
    }
}

fn parse_bool(meta: &syn::meta::ParseNestedMeta<'_>) -> Result<bool> {
    if meta.input.peek(syn::Token![=]) {
        return Ok(meta.value()?.parse::<syn::LitBool>()?.value);
    }
    // `predict` / `predict(false)` both supported.
    if meta.input.peek(syn::token::Paren) {
        let content;
        syn::parenthesized!(content in meta.input);
        return Ok(content.parse::<syn::LitBool>()?.value);
    }
    Ok(true)
}

/// Options for one struct field.
#[derive(Clone, Default)]
pub struct FieldOptions {
    pub skip: bool,
    pub param: bool,
    pub role: ParamRole,
    pub rename: Option<String>,
    /// Marks the field an enum variant forwards to (see `#[derive(Module)]` on
    /// enums). Ignored on struct fields.
    pub body: bool,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum ParamRole {
    #[default]
    Trainable,
    Frozen,
    Buffer,
}

impl ParamRole {
    pub fn ident(self) -> Ident {
        match self {
            ParamRole::Trainable => parse_quote!(Trainable),
            ParamRole::Frozen => parse_quote!(Frozen),
            ParamRole::Buffer => parse_quote!(Buffer),
        }
    }
}

impl FieldOptions {
    pub fn parse(attrs: &[Attribute]) -> Result<Self> {
        let mut options = FieldOptions::default();
        for attr in attrs {
            if attr.path().is_ident("module") {
                attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("skip") {
                        options.skip = true;
                        return Ok(());
                    }
                    if meta.path.is_ident("rename") {
                        options.rename = Some(meta.value()?.parse::<LitStr>()?.value());
                        return Ok(());
                    }
                    if meta.path.is_ident("param") {
                        options.param = true;
                        return Ok(());
                    }
                    if meta.path.is_ident("body") {
                        options.body = true;
                        return Ok(());
                    }
                    Err(meta.error(
                        "unknown #[module] option; expected `skip`, `rename`, `param` or `body`",
                    ))
                })?;
            }
            if attr.path().is_ident("param") {
                options.param = true;
                if matches!(attr.meta, syn::Meta::List(_)) {
                    attr.parse_nested_meta(|meta| {
                        if meta.path.is_ident("role") {
                            let value: LitStr = meta.value()?.parse()?;
                            options.role = match value.value().as_str() {
                                "trainable" => ParamRole::Trainable,
                                "frozen" => ParamRole::Frozen,
                                "buffer" => ParamRole::Buffer,
                                other => {
                                    return Err(meta.error(format!(
                                        "unknown parameter role {other:?}; expected \
                                         \"trainable\", \"frozen\" or \"buffer\""
                                    )));
                                }
                            };
                            return Ok(());
                        }
                        if meta.path.is_ident("trainable") {
                            let value = parse_bool(&meta)?;
                            options.role = if value {
                                ParamRole::Trainable
                            } else {
                                ParamRole::Frozen
                            };
                            return Ok(());
                        }
                        Err(meta.error("unknown #[param] option; expected `role`"))
                    })?;
                }
            }
        }
        if options.skip && options.param {
            return Err(Error::new(
                proc_macro2::Span::call_site(),
                "#[module(skip)] and #[param] cannot be combined",
            ));
        }
        Ok(options)
    }
}
