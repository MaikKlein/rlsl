extern crate syn;
#[macro_use]
extern crate quote;
extern crate proc_macro2;

use proc_macro::TokenStream;
use proc_macro2::TokenTree;
use syn::parse::{Parse, ParseBuffer, ParseStream, Result};
use syn::spanned::Spanned;
use syn::{
    braced, parenthesized, parse_macro_input, punctuated::Punctuated, Expr, FnArg, Ident, Token,
    Type, Visibility,
};

#[derive(Debug)]
struct Functions {
    fns: Vec<Function>,
}
impl Parse for Functions {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut fns = Vec::new();
        while let Ok(f) = input.call(Function::parse) {
            fns.push(f);
        }
        Ok(Functions { fns })
    }
}

#[derive(Debug)]
struct Function {
    ident: Ident,
    args: Punctuated<FnArg, Token![,]>,
}

impl Parse for Function {
    fn parse(input: ParseStream) -> Result<Self> {
        let paren;
        let body;
        let vis: Visibility = input.parse()?;
        input.parse::<Token![fn]>()?;
        let ident: Ident = input.parse()?;
        parenthesized!(paren in input);
        let args = paren.parse_terminated(FnArg::parse)?;
        input.parse::<Token![->]>()?;
        let ty: Type = input.parse()?;
        braced!(body in input);
        let tts: proc_macro2::TokenStream = body.parse()?;
        Ok(Function { ident, args })
    }
}
#[proc_macro]
pub fn rlsl_test(input: TokenStream) -> TokenStream {
    println!("{:#?}", input);
    let f = parse_macro_input!(input as Functions);
    println!("{:?}", f);
    let q = quote!{};
    TokenStream::from(q)
}
