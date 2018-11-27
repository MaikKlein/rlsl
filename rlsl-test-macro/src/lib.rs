extern crate syn;
#[macro_use]
extern crate quote;
extern crate proc_macro2;

use proc_macro::TokenStream;
use proc_macro2::Span;
use std::env;
use std::process::Command;
use syn::parse::{Parse, ParseStream, Result};
use syn::{parse_macro_input, Ident, ItemFn};

mod project {
    use crate::Functions;
    use std::fs::{create_dir, remove_dir_all, write, File};
    use std::io::{self, Write};
    use std::path::Path;
    // TODO: Don't hardcode rlsl-math
    const CARGO_TOML: &'static str = r#"
        [package]
        name = "shadertest"
        version = "0.1.0"
        [dependencies]
        rlsl-math = { path =  "/home/maik/projects/rlsl/rlsl-math" }
        "#;

    pub(crate) fn generate_shader_bin(functions: &Functions, path: &Path) -> io::Result<()> {
        for f in &functions.fns {
            let mut bin_path = path.join(f.ident.to_string()).with_extension("rs");
            let mut bin_rs = File::create(bin_path)?;
            let ident = &f.ident;
            write!(
                &mut bin_rs,
                "{}",
                quote!{
                    #![feature(custom_attribute)]
                    extern crate rlsl_math;
                    use rlsl_math::prelude::*;
                    #f
                    #[allow(unused_attributes, dead_code)]
                    #[spirv(compute)]
                    fn compute(compute: Compute, buffer: Buffer<N0, N0, RuntimeArray<f32>>) {
                        let index = compute.global_invocation_index.x;
                        let value = buffer.data.get(index);
                        let result = #ident(index, value);
                        buffer.data.store(index, result);
                    }
                    fn main() {}
                }
            );
        }
        Ok(())
    }
    pub(crate) fn generate_project(location: &Path, functions: &Functions) -> io::Result<()> {
        let path = location;
        // If the dir already exists, lets remove it
        if path.is_dir() {
            remove_dir_all(&path)?;
        }
        create_dir(&path)?;
        write(path.join("Cargo.toml"), CARGO_TOML)?;
        let src = path.join("src");
        let bin = src.join("bin");
        create_dir(&src)?;
        create_dir(&bin)?;
        generate_shader_bin(functions, &bin)?;
        Ok(())
    }
}
pub(crate) struct Functions {
    pub fns: Vec<ItemFn>,
}

impl Parse for Functions {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut fns = Vec::new();
        while let Ok(f) = input.call(ItemFn::parse) {
            fns.push(f);
        }
        Ok(Functions { fns })
    }
}
#[proc_macro]
pub fn rlsl_test(input: TokenStream) -> TokenStream {
    let functions = parse_macro_input!(input as Functions);
    let path = env::temp_dir().join("shader_test");
    project::generate_project(&path, &functions).expect("");
    let function_iter = functions.fns.iter();
    let idents = functions
        .fns
        .iter()
        .map(|f| Ident::new(&format!("compute_{}", f.ident), Span::call_site()));
    let idents1 = functions.fns.iter().map(|f| &f.ident);
    let file_names = functions.fns.iter().map(|f| {
        let mut path = path.join(".shaders").join(f.ident.to_string());
        path.set_extension("spv");
        path.display().to_string()
    });
    let test = quote!{
        #[cfg(test)]
        mod tests {
            use ::*;
            #(#function_iter)*
            use quickcheck::TestResult;
            quickcheck! {
                #(
                    fn #idents(input: Vec<f32>) -> TestResult {
                        compute("compute", input, #file_names, #idents1)
                    }
                )*
            }
        }
    };
    Command::new("cargo")
        .arg("build")
        .env("RUSTC", "rlsl")
        .env("CARGO_TARGET_DIR", ".spirv_target")
        .current_dir(&path)
        .spawn()
        .expect("");
    TokenStream::from(test)
}
