#![feature(rustc_private)]
#![feature(box_syntax)]
#![feature(test)]
//#[macro_use]
//extern crate debugit;

extern crate arena;
extern crate env_logger;
extern crate getopts;
extern crate log;
extern crate rlsl;
//extern crate rspirv;
extern crate rustc;
extern crate rustc_borrowck;
extern crate rustc_codegen_utils;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_incremental;
extern crate rustc_mir;
extern crate rustc_passes;
extern crate rustc_plugin;
extern crate rustc_resolve;
extern crate rustc_save_analysis as save;
extern crate syntax;
extern crate syntax_pos;
use rustc::session::Session;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_driver::driver::{CompileController, CompileState};
use rustc_driver::RustcDefaultCalls;
use rustc_driver::{run, run_compiler, Compilation, CompilerCalls};
use rustc_mir::monomorphize::collector::{collect_crate_mono_items, MonoItemCollectionMode};
struct RlslCompilerCalls;
impl RlslCompilerCalls {
    fn print_crate_info(codegen_backend: &CodegenBackend,
                        sess: &Session,
                        input: Option<&Input>,
                        odir: &Option<PathBuf>,
                        ofile: &Option<PathBuf>)
                        -> Compilation {
        use rustc::session::config::PrintRequest::*;
        // PrintRequest::NativeStaticLibs is special - printed during linking
        // (empty iterator returns true)
        if sess.opts.prints.iter().all(|&p| p==PrintRequest::NativeStaticLibs) {
            return Compilation::Continue;
        }

        let attrs = match input {
            None => None,
            Some(input) => {
                let result = parse_crate_attrs(sess, input);
                match result {
                    Ok(attrs) => Some(attrs),
                    Err(mut parse_error) => {
                        parse_error.emit();
                        return Compilation::Stop;
                    }
                }
            }
        };
        for req in &sess.opts.prints {
            match *req {
                TargetList => {
                    let mut targets = rustc_target::spec::get_targets().collect::<Vec<String>>();
                    targets.sort();
                    println!("{}", targets.join("\n"));
                },
                Sysroot => println!("{}", sess.sysroot().display()),
                TargetSpec => println!("{}", sess.target.target.to_json().pretty()),
                FileNames | CrateName => {
                    let input = match input {
                        Some(input) => input,
                        None => early_error(ErrorOutputType::default(), "no input file provided"),
                    };
                    let attrs = attrs.as_ref().unwrap();
                    let t_outputs = driver::build_output_filenames(input, odir, ofile, attrs, sess);
                    let id = rustc_codegen_utils::link::find_crate_name(Some(sess), attrs, input);
                    if *req == PrintRequest::CrateName {
                        println!("{}", id);
                        continue;
                    }
                    let crate_types = driver::collect_crate_types(sess, attrs);
                    for &style in &crate_types {
                        let fname = rustc_codegen_utils::link::filename_for_input(
                            sess,
                            style,
                            &id,
                            &t_outputs
                        );
                        println!("{}",
                                 fname.file_name()
                                      .unwrap()
                                      .to_string_lossy());
                    }
                }
                Cfg => {
                    let allow_unstable_cfg = UnstableFeatures::from_environment()
                        .is_nightly_build();

                    let mut cfgs = Vec::new();
                    for &(name, ref value) in sess.parse_sess.config.iter() {
                        let gated_cfg = GatedCfg::gate(&ast::MetaItem {
                            ident: ast::Path::from_ident(name.to_ident()),
                            node: ast::MetaItemKind::Word,
                            span: DUMMY_SP,
                        });

                        // Note that crt-static is a specially recognized cfg
                        // directive that's printed out here as part of
                        // rust-lang/rust#37406, but in general the
                        // `target_feature` cfg is gated under
                        // rust-lang/rust#29717. For now this is just
                        // specifically allowing the crt-static cfg and that's
                        // it, this is intended to get into Cargo and then go
                        // through to build scripts.
                        let value = value.as_ref().map(|s| s.as_str());
                        let value = value.as_ref().map(|s| s.as_ref());
                        if name != "target_feature" || value != Some("crt-static") {
                            if !allow_unstable_cfg && gated_cfg.is_some() {
                                continue;
                            }
                        }

                        cfgs.push(if let Some(value) = value {
                            format!("{}=\"{}\"", name, value)
                        } else {
                            format!("{}", name)
                        });
                    }

                    cfgs.sort();
                    for cfg in cfgs {
                        println!("{}", cfg);
                    }
                }
                RelocationModels | CodeModels | TlsModels | TargetCPUs | TargetFeatures => {
                    codegen_backend.print(*req, sess);
                }
                // Any output here interferes with Cargo's parsing of other printed output
                PrintRequest::NativeStaticLibs => {}
            }
        }
        return Compilation::Stop;
    }
}

use rustc::session::config::{self, ErrorOutputType, Input};
use rustc_errors as errors;
use std::path::PathBuf;
use syntax::ast;
impl<'a> CompilerCalls<'a> for RlslCompilerCalls {
    fn early_callback(
        &mut self,
        _matches: &getopts::Matches,
        _: &config::Options,
        _: &ast::CrateConfig,
        _: &errors::registry::Registry,
        _: ErrorOutputType,
    ) -> Compilation {
        Compilation::Continue
    }
    fn late_callback(
        &mut self,
        trans: &CodegenBackend,
        matches: &getopts::Matches,
        sess: &Session,
        cstore: &rustc::middle::cstore::CrateStore,
        input: &Input,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
    ) -> Compilation {
        RustcDefaultCalls::print_crate_info(trans, sess, Some(input), odir, ofile)
            .and_then(|| RustcDefaultCalls::list_metadata(sess, cstore, matches, input))
    }
    fn no_input(
        &mut self,
        _matches: &getopts::Matches,
        _: &config::Options,
        _: &ast::CrateConfig,
        _: &Option<PathBuf>,
        _: &Option<PathBuf>,
        _: &errors::registry::Registry,
    ) -> Option<(Input, Option<PathBuf>)> {
        None
    }
    fn build_controller<'tcx>(
        &'tcx mut self,
        session: &rustc::session::Session,
        matches: &getopts::Matches,
    ) -> CompileController<'a> {
        let mut controller = CompileController::basic();
        session.abort_if_errors();
        controller.keep_ast = session.opts.debugging_opts.keep_ast;
        controller.continue_parse_after_error =
            session.opts.debugging_opts.continue_parse_after_error;
        if let Some(ref crate_type) = matches.opt_str("crate-type") {
            if crate_type == "bin" {
                controller.after_analysis.stop = Compilation::Stop;
                controller.keep_ast = true;
                controller.make_glob_map = rustc_resolve::MakeGlobMap::Yes;
                controller.after_analysis.run_callback_on_error = false;
                controller.after_analysis.callback = box |state: &mut CompileState| {
                    let tcx = &state.tcx.unwrap();
                    let f = rustc_driver::driver::build_output_filenames(
                        state.input,
                        &state.out_dir.map(|p| p.into()),
                        &state.out_file.map(|p| p.into()),
                        &[],
                        tcx.sess,
                    );
                    //eprintln!("err files: {:?}", f);
                    let _ = rustc_mir::transform::dump_mir::emit_mir(*tcx, &f);
                    let (items, _) = collect_crate_mono_items(*tcx, MonoItemCollectionMode::Eager);
                    // TODO: Custom collector not needed anymore?
                    let items = rlsl::collector::trans_all_items(*tcx, &items);
                    rlsl::trans_spirv(*tcx, &items);
                };
            }
        }
        controller
    }
}
fn main() {
    let mut args: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_str().unwrap().into())
        .collect();
    let is_build_script = args.iter()
        .filter(|arg| arg.as_str() == "build_script_main")
        .nth(0)
        .is_some();
    if is_build_script {
        use std::process::Command;
        Command::new("rustc")
            .args(&args[1..])
            .status()
            .expect("rustc");
        return;
    }
    let home_dir = std::env::home_dir().expect("home_dir");
    let lib_search_path = home_dir.join(".rlsl").join("lib");
    let l = format!("{}", lib_search_path.as_path().display());
    let core = lib_search_path.join("libcore.rlib");
    let std = lib_search_path.join("libstd.rlib");
    let builtins = lib_search_path.join("libcompiler_builtins.rlib");
    let core_path = format!("core={}", core.display());
    let std_path = format!("std={}", std.display());
    let builtins_path = format!("compiler_builtins={}", builtins.display());
    args.extend_from_slice(&["--extern".into(), core_path]);
    args.extend_from_slice(&["--extern".into(), std_path]);
    args.extend_from_slice(&["--extern".into(), builtins_path]);
    args.extend_from_slice(&["-L".into(), l]);
    //args.extend_from_slice(&["--cfg".into(), "spirv".into()]);
    args.extend_from_slice(&["-Z".into(), "always-encode-mir".into()]);
    args.extend_from_slice(&["-Z".into(), "mir-opt-level=3".into()]);
    let mut calls = RlslCompilerCalls;
    let _ = run(move || run_compiler(&args, &mut calls, None, None));
}
