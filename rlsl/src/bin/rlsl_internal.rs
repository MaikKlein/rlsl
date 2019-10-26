// use std::process::Command;
// // The whole point of this wrapper is to call `rlsl_internal` in the correct enviroment. We use
// // rustup to force the correct toolchain.
// fn main() {
//     let args: Vec<String> = std::env::args_os()
//         .map(|arg| arg.to_str().unwrap().into())
//         .collect();
//     let is_build_script = args
//         .iter()
//         .find(|arg| arg.as_str() == "build_script_build")
//         .is_some();
//     // We do not call into rlsl when we want to compile a build script. This should run on the CPU,
//     // we still want to make sure that we use the correct version of rustc.
//     if is_build_script {
//         Command::new("rustup")
//             .arg("run")
//             .arg("rlsl")
//             .arg("rustc")
//             .args(&args[1..])
//             .status()
//             .expect("Unable to start rustc");
//         return;
//     }
//     Command::new("rustup")
//         .arg("run")
//         .arg("rlsl")
//         .arg("rlsl_internal")
//         .args(&args[1..])
//         .status()
//         .expect("Unable to start rlsl");
// }
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
extern crate rustc_interface;
extern crate rustc_mir;
extern crate rustc_passes;
extern crate rustc_plugin;
extern crate rustc_resolve;
extern crate rustc_save_analysis as save;
extern crate syntax;
extern crate syntax_pos;
use rustc::session::config::PrintRequest;
use rustc::session::{early_error, Session};
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use std::process::Command;
use syntax::feature_gate::{GatedCfg, UnstableFeatures};
// use rustc_driver::{CompileController, CompileState};
use rustc_driver::{run_compiler, Callbacks};
use rustc_interface::interface;
use rustc_mir::monomorphize::collector::{collect_crate_mono_items, MonoItemCollectionMode};
use syntax_pos::DUMMY_SP;
pub struct RlslCompilerCalls {
    pub is_bin: bool,
}

use rustc::session::config::{self, ErrorOutputType, Input};
use rustc_errors as errors;
use std::path::PathBuf;
use syntax::ast;
impl Callbacks for RlslCompilerCalls {
    fn config(&mut self, config: &mut interface::Config) {}
    /// Called after analysis and returns true to continue execution
    fn after_analysis(&mut self, compiler: &interface::Compiler) -> bool {
        use std::io::Write;
        let mut f = std::fs::File::create("/home/maik/projects/rlsl/output").unwrap();
        writeln!(&mut f, "test");
        if !self.is_bin {
            return true;
        }
        compiler.global_ctxt().unwrap().peek_mut().enter(|tcx| {
            let (items, _) = collect_crate_mono_items(tcx, MonoItemCollectionMode::Eager);
            let items = rlsl::collector::trans_all_items(tcx, &items);
            rlsl::trans_spirv(tcx, &items);
            writeln!(&mut f, "{:#?}", items);
        });

        // let mut controller = CompileController::basic();
        // session.abort_if_errors();
        // controller.keep_ast = session.opts.debugging_opts.keep_ast;
        // controller.continue_parse_after_error =
        //     session.opts.debugging_opts.continue_parse_after_error;
        //if let Some(ref crate_type) = matches.opt_str("crate-type") {
        //    if crate_type == "bin" {
        //        controller.after_analysis.stop = Compilation::Stop;
        //        controller.keep_ast = true;
        //        controller.make_glob_map = rustc_resolve::MakeGlobMap::Yes;
        //        controller.after_analysis.run_callback_on_error = false;
        //        controller.after_analysis.callback = box |state: &mut CompileState| {
        //            let tcx = &state.tcx.unwrap();
        //            let f = rustc_driver::driver::build_output_filenames(
        //                state.input,
        //                &state.out_dir.map(|p| p.into()),
        //                &state.out_file.map(|p| p.into()),
        //                &[],
        //                tcx.sess,
        //            );
        //            //eprintln!("err files: {:?}", f);
        //            let _ = rustc_mir::transform::dump_mir::emit_mir(*tcx, &f);
        //            let (items, _) = collect_crate_mono_items(*tcx, MonoItemCollectionMode::Eager);
        //            // TODO: Custom collector not needed anymore?
        //            let items = rlsl::collector::trans_all_items(*tcx, &items);
        //            rlsl::trans_spirv(*tcx, &items);
        //        };
        //    }
        //}
        false
    }
}
//impl<'a> CompilerCalls<'a> for RlslCompilerCalls {
//    fn early_callback(
//        &mut self,
//        _matches: &getopts::Matches,
//        _: &config::Options,
//        _: &ast::CrateConfig,
//        _: &errors::registry::Registry,
//        _: ErrorOutputType,
//    ) -> Compilation {
//        Compilation::Continue
//    }
//    fn late_callback(
//        &mut self,
//        trans: &CodegenBackend,
//        matches: &getopts::Matches,
//        sess: &Session,
//        cstore: &rustc::middle::cstore::CrateStore,
//        input: &Input,
//        odir: &Option<PathBuf>,
//        ofile: &Option<PathBuf>,
//    ) -> Compilation {
//        RustcDefaultCalls::print_crate_info(trans, sess, Some(input), odir, ofile)
//            .and_then(|| RustcDefaultCalls::list_metadata(sess, cstore, matches, input))
//    }
//    fn no_input(
//        &mut self,
//        _matches: &getopts::Matches,
//        _: &config::Options,
//        _: &ast::CrateConfig,
//        _: &Option<PathBuf>,
//        _: &Option<PathBuf>,
//        _: &errors::registry::Registry,
//    ) -> Option<(Input, Option<PathBuf>)> {
//        None
//    }
//    fn build_controller<'tcx>(
//        self: Box<Self>,
//        session: &rustc::session::Session,
//        matches: &getopts::Matches,
//    ) -> CompileController<'a> {

//        let mut controller = CompileController::basic();
//        session.abort_if_errors();
//        controller.keep_ast = session.opts.debugging_opts.keep_ast;
//        controller.continue_parse_after_error =
//            session.opts.debugging_opts.continue_parse_after_error;
//        if let Some(ref crate_type) = matches.opt_str("crate-type") {
//            if crate_type == "bin" {
//                controller.after_analysis.stop = Compilation::Stop;
//                controller.keep_ast = true;
//                controller.make_glob_map = rustc_resolve::MakeGlobMap::Yes;
//                controller.after_analysis.run_callback_on_error = false;
//                controller.after_analysis.callback = box |state: &mut CompileState| {
//                    let tcx = &state.tcx.unwrap();
//                    let f = rustc_driver::driver::build_output_filenames(
//                        state.input,
//                        &state.out_dir.map(|p| p.into()),
//                        &state.out_file.map(|p| p.into()),
//                        &[],
//                        tcx.sess,
//                    );
//                    //eprintln!("err files: {:?}", f);
//                    let _ = rustc_mir::transform::dump_mir::emit_mir(*tcx, &f);
//                    let (items, _) = collect_crate_mono_items(*tcx, MonoItemCollectionMode::Eager);
//                    // TODO: Custom collector not needed anymore?
//                    let items = rlsl::collector::trans_all_items(*tcx, &items);
//                    rlsl::trans_spirv(*tcx, &items);
//                };
//            }
//        }
//        controller
//    }
//}
fn main() {
    use std::io::Write;
    let mut args: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_str().unwrap().into())
        .collect();
    let is_build_script = args
        .iter()
        .filter(|arg| arg.as_str() == "build_script_build")
        .nth(0)
        .is_some();
    if is_build_script {
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
    // args.extend_from_slice(&["--extern".into(), core_path]);
    // args.extend_from_slice(&["--extern".into(), std_path]);
    // args.extend_from_slice(&["--extern".into(), builtins_path]);
    args.extend_from_slice(&["-L".into(), l]);
    args.extend_from_slice(&["--cfg".into(), "spirv".into()]);
    args.extend_from_slice(&["-Z".into(), "always-encode-mir".into()]);
    args.extend_from_slice(&["-Z".into(), "mir-opt-level=3".into()]);
    let is_bin = args.iter().find(|s| s.as_str() == "bin").is_some();
    let mut rlsl = RlslCompilerCalls { is_bin };
    run_compiler(&args, &mut rlsl, None, None);
}
