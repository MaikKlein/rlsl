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
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_incremental;
extern crate rustc_mir;
extern crate rustc_passes;
extern crate rustc_plugin;
extern crate rustc_resolve;
extern crate rustc_save_analysis as save;
extern crate rustc_trans;
extern crate syntax;
extern crate syntax_pos;
use rustc_driver::{get_args, run, run_compiler, Compilation, CompilerCalls};
use rustc_driver::driver::{CompileController, CompileState};
use rustc_driver::RustcDefaultCalls;
use rustc::session::Session;

struct RlslCompilerCalls;

use rustc::session::config::{self, ErrorOutputType, Input};
use std::path::PathBuf;
use syntax::ast;
use rustc_errors as errors;
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
        matches: &getopts::Matches,
        sess: &Session,
        cstore: &rustc::middle::cstore::CrateStore,
        input: &Input,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
    ) -> Compilation {
        RustcDefaultCalls::print_crate_info(sess, Some(input), odir, ofile).and_then(|| {
            RustcDefaultCalls::list_metadata(sess, cstore, matches, input)
        })
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
        if let Some(ref crate_type) = matches.opt_str("crate-type") {
            if crate_type == "bin" {
                controller.after_analysis.stop = Compilation::Stop;
                controller.keep_ast = true;
                controller.make_glob_map = rustc_resolve::MakeGlobMap::Yes;
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
                    let (items, _) = rustc_trans::collect_crate_translation_items(
                        *tcx,
                        rustc_trans::TransItemCollectionMode::Eager,
                    );
                    let items = rlsl::collector::trans_all_items(*tcx, &items);
                    rlsl::trans_spirv(*tcx, &items);
                };
            }
        }
        controller
    }
}
fn main() {
    let mut args = get_args();
    eprintln!("{:?}", args);
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
    let core_path = format!("core={}", core.display());
    let std_path = format!("std={}", std.display());
    args.extend_from_slice(&["--extern".into(), core_path]);
    args.extend_from_slice(&["--extern".into(), std_path]);
    args.extend_from_slice(&["-L".into(), l]);
    args.extend_from_slice(&["--cfg".into(), "spirv".into()]);
    args.extend_from_slice(&["-Z".into(), "always-encode-mir".into()]);
    let mut calls = RlslCompilerCalls;
    let _ = run(move || run_compiler(&args, &mut calls, None, None));
}
