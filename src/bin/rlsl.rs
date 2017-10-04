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
extern crate rustc_trans;
extern crate syntax;
extern crate syntax_pos;
use rustc_driver::{get_args, run, run_compiler, Compilation, CompilerCalls};
use rustc_driver::driver::{CompileController, CompileState};
use rustc::session::Session;

struct RlslCompilerCalls;

use rustc::session::config::{self, ErrorOutputType, Input};
use std::path::PathBuf;
use syntax::ast;
use rustc_errors as errors;
impl<'a> CompilerCalls<'a> for RlslCompilerCalls {
    fn early_callback(
        &mut self,
        matches: &getopts::Matches,
        _: &config::Options,
        _: &ast::CrateConfig,
        _: &errors::registry::Registry,
        _: ErrorOutputType,
    ) -> Compilation {
        println!(" matches = {:?}", matches.free);
        Compilation::Continue
    }
    fn late_callback(
        &mut self,
        matches: &getopts::Matches,
        sess: &Session,
        _: &rustc::middle::cstore::CrateStore,
        input: &Input,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
    ) -> Compilation {
        Compilation::Continue
    }
    fn no_input(
        &mut self,
        matches: &getopts::Matches,
        _: &config::Options,
        _: &ast::CrateConfig,
        _: &Option<PathBuf>,
        _: &Option<PathBuf>,
        _: &errors::registry::Registry,
    ) -> Option<(Input, Option<PathBuf>)> {
        println!("no input");
        println!("matches = {:?}", matches.free);
        None
    }
    fn build_controller<'tcx>(
        &'tcx mut self,
        _: &rustc::session::Session,
        _: &getopts::Matches,
    ) -> CompileController<'a> {
        let mut controller = CompileController::basic();
        controller.after_analysis.stop = Compilation::Stop;
        controller.after_analysis.callback = box |s: &mut CompileState| {
            let tcx = &s.tcx.unwrap();
            let time_passes = tcx.sess.time_passes();
            let f = rustc_driver::driver::build_output_filenames(
                s.input,
                &s.out_dir.map(|p| p.into()),
                &s.out_file.map(|p| p.into()),
                &[],
                tcx.sess,
            );
            let crate_types = &tcx.sess.crate_types;
            println!("crate_types = {:?}", crate_types);

            rustc_mir::transform::dump_mir::emit_mir(*tcx, &f);
            let (items, _) = rustc_trans::collect_crate_translation_items(
                *tcx,
                rustc_trans::TransItemCollectionMode::Eager,
            );
            let items = rlsl::trans::spirv::trans_all_items(*tcx, &items);
            rlsl::trans::spirv::trans_spirv(*tcx, &items);
        };
        controller
    }
}
fn main() {
    let mut calls = RlslCompilerCalls;
    let result = run(move || {
        let mut args = get_args();
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
        let (a, b) = run_compiler(&args, &mut calls, None, None);
        (a, b)
    });
}
