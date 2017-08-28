#![feature(rustc_private)]
extern crate rustc;
extern crate rustc_driver;
extern crate syntax;
extern crate getopts;
extern crate env_logger;
use std::path::Path;
use rustc_driver::{run, run_compiler, get_args, CompilerCalls, Compilation};
use rustc_driver::driver::{CompileState, CompileController};
struct RlslCompilerCalls;
impl<'a> CompilerCalls<'a> for RlslCompilerCalls {
    fn build_controller(
        &mut self,
        _: &rustc::session::Session,
        _: &getopts::Matches,
    ) -> CompileController<'a> {
        let mut controller = CompileController::basic();
        controller.after_analysis.stop = Compilation::Stop;
        controller
    }
}
fn main() {
    env_logger::init().unwrap();
    let mut calls = RlslCompilerCalls;
    let result = run(move || {
        let (a, b) = run_compiler(&get_args(), &mut calls, None, None);
        println!("R = {:?}", a);
        (a, b)
    });

}
