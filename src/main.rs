#![feature(rustc_private)]
#![feature(box_syntax)]
//#[macro_use]
//extern crate debugit;
extern crate rustc;
extern crate rustc_driver;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate getopts;
extern crate env_logger;

use rustc::hir::intravisit as hir_visit;
use rustc::hir::intravisit::*;
use rustc::hir::*;
use syntax_pos::Span;
use syntax::ast::NodeId;
use rustc::hir;
pub struct RlslVisitor<'a> {
    pub map: &'a hir::map::Map<'a>,
}
impl<'v> rustc::hir::intravisit::Visitor<'v> for RlslVisitor<'v> {
    fn nested_visit_map<'this>(&'this mut self) -> hir_visit::NestedVisitorMap<'this, 'v> {
        hir_visit::NestedVisitorMap::All(self.map)
    }

    fn visit_item(&mut self, i: &'v hir::Item) {
        //self.record("Item", Id::Node(i.id), i);
        walk_item(self, i)
    }
    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl, b: BodyId, s: Span, id: NodeId) {
        match fk {
            FnKind::ItemFn(ref name, ..) => {
                println!("FN {:?}", name);
            }
            _ => (),
        };
        walk_fn(self, fk, fd, b, s, id)
    }
    fn visit_body(&mut self, b: &'v Body) {
        println!("body");
        walk_body(self, b);
    }
    fn visit_fn_decl(&mut self, fd: &'v FnDecl) {
        println!("FN");
        walk_fn_decl(self, fd)
    }
    fn visit_ty(&mut self, t: &'v Ty) {
        println!("Ty {:?}", t);
        walk_ty(self, t)
    }
    fn visit_local(&mut self, l: &'v Local) {
        println!("local {:?}", l);
        walk_local(self, l)
    }
}
use std::path::Path;
use rustc_driver::{run, run_compiler, get_args, CompilerCalls, Compilation};
use rustc_driver::driver::{CompileState, CompileController};

fn hir<'v, 'tcx: 'v>(s: &'tcx mut CompileState<'v, 'tcx>) {
    let krate = s.hir_crate.unwrap();
    let map = s.hir_map.unwrap();
    let mut visitor = RlslVisitor { map };
    hir_visit::walk_crate(&mut visitor, krate);
}

struct RlslCompilerCalls;
impl<'a> CompilerCalls<'a> for RlslCompilerCalls {
    fn build_controller<'tcx>(
        &'tcx mut self,
        _: &rustc::session::Session,
        _: &getopts::Matches,
    ) -> CompileController<'a> {
        let mut controller = CompileController::basic();
        controller.after_analysis.stop = Compilation::Stop;
        let hir: fn(&mut CompileState) = unsafe { std::mem::transmute(hir as *const ()) };
        controller.after_hir_lowering.callback = box hir;
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

//fn main() {}
