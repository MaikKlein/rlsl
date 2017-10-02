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


//impl<'a, 'v: 'a> rustc::hir::intravisit::Visitor<'v> for RlslVisitor<'a, 'v> {
//    fn nested_visit_map<'this>(&'this mut self) -> hir_visit::NestedVisitorMap<'this, 'v> {
//        hir_visit::NestedVisitorMap::All(self.map)
//    }
//
//    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl, b: BodyId, s: Span, id: NodeId) {
//        use rustc::ty::{Slice, FnSig};
//        use rustc::ty;
//        walk_fn(self, fk, fd, b, s, id);
//        let def_id = self.map.local_def_id(id);
//        let node = self.map.find(id);
//        //println!("node = {:?}", node);
//        println!("name = {:?}", self.map.name(id));
//        let mir_fn = self.ty_ctx.maybe_optimized_mir(def_id);
//        //println!("mir_fn = {:#?}", mir_fn);
//        //for v in mir::traversal::preorder(mir_fn){
//        //    println!("{:?}", v);
//        //}
//
//        self.mir = mir_fn;
//        //self.visit_mir(mir_fn.unwrap());
//        self.mir = None;
//        //println!("mir_fn = {:#?}", mir_fn);
//        //        let sigs = self.ty_ctx.body_tables(b).liberated_fn_sigs();
//        //        let hir_id = self.map.node_to_hir_id(id);
//        //        let fn_sig = *sigs.get(hir_id).expect("sig");
//        //        let ty = self.ty_ctx.mk_fn_ptr(ty::Binder(fn_sig));
//        //        let fn_ty_spirv = self.ty_cache.from_ty(&mut self.ctx.builder, ty);
//        //        let ret_ty_spirv = self.ty_cache.from_ty(
//        //            &mut self.ctx.builder,
//        //            fn_sig.output(),
//        //        );
//        //        let spirv_function = self.ctx
//        //            .builder
//        //            .begin_function(
//        //                ret_ty_spirv.word,
//        //                None,
//        //                spirv::FunctionControl::empty(),
//        //                fn_ty_spirv.word,
//        //            )
//        //            .expect("begin fn");
//        //        println!("BLOCK");
//        //        self.ctx.builder.begin_basic_block(None).expect("block");
//        //        self.ctx.builder.end_function().expect("end fn");
//    }
//
//    fn visit_expr(&mut self, ex: &'v Expr) {
//        walk_expr(self, ex);
//        //let spirv_expr = match ex.node {
//        //    Expr_::ExprLit(ref lit) => {
//        //        match lit.node {
//        //            syntax::ast::LitKind::FloatUnsuffixed(sym) => {
//        //                let ty = self.get_table().expr_ty(ex);
//        //                let spirv_ty = self.ty_cache.from_ty(&mut self.ctx.builder, ty);
//        //                let f = sym.as_str().parse::<f32>().expect("parse");
//        //                Some(SpirvExpr(self.ctx.builder.constant_f32(spirv_ty.word, f)))
//        //            }
//        //            ref rest => unimplemented!("{:?}", rest),
//        //        }
//        //    }
//        //    //Expr_::ExprPath(ref path) => {
//        //    //    if let &QPath::Resolved(_, ref p) = path {
//        //    //        if let hir::def::Def::Local(l) = p.def {
//        //    //            println!("EXPR {:?}", self.map.get_if_local(l));
//        //    //        }
//        //    //    }
//        //    //}
//        //    _ => None,
//        //};
//        //if let Some(spirv_expr) = spirv_expr {
//        //    self.ctx.exprs.insert(ex.id, spirv_expr);
//        //}
//    }
//    fn visit_body(&mut self, b: &'v Body) {
//        self.current_table.push(self.ty_ctx.body_tables(b.id()));
//        walk_body(self, b);
//        //for arg in b.arguments.iter() {
//        //    match arg.pat.node {
//        //        hir::PatKind::Binding(_, def_id, ..) => {
//        //            //let local = self.ty_ctx.
//        //            println!("local = {:?}", self.map.get_if_local(def_id));
//        //        }
//        //        ref rest => unimplemented!("{:?}", rest),
//        //    };
//        //}
//        //        if let Some(ret_expr) = self.ctx.exprs.get(&b.value.id) {
//        //            self.ctx.builder.ret_value(ret_expr.0).expect("ret value");
//        //        } else {
//        //            self.ctx.builder.ret().expect("ret");
//        //        }
//        let _ = self.current_table.pop();
//    }
//    fn visit_fn_decl(&mut self, fd: &'v FnDecl) {
//        //println!("FN");
//        walk_fn_decl(self, fd)
//    }
//    fn visit_ty(&mut self, t: &'v Ty) {
//        //println!("Ty {:?}", t);
//        walk_ty(self, t);
//    }
//    fn visit_local(&mut self, l: &'v Local) {
//
//        walk_local(self, l);
//        //        let ty = self.get_table().node_id_to_type(l.hir_id);
//        //        let spirv_ty = self.ty_cache.from_ty(&mut self.ctx.builder, ty);
//        //        let init_expr = l.init
//        //            .as_ref()
//        //            .and_then(|ex| self.ctx.exprs.get(&ex.id))
//        //            .map(|ex| ex.0);
//        //        let spirv_var = self.ctx.builder.variable(
//        //            spirv_ty.word,
//        //            None,
//        //            spirv::StorageClass::Function,
//        //            init_expr,
//        //        );
//        //        self.ctx.vars.insert(l.id, SpirvVar(spirv_var));
//        //println!("local = {:?}", self.map.find(l.id));
//    }
//    fn visit_variant_data(
//        &mut self,
//        s: &'v VariantData,
//        _: Symbol,
//        _: &'v Generics,
//        _parent_id: NodeId,
//        _: Span,
//    ) {
//        walk_struct_def(self, s);
//        //println!("typ {:#?}", self.map.find(s.id()));
//    }
//    fn visit_struct_field(&mut self, s: &'v StructField) {
//        walk_struct_field(self, s)
//    }
//}

struct RlslCompilerCalls;

use rustc::session::config::{self, ErrorOutputType, Input};
use std::path::PathBuf;
use syntax::ast;
use rustc_errors as errors;
impl<'a> CompilerCalls<'a> for RlslCompilerCalls {
    fn early_callback(
        &mut self,
        _: &getopts::Matches,
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
        _: &rustc::middle::cstore::CrateStore,
        input: &Input,
        odir: &Option<PathBuf>,
        ofile: &Option<PathBuf>,
    ) -> Compilation {
        println!("late");
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
        let (a, b) = run_compiler(&get_args(), &mut calls, None, Some(box std::io::stdout()));
        (a, b)
    });
}

//fn main() {}
