#![feature(rustc_private)]
#![feature(box_syntax)]
#![feature(test)]
//#[macro_use]
//extern crate debugit;
#[macro_use]
extern crate log;
extern crate rustc;
extern crate rustc_driver;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate spirv_headers as spirv;
extern crate rspirv;
extern crate getopts;
extern crate env_logger;
extern crate rustc_mir;
extern crate rustc_passes;
extern crate arena;
extern crate rustc_plugin;
extern crate rustc_borrowck;
extern crate rustc_errors;
extern crate test;

//use rustc_borrowck::borrowck;
use rustc_passes::*;
use rustc_passes::loops;
use rustc_passes::static_recursion;
use rustc_plugin as plugin;
use rustc::middle;
//use rustc_passes::mir_stats;
//use rustc_driver::derive_registrar;
use rustc::middle::stability;
use rustc::util::common::time;
use rustc::mir;
use rspirv::mr::{Operand, Builder};
use rspirv::binary::Disassemble;
use rspirv::binary::Assemble;
use std::mem;
use rustc::hir::intravisit as hir_visit;
use rustc::hir::intravisit::*;
use rustc::hir::*;
use syntax_pos::Span;
use syntax::ast::NodeId;
use rustc::hir;
use syntax_pos::symbol::Symbol;
use std::path::Path;
use rustc_driver::{run, run_compiler, get_args, CompilerCalls, Compilation};
use rustc_driver::driver::{CompileState, CompileController};
use rustc::mir::visit::Visitor;


use rustc::session::CompileIncomplete;
use arena::DroplessArena;
use rustc::ty::{Resolutions, GlobalArenas};
use rustc::session::{Session, CompileResult};
pub struct SpirvCtx {
    pub builder: Builder,
    pub vars: HashMap<NodeId, SpirvVar>,
    pub exprs: HashMap<NodeId, SpirvExpr>,
}
impl SpirvCtx {
    pub fn new() -> Self {
        SpirvCtx {
            builder: Builder::new(),
            vars: HashMap::new(),
            exprs: HashMap::new(),
        }
    }
}

pub struct RlslVisitor<'a, 'tcx: 'a> {
    pub map: &'a hir::map::Map<'tcx>,
    pub ty_ctx: rustc::ty::TyCtxt<'a, 'tcx, 'tcx>,
    current_table: Vec<&'a rustc::ty::TypeckTables<'tcx>>,
    pub ty_cache: SpirvTyCache<'tcx>,
    ctx: SpirvCtx,
}
impl<'a, 'tcx: 'a> RlslVisitor<'a, 'tcx> {
    pub fn get_table(&self) -> &'a rustc::ty::TypeckTables<'tcx> {
        self.current_table.last().expect("no table yet")
    }
    pub fn new(state: &CompileState<'a, 'tcx>) -> Self {
        let mut ctx = SpirvCtx::new();
        ctx.builder.capability(spirv::Capability::Shader);
        ctx.builder.ext_inst_import("GLSL.std.450");
        ctx.builder.memory_model(
            spirv::AddressingModel::Logical,
            spirv::MemoryModel::GLSL450,
        );
        let tcx = state.tcx.expect("tcx");
        let mut visitor = RlslVisitor {
            map: &tcx.hir,
            ty_ctx: tcx,
            current_table: Vec::new(),
            ctx,
            ty_cache: SpirvTyCache::new(),
        };
        //println!("{:#?}", visitor.map.krate());
        hir_visit::walk_crate(&mut visitor, visitor.map.krate());
        visitor
    }

    pub fn build_module(self) {
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create("shader.spv").unwrap();
        let spirv_module = self.ctx.builder.module();
        let bytes: Vec<u8> = spirv_module
            .assemble()
            .iter()
            .flat_map(|val| {
                (0..mem::size_of::<u32>()).map(move |i| ((val >> (8 * i)) & 0xff) as u8)
            })
            .collect();
        let mut loader = rspirv::mr::Loader::new();
        //let bytes = b.module().assemble_bytes();
        rspirv::binary::parse_bytes(&bytes, &mut loader);
        println!("{}", loader.module().disassemble());
        f.write_all(&bytes);
    }
}
#[derive(Copy, Clone, Debug)]
pub struct SpirvTy {
    pub word: spirv::Word,
}
impl From<spirv::Word> for SpirvTy {
    fn from(word: spirv::Word) -> SpirvTy {
        SpirvTy { word: word }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SpirvVar(pub spirv::Word);
#[derive(Copy, Clone, Debug)]
pub struct SpirvExpr(pub spirv::Word);

use std::collections::HashMap;
pub struct SpirvTyCache<'a> {
    pub ty_cache: HashMap<rustc::ty::Ty<'a>, SpirvTy>,
}
use rustc::ty;
impl<'a> SpirvTyCache<'a> {
    pub fn new() -> Self {
        SpirvTyCache { ty_cache: HashMap::new() }
    }
    pub fn from_ty<'tcx>(&'tcx mut self, builder: &mut Builder, ty: rustc::ty::Ty<'a>) -> SpirvTy {
        use rustc::ty::TypeVariants;
        if let Some(ty) = self.ty_cache.get(ty) {
            return *ty;
        }
        let spirv_type: SpirvTy = match ty.sty {
            TypeVariants::TyBool => builder.type_bool().into(),
            TypeVariants::TyFloat(f_ty) => {
                use syntax::ast::FloatTy;
                match f_ty {
                    FloatTy::F32 => builder.type_float(32).into(),
                    FloatTy::F64 => builder.type_float(64).into(),
                }
            }
            TypeVariants::TyTuple(slice, _) if slice.len() == 0 => builder.type_void().into(),
            TypeVariants::TyFnPtr(sig) => {
                let ret_ty = self.from_ty(builder, sig.output().skip_binder());
                let input_ty: Vec<_> = sig.inputs()
                    .skip_binder()
                    .iter()
                    .map(|ty| self.from_ty(builder, ty).word)
                    .collect();
                builder.type_function(ret_ty.word, &input_ty).into()
            }
            ref r => unimplemented!("{:?}", r),
        };
        self.ty_cache.insert(ty, spirv_type);
        spirv_type
    }
    pub fn ty_ast_ptr(&mut self, builder: &mut Builder, ty: ty::Ty) -> SpirvTy {
        unimplemented!()
    }
}

impl<'a, 'tcx: 'a> rustc::mir::visit::Visitor<'tcx> for RlslVisitor<'a, 'tcx> {
    fn visit_statement(
        &mut self,
        block: mir::BasicBlock,
        statement: &mir::Statement<'tcx>,
        location: mir::Location,
    ) {
//        match statement.kind {
//            mir::StatementKind::Assign(_, _) => println!("ASS"),
//            _ => (),
//        };
        self.super_statement(block, statement, location);
    }

    fn visit_lvalue(
        &mut self,
        lvalue: &mir::Lvalue<'tcx>,
        context: mir::visit::LvalueContext<'tcx>,
        location: mir::Location,
    ) {
        //println!("lvalue = {:?}", lvalue);
        self.super_lvalue(lvalue, context, location);
    }
    fn visit_local_decl(&mut self, local_decl: &mir::LocalDecl<'tcx>) {
        //println!("decl = {:?}", local_decl.ty);
        match local_decl.ty.sty {
            ty::TypeVariants::TyAdt(def, subs) => {
                //println!("variants {:#?}", def.variants);
                let def_id = def.variants.first().unwrap().did;
                let node_id = self.map.as_local_node_id(def_id).unwrap();
                let node = self.map.find(node_id).unwrap();
                match node {
                    hir::map::Node::NodeItem(ref item) => {
                        let intrinsic = extract_intrinsic(&item.attrs);
                        //println!("intrinsic = {:?}", intrinsic);
                    }
                    _ => (),
                };
                //println!("node = {:#?}", node);
            }
            _ => (),
        }
        self.super_local_decl(local_decl);
    }
}

#[derive(Debug)]
pub enum Intrinsic {
    Vec(u32),
}

pub fn extract_intrinsic(attr: &[syntax::ast::Attribute]) -> Option<Vec<Intrinsic>> {
    let spirv = attr.iter().filter_map(|a| a.meta()).find(|meta| {
        meta.name.as_str() == "spirv"
    });
    if let Some(spirv) = spirv {
        let list = spirv.meta_item_list().map(|nested_list| {
            nested_list
                .iter()
                .map(|nested_meta| match nested_meta.node {
                    syntax::ast::NestedMetaItemKind::Literal(ref lit) => {
                        match lit.node {
                            syntax::ast::LitKind::Str(ref sym, _) => Intrinsic::Vec(2),
                            ref rest => unimplemented!("{:?}", rest),
                        }
                    }
                    syntax::ast::NestedMetaItemKind::MetaItem(ref meta) => {
                        match &*meta.name.as_str() {
                            "Vec2" => Intrinsic::Vec(2),
                            ref rest => unimplemented!("{:?}", rest),
                        }
                    }
                })
                .collect::<Vec<_>>()
        });
        return list;
    }
    None
}
impl<'a, 'v: 'a> rustc::hir::intravisit::Visitor<'v> for RlslVisitor<'a, 'v> {
    fn nested_visit_map<'this>(&'this mut self) -> hir_visit::NestedVisitorMap<'this, 'v> {
        hir_visit::NestedVisitorMap::All(self.map)
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl, b: BodyId, s: Span, id: NodeId) {
        use rustc::ty::{Slice, FnSig};
        use rustc::ty;
        let def_id = self.map.local_def_id(id);
        let node = self.map.find(id);
        //println!("node = {:?}", node);
        let mir_fn = self.ty_ctx.maybe_optimized_mir(def_id).unwrap();
        //println!("mir_fn = {:#?}", mir_fn);
        //for v in mir::traversal::preorder(mir_fn){
        //    println!("{:?}", v);
        //}
        self.visit_mir(mir_fn);
        //println!("mir_fn = {:#?}", mir_fn);
        let sigs = self.ty_ctx.body_tables(b).liberated_fn_sigs();
        let hir_id = self.map.node_to_hir_id(id);
        let fn_sig = *sigs.get(hir_id).expect("sig");
        let ty = self.ty_ctx.mk_fn_ptr(ty::Binder(fn_sig));
        let fn_ty_spirv = self.ty_cache.from_ty(&mut self.ctx.builder, ty);
        let ret_ty_spirv = self.ty_cache.from_ty(
            &mut self.ctx.builder,
            fn_sig.output(),
        );
        let spirv_function = self.ctx
            .builder
            .begin_function(
                ret_ty_spirv.word,
                None,
                spirv::FunctionControl::empty(),
                fn_ty_spirv.word,
            )
            .expect("begin fn");
        println!("BLOCK");
        self.ctx.builder.begin_basic_block(None).expect("block");
        walk_fn(self, fk, fd, b, s, id);
        self.ctx.builder.end_function().expect("end fn");
    }
    fn visit_expr(&mut self, ex: &'v Expr) {
        walk_expr(self, ex);
        let spirv_expr = match ex.node {
            Expr_::ExprLit(ref lit) => {
                match lit.node {
                    syntax::ast::LitKind::FloatUnsuffixed(sym) => {
                        let ty = self.get_table().expr_ty(ex);
                        let spirv_ty = self.ty_cache.from_ty(&mut self.ctx.builder, ty);
                        let f = sym.as_str().parse::<f32>().expect("parse");
                        Some(SpirvExpr(self.ctx.builder.constant_f32(spirv_ty.word, f)))
                    }
                    ref rest => unimplemented!("{:?}", rest),
                }
            }
            //Expr_::ExprPath(ref path) => {
            //    if let &QPath::Resolved(_, ref p) = path {
            //        if let hir::def::Def::Local(l) = p.def {
            //            println!("EXPR {:?}", self.map.get_if_local(l));
            //        }
            //    }
            //}
            _ => None,
        };
        if let Some(spirv_expr) = spirv_expr {
            self.ctx.exprs.insert(ex.id, spirv_expr);
        }
    }
    fn visit_body(&mut self, b: &'v Body) {
        self.current_table.push(self.ty_ctx.body_tables(b.id()));
        walk_body(self, b);
        //for arg in b.arguments.iter() {
        //    match arg.pat.node {
        //        hir::PatKind::Binding(_, def_id, ..) => {
        //            //let local = self.ty_ctx.
        //            println!("local = {:?}", self.map.get_if_local(def_id));
        //        }
        //        ref rest => unimplemented!("{:?}", rest),
        //    };
        //}
        if let Some(ret_expr) = self.ctx.exprs.get(&b.value.id) {
            self.ctx.builder.ret_value(ret_expr.0).expect("ret value");
        } else {
            self.ctx.builder.ret().expect("ret");
        }
        let _ = self.current_table.pop();
    }
    fn visit_fn_decl(&mut self, fd: &'v FnDecl) {
        //println!("FN");
        walk_fn_decl(self, fd)
    }
    fn visit_ty(&mut self, t: &'v Ty) {
        //println!("Ty {:?}", t);
        walk_ty(self, t);
    }
    fn visit_local(&mut self, l: &'v Local) {

        walk_local(self, l);
        let ty = self.get_table().node_id_to_type(l.hir_id);
        let spirv_ty = self.ty_cache.from_ty(&mut self.ctx.builder, ty);
        let init_expr = l.init
            .as_ref()
            .and_then(|ex| self.ctx.exprs.get(&ex.id))
            .map(|ex| ex.0);
        let spirv_var = self.ctx.builder.variable(
            spirv_ty.word,
            None,
            spirv::StorageClass::Function,
            init_expr,
        );
        self.ctx.vars.insert(l.id, SpirvVar(spirv_var));
        //println!("local = {:?}", self.map.find(l.id));
    }
    fn visit_variant_data(
        &mut self,
        s: &'v VariantData,
        _: Symbol,
        _: &'v Generics,
        _parent_id: NodeId,
        _: Span,
    ) {
        walk_struct_def(self, s);
        //println!("typ {:#?}", self.map.find(s.id()));
    }
    fn visit_struct_field(&mut self, s: &'v StructField) {
        walk_struct_field(self, s)
    }
}

struct RlslCompilerCalls;

struct RlslMir;
use mir::transform::*;

impl mir::transform::PassHook for RlslMir {
    fn on_mir_pass<'a, 'tcx: 'a>(
        &self,
        tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
        suite: MirSuite,
        pass_num: MirPassIndex,
        pass_name: &str,
        source: MirSource,
        mir: &mir::Mir<'tcx>,
        is_after: bool,
    ) {
        println!("_________________________");
        //unimplemented!()
    }
}
use rustc::session::config::{ErrorOutputType, Input, self };
use std::path::PathBuf;
use syntax::ast;
use rustc_errors as errors;
impl<'a> CompilerCalls<'a> for RlslCompilerCalls {
    fn early_callback(&mut self,
                      _: &getopts::Matches,
                      _: &config::Options,
                      _: &ast::CrateConfig,
                      _: &errors::registry::Registry,
                      _: ErrorOutputType)
                      -> Compilation {
        println!("early");
        Compilation::Continue
    }
    fn late_callback(&mut self,
                     matches: &getopts::Matches,
                     sess: &Session,
                     input: &Input,
                     odir: &Option<PathBuf>,
                     ofile: &Option<PathBuf>)
                     -> Compilation {
        println!("late");
        Compilation::Continue
    }
    fn no_input(&mut self,
                matches: &getopts::Matches,
                _: &config::Options,
                _: &ast::CrateConfig,
                _: &Option<PathBuf>,
                _: &Option<PathBuf>,
                _: &errors::registry::Registry)
                -> Option<(Input, Option<PathBuf>)> {
        println!("no input");
        println!("matches = {:?}", matches.free);
        None
    }
    fn build_controller<'tcx>(
        &'tcx mut self,
        _: &rustc::session::Session,
        _: &getopts::Matches,
    ) -> CompileController<'a> {
        println!("TT");
        let mut controller = CompileController::basic();
        controller.keep_ast = true;
        controller.after_analysis.stop = Compilation::Stop;
        controller.after_hir_lowering.callback = box |s: &mut CompileState| {
            let sess = s.session;
            let hir_map = s.hir_map.unwrap();
            let analysis = s.analysis.unwrap();
            let reso = s.resolutions.unwrap();
            let arena = s.arena.unwrap();
            let arenas = s.arenas.unwrap();
            //rustc_driver::driver::phase_3_run_analysis_passes(
            //    sess,
            //    hir_map.clone(),
            //    analysis.clone(),
            //    reso.clone(),
            //    arena,
            //    arenas,
            //    "hello",
            //    |_, _, _, _| 4,
            //);
        };
        controller.after_analysis.callback = box |s: &mut CompileState| {
            let tcx = &s.tcx.unwrap();
            let time_passes = tcx.sess.time_passes();
            //let visitor = RlslVisitor::new(s);
            //visitor.build_module();
        };
        controller
    }
}
fn main() {
    env_logger::init();
    info!("hello");
    debug!("debug");
    error!("error");
    let mut calls = RlslCompilerCalls;
    let result = run(move || {
        let (a, b) = run_compiler(&get_args(), &mut calls, None, None);
        (a, b)
    });
}

//fn main() {}
