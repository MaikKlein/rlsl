#![feature(rustc_private)]
#![feature(box_syntax)]
//#[macro_use]
//extern crate debugit;
extern crate rustc;
extern crate rustc_driver;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
extern crate spirv_headers as spirv;
extern crate rspirv;
extern crate getopts;
extern crate env_logger;

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

pub struct SpirvCtx {
    pub builder: Builder,
}
impl SpirvCtx {
    pub fn new() -> Self {
        SpirvCtx { builder: Builder::new() }
    }
}

pub struct RlslVisitor<'a, 'tcx: 'a> {
    pub map: &'a hir::map::Map<'tcx>,
    pub ty_ctx: rustc::ty::TyCtxt<'a, 'tcx, 'tcx>,
    current_table: Option<&'a rustc::ty::TypeckTables<'tcx>>,
    pub ty_cache: SpirvTyCache<'tcx>,
    ctx: SpirvCtx,
}
impl<'a, 'tcx: 'a> RlslVisitor<'a, 'tcx> {
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
            current_table: None,
            ctx,
            ty_cache: SpirvTyCache::new(),
        };
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
#[derive(Copy, Clone)]
pub struct SpirvTy {
    pub ty: spirv::Word,
}
impl From<spirv::Word> for SpirvTy {
    fn from(word: spirv::Word) -> SpirvTy {
        SpirvTy { ty: word }
    }
}
use std::collections::HashMap;
pub struct SpirvTyCache<'a> {
    pub ty_cache: HashMap<rustc::ty::Ty<'a>, SpirvTy>,
}

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
                    .map(|ty| self.from_ty(builder, ty).ty)
                    .collect();
                builder.type_function(ret_ty.ty, &input_ty).into()
            }
            ref r => {
                println!("r = {:?}", r);
                unimplemented!()
            }
        };
        self.ty_cache.insert(ty, spirv_type);
        spirv_type
    }
}
impl<'a, 'v: 'a> rustc::hir::intravisit::Visitor<'v> for RlslVisitor<'a, 'v> {
    fn nested_visit_map<'this>(&'this mut self) -> hir_visit::NestedVisitorMap<'this, 'v> {
        hir_visit::NestedVisitorMap::All(self.map)
    }

    fn visit_fn(&mut self, fk: FnKind<'v>, fd: &'v FnDecl, b: BodyId, s: Span, id: NodeId) {
        use rustc::ty::{Slice, FnSig};
        use rustc::ty;
        walk_fn(self, fk, fd, b, s, id);
        let hir_id = self.map.node_to_hir_id(id);
        let ty = self.current_table
            .map(|t| {
                let sigs = t.liberated_fn_sigs();
                let fn_sig = *sigs.get(hir_id).expect("sig");
                self.ty_ctx.mk_fn_ptr(ty::Binder(fn_sig))
            })
            .expect("fn_ptr ty");
        let fn_ty_spirv = self.ty_cache.from_ty(&mut self.ctx.builder, ty);
    }
    fn visit_expr(&mut self, ex: &'v Expr) {
        let ty = self.current_table.and_then(|t| t.expr_ty_opt(ex));
        println!("ex ty = {:?}", ty);
        walk_expr(self, ex);
    }
    fn visit_body(&mut self, b: &'v Body) {
        let old_table = self.current_table;
        self.current_table = Some(self.ty_ctx.body_tables(b.id()));
        walk_body(self, b);
        //self.current_table = old_table;
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
        let ty = self.current_table.and_then(
            |t| t.node_id_to_type_opt(l.hir_id),
        );
        println!("local = {:?}", ty);
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
impl<'a> CompilerCalls<'a> for RlslCompilerCalls {
    fn build_controller<'tcx>(
        &'tcx mut self,
        _: &rustc::session::Session,
        _: &getopts::Matches,
    ) -> CompileController<'a> {
        let mut controller = CompileController::basic();
        controller.keep_ast = true;
        controller.after_analysis.stop = Compilation::Stop;
        controller.after_analysis.callback = box |s: &mut CompileState| {
            let visitor = RlslVisitor::new(s);
            visitor.build_module();
        };
        controller
    }
}
fn main() {
    env_logger::init().unwrap();
    let mut calls = RlslCompilerCalls;
    let result = run(move || {
        let (a, b) = run_compiler(&get_args(), &mut calls, None, None);
        //println!("R = {:?}", a);
        (a, b)
    });
}

//fn main() {}
