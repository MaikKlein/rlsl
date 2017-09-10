pub mod terminator;
pub mod rvalue;
pub mod tycache;

use rspirv;
use std::collections::HashMap;
use rustc;
use rustc::{hir, mir};
use spirv;
use syntax::ast::NodeId;
use rustc_driver::driver::CompileState;
use rustc_data_structures::fx::FxHashSet;
use rustc_trans::{SharedCrateContext, TransItem};
use rustc::ty;
use rustc_trans;
use rustc_trans::find_exported_symbols;
use rustc_trans::back::symbol_export::ExportedSymbols;
use rustc::session::config::OutputFilenames;
use rspirv::mr::Builder;
use rustc_data_structures::indexed_vec::IndexVec;
use syntax;
pub fn trans_items<'a, 'tcx>(
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    analysis: ty::CrateAnalysis,
    //incremental_hashes_map: IncrementalHashesMap,
    output_filenames: &OutputFilenames,
) -> FxHashSet<TransItem<'tcx>> {
    let ty::CrateAnalysis { reachable, .. } = analysis;
    let check_overflow = tcx.sess.overflow_checks();
    //let link_meta = link::build_link_meta(&incremental_hashes_map);
    let exported_symbol_node_ids = find_exported_symbols(tcx, &reachable);
    let shared_ccx = SharedCrateContext::new(tcx, check_overflow, output_filenames);
    let exported_symbols = ExportedSymbols::compute(tcx, &exported_symbol_node_ids);
    let (items, _) = rustc_trans::collector::collect_crate_translation_items(
        &shared_ccx,
        &exported_symbols,
        rustc_trans::collector::TransItemCollectionMode::Lazy,
    );
    items
}

pub struct SpirvCtx<'a> {
    pub builder: Builder,
    pub ty_cache: tycache::SpirvTyCache<'a>,
    pub vars: HashMap<mir::Local, SpirvVar>,
    pub exprs: HashMap<mir::Location, SpirvExpr>,
}
impl<'a> SpirvCtx<'a> {
    pub fn load_operand(
        &mut self,
        local_decls: &IndexVec<mir::Local, mir::LocalDecl<'a>>,
        operand: &mir::Operand,
    ) -> SpirvExpr {
        match operand {
            &mir::Operand::Consume(ref lvalue) => match lvalue {
                &mir::Lvalue::Local(local) => {
                    let local_decl = &local_decls[local];
                    let spirv_ty = self.from_ty(local_decl.ty);
                    let spirv_var = self.vars.get(&local).expect("local");

                    let expr = self.builder
                        .load(spirv_ty.word, None, spirv_var.0, None, &[])
                        .expect("load");
                    let spirv_expr = SpirvExpr(expr);
                    spirv_expr
                }
                ref rest => unimplemented!("{:?}", rest),
            },
            &mir::Operand::Constant(ref constant) => match constant.literal {
                mir::Literal::Value { ref value } => {
                    use rustc::middle::const_val::ConstVal;
                    match value {
                        &ConstVal::Float(f) => {
                            use syntax::ast::FloatTy;
                            match f.ty {
                                FloatTy::F32 => {
                                    let val: f32 = unsafe { ::std::mem::transmute(f.bits as u32) };
                                    SpirvExpr(self.builder.constant_f32(0, val))
                                }
                                _ => panic!("f64 not supported"),
                            }
                        }
                        ref rest => unimplemented!("{:?}", rest),
                    }
                }
                ref rest => unimplemented!("{:?}", rest),
            },
            ref rest => unimplemented!("{:?}", rest),
        }
    }
    pub fn from_ty(&mut self, ty: ty::Ty<'a>) -> SpirvTy {
        self.ty_cache.from_ty(&mut self.builder, ty)
    }
    pub fn new() -> Self {
        SpirvCtx {
            builder: Builder::new(),
            ty_cache: tycache::SpirvTyCache::new(),
            vars: HashMap::new(),
            exprs: HashMap::new(),
        }
    }
}
#[derive(Copy, Clone, Debug)]
pub struct SpirvVar(pub spirv::Word);
#[derive(Copy, Clone, Debug)]
pub struct SpirvExpr(pub spirv::Word);
#[derive(Copy, Clone, Debug)]
pub struct SpirvTy {
    pub word: spirv::Word,
}
impl From<spirv::Word> for SpirvTy {
    fn from(word: spirv::Word) -> SpirvTy {
        SpirvTy { word: word }
    }
}
pub struct RlslVisitor<'a, 'tcx: 'a> {
    pub map: &'a hir::map::Map<'tcx>,
    pub ty_ctx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    current_table: Vec<&'a rustc::ty::TypeckTables<'tcx>>,
    pub mir: Option<&'a mir::Mir<'tcx>>,
    ctx: SpirvCtx<'tcx>,
}

#[derive(Debug)]
pub enum Intrinsic {
    Vec(u32),
}
pub fn extract_intrinsic(attr: &[syntax::ast::Attribute]) -> Option<Vec<Intrinsic>> {
    let spirv = attr.iter()
        .filter_map(|a| a.meta())
        .find(|meta| meta.name.as_str() == "spirv");
    if let Some(spirv) = spirv {
        let list = spirv.meta_item_list().map(|nested_list| {
            nested_list
                .iter()
                .map(|nested_meta| match nested_meta.node {
                    syntax::ast::NestedMetaItemKind::Literal(ref lit) => match lit.node {
                        syntax::ast::LitKind::Str(ref sym, _) => Intrinsic::Vec(2),
                        ref rest => unimplemented!("{:?}", rest),
                    },
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

pub fn trans_spirv<'a, 'tcx>(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>, items: &FxHashSet<TransItem>) {
    use rustc::mir::visit::Visitor;
    let mut visitor = RlslVisitor::new(tcx);
    for item in items {
        if let &TransItem::Fn(ref instance) = item {
            let mir = tcx.maybe_optimized_mir(instance.def_id());
            if let Some(ref mir) = mir {
                visitor.mir = Some(mir);
                visitor.visit_mir(mir);
                visitor.mir = None;
            }
        }
    }
    visitor.build_module();
}

pub fn resolve_fn_call<'a, 'tcx>(
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    def_id: hir::def_id::DefId,
    substs: &'tcx ty::subst::Substs<'tcx>,
) -> hir::def_id::DefId {
    if let Some(trait_id) = tcx.opt_associated_item(def_id).and_then(
        |associated_item| match associated_item.container {
            ty::TraitContainer(def_id) => Some(def_id),
            ty::ImplContainer(_) => None,
        },
    ) {
        let tref = ty::TraitRef::from_method(tcx, trait_id, substs);
        let assoc_item = tcx.associated_items(trait_id)
            .find(|a| a.def_id == def_id)
            .unwrap();
        let vtable = tcx.trans_fulfill_obligation(syntax::codemap::DUMMY_SP, ty::Binder(tref));
        use rustc::traits::Vtable;
        return match vtable {
            Vtable::VtableImpl(ref data) => {
                let (def_id, substs) =
                    rustc::traits::find_associated_item(tcx, &assoc_item, substs, data);
                def_id
            }
            _ => unimplemented!(),
        };
    }
    unimplemented!()
}
impl<'a, 'tcx: 'a> RlslVisitor<'a, 'tcx> {
    pub fn get_table(&self) -> &'a ty::TypeckTables<'tcx> {
        self.current_table.last().expect("no table yet")
    }
    pub fn new(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>) -> Self {
        let mut ctx = SpirvCtx::new();
        ctx.builder.capability(spirv::Capability::Shader);
        ctx.builder.ext_inst_import("GLSL.std.450");
        ctx.builder
            .memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        let mut visitor = RlslVisitor {
            map: &tcx.hir,
            ty_ctx: tcx,
            current_table: Vec::new(),
            ctx,
            mir: None,
        };
        visitor
    }
    pub fn build_module(self) {
        use rspirv::binary::Assemble;
        use rspirv::binary::Disassemble;
        use std::mem::size_of;
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create("shader.spv").unwrap();
        let spirv_module = self.ctx.builder.module();
        let bytes: Vec<u8> = spirv_module
            .assemble()
            .iter()
            .flat_map(|val| {
                (0..size_of::<u32>()).map(move |i| ((val >> (8 * i)) & 0xff) as u8)
            })
            .collect();
        let mut loader = rspirv::mr::Loader::new();
        //let bytes = b.module().assemble_bytes();
        rspirv::binary::parse_bytes(&bytes, &mut loader);
        println!("{}", loader.module().disassemble());
        f.write_all(&bytes);
    }
}
impl<'a, 'tcx: 'a> rustc::mir::visit::Visitor<'tcx> for RlslVisitor<'a, 'tcx> {
    fn visit_statement(
        &mut self,
        block: mir::BasicBlock,
        statement: &mir::Statement<'tcx>,
        location: mir::Location,
    ) {
        self.super_statement(block, statement, location);
    }
    fn visit_mir(&mut self, mir: &mir::Mir<'tcx>) {
        let ret_ty_spirv = self.ctx.from_ty(mir.return_ty);
        let args_ty = mir.args_iter().map(|l| mir.local_decls[l].ty);
        let fn_sig = self.ty_ctx.mk_fn_sig(
            args_ty,
            mir.return_ty,
            false,
            hir::Unsafety::Normal,
            syntax::abi::Abi::Rust,
        );
        let fn_ty = self.ty_ctx.mk_fn_ptr(ty::Binder(fn_sig));
        let fn_ty_spirv = self.ctx.from_ty(fn_ty);

        let spirv_function = self.ctx
            .builder
            .begin_function(
                ret_ty_spirv.word,
                None,
                spirv::FunctionControl::empty(),
                fn_ty_spirv.word,
            )
            .expect("begin fn");
        self.ctx.builder.begin_basic_block(None).expect("block");
        for local_arg in mir.args_iter() {
            let local_decl = &mir.local_decls[local_arg];
            let spirv_arg_ty = self.ctx.from_ty(local_decl.ty);
            let spirv_param = self.ctx
                .builder
                .function_parameter(spirv_arg_ty.word)
                .expect("fn param");
            self.ctx.vars.insert(local_arg, SpirvVar(spirv_param));
        }
        for local_var in mir.vars_and_temps_iter() {
            let local_decl = &mir.local_decls[local_var];
            let spirv_var_ty = self.ctx.from_ty(local_decl.ty);
            let spirv_var = self.ctx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.ctx.vars.insert(local_var, SpirvVar(spirv_var));
        }
        {
            use rustc_data_structures::indexed_vec::Idx;
            let local = mir::Local::new(0);
            let local_decl = &mir.local_decls[local];
            let spirv_var_ty = self.ctx.from_ty(local_decl.ty);
            let spirv_var = self.ctx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.ctx.vars.insert(local, SpirvVar(spirv_var));
        }
        self.super_mir(mir);
        match mir.return_ty.sty {
            ty::TypeVariants::TyTuple(ref slice, _) if slice.len() == 0 => {
                self.ctx.builder.ret().expect("ret");
            }
            _ => {
                use rustc_data_structures::indexed_vec::Idx;
                let var = self.ctx.vars.get(&mir::Local::new(0)).unwrap();
                self.ctx.builder.ret_value(var.0).expect("ret value");
            }
        };
        self.ctx.builder.end_function().expect("end fn");
    }
    fn visit_assign(
        &mut self,
        block: mir::BasicBlock,
        lvalue: &mir::Lvalue<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: mir::Location,
    ) {
        self.super_assign(block, lvalue, rvalue, location);
        let ty = rvalue.ty(&self.mir.unwrap().local_decls, self.ty_ctx);
        if let ty::TypeVariants::TyTuple(ref slice, _) = ty.sty{
            if slice.len() == 0 {
                return;
            }
        }
        println!("lvalue = {:?}", lvalue);
        println!("ty = {:?}", ty);
        match lvalue {
            &mir::Lvalue::Local(local) => {
                let expr = self.ctx.exprs.get(&location);
                println!("expr = {:?}", expr);
            }
            rest => unimplemented!("{:?}", rest),
        };
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
    fn visit_terminator_kind(
        &mut self,
        block: mir::BasicBlock,
        kind: &mir::TerminatorKind<'tcx>,
        location: mir::Location,
    ) {
        self.super_terminator_kind(block, kind, location);
        //        match kind {
        //            &mir::TerminatorKind::Call { ref func, .. } => {
        //                match func {
        //                    &mir::Operand::Constant(ref constant) => {
        //                        println!("const ty {:?}", constant.ty);
        //                        if let mir::Literal::Value { ref value } = constant.literal {
        //                            use rustc::middle::const_val::ConstVal;
        //                            if let &ConstVal::Function(def_id, ref subst) = value {
        //                                //                                let mir = self.ty_ctx.maybe_optimized_mir(def_id).or_else(||
        //                                //                                                                                              self.ty_ctx.maybe_optimized_mir(
        //                                //                                                                                                  resolve_fn_call(self.ty_ctx, def_id, subst),
        //                                //                                                                                              ),
        //                                //                                );
        //                                //println!("fn call {:#?}", mir);
        //                            }
        //                        }
        //                    }
        //                    _ => (),
        //                }
        //            }
        //            _ => (),
        //        };
    }
    fn visit_local_decl(&mut self, local_decl: &mir::LocalDecl<'tcx>) {
        self.super_local_decl(local_decl);
    }
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: mir::Location) {
        use rustc::mir::HasLocalDecls;
        self.super_rvalue(rvalue, location);
        //println!("location = {:?}", location);
        let local_decls = &self.mir.unwrap().local_decls;
        match rvalue {
            &mir::Rvalue::BinaryOp(op, ref l, ref r) => {
                let l_load = self.ctx.load_operand(local_decls, l);
                let r_load = self.ctx.load_operand(local_decls, r);
                let spirv_ty = self.ctx.from_ty(rvalue.ty(local_decls, self.ty_ctx));
                let add = self.ctx
                    .builder
                    .fadd(spirv_ty.word, None, l_load.0, r_load.0)
                    .expect("fadd");
                self.ctx.exprs.insert(location, SpirvExpr(add));
            }
            &mir::Rvalue::Use(ref operand) => {
                let load = self.ctx.load_operand(local_decls, operand);
                self.ctx.exprs.insert(location, load);

            }
            &mir::Rvalue::NullaryOp(..) => {}
            &mir::Rvalue::CheckedBinaryOp(..) => {}
            &mir::Rvalue::Discriminant(..) => {}
            &mir::Rvalue::Aggregate(..) => {}
            rest => unimplemented!("{:?}", rest),
        }
    }
}
