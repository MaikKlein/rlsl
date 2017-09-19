pub mod terminator;
pub mod rvalue;

use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::control_flow_graph::ControlFlowGraph;
use rustc::mir::visit::Visitor;
use rustc::hir::map::Map;
use rspirv;
use std::collections::HashMap;
use rustc;
use rustc::{hir, mir};
use spirv;
use rustc_data_structures::fx::FxHashSet;
use rustc_trans::{SharedCrateContext, TransItem};
use rustc::ty;
use rustc_trans;
use rustc_trans::find_exported_symbols;
use rustc_trans::back::symbol_export::ExportedSymbols;
use rustc::session::config::OutputFilenames;
use rspirv::mr::Builder;
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

pub struct SpirvCtx<'a, 'tcx: 'a> {
    pub tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    pub builder: Builder,
    pub ty_cache: HashMap<rustc::ty::Ty<'tcx>, SpirvTy>,
    pub forward_fns: HashMap<hir::def_id::DefId, SpirvFn>,
}

pub type MergeCollector = HashMap<mir::Location, mir::BasicBlock>;
pub fn merge_collector(mir: &mir::Mir) -> MergeCollector {
    let mut merge_collector = HashMap::new();
    for (block, ref data) in mir.basic_blocks().iter_enumerated() {
        let location = mir::Location {
            block,
            statement_index: data.statements.len(),
        };
        if let Some(merge_block) = merge_collector_impl(mir, block, data) {
            merge_collector.insert(location, merge_block);
        }
    }
    merge_collector
}
fn merge_collector_impl(
    mir: &mir::Mir,
    block: mir::BasicBlock,
    block_data: &mir::BasicBlockData,
) -> Option<mir::BasicBlock> {
    if let mir::TerminatorKind::SwitchInt { ref targets, .. } = block_data.terminator().kind {
        let target = targets
            .iter()
            .filter_map(|&block| {
                let block_data = &mir.basic_blocks()[block];
                match block_data.terminator().kind {
                    mir::TerminatorKind::Goto { target } => Some(target),
                    _ => mir.successors(block)
                        .filter_map(|successor_block| {
                            let data = &mir.basic_blocks()[successor_block];
                            merge_collector_impl(mir, successor_block, data)
                        })
                        .nth(0),
                }
            })
            .nth(0);
        return target;
    }
    None
}


// Collects constants when they appear in function calls
pub struct ConstantArgCollector<'tcx> {
    constants: Vec<mir::Constant<'tcx>>,
}
impl<'tcx> rustc::mir::visit::Visitor<'tcx> for ConstantArgCollector<'tcx> {
    fn visit_constant(&mut self, constant: &mir::Constant<'tcx>, location: mir::Location) {
        self.constants.push(constant.clone());
    }
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: mir::Location) {}
    fn visit_terminator_kind(
        &mut self,
        block: mir::BasicBlock,
        kind: &mir::TerminatorKind<'tcx>,
        location: mir::Location,
    ) {
        //self.super_terminator_kind(block, kind, location);
        match kind {
            &mir::TerminatorKind::Call { ref args, .. } => for arg in args {
                self.visit_operand(arg, location);
            },
            _ => (),
        };
    }
}

pub fn collect_constants<'tcx>(mir: &mir::Mir<'tcx>) -> Vec<mir::Constant<'tcx>> {
    let mut v = ConstantArgCollector {
        constants: Vec::new(),
    };
    v.visit_mir(mir);
    v.constants
}
pub enum SpirvOperand<'tcx> {
    Consume(SpirvVar<'tcx>),
    ConstVal(SpirvExpr),
}
impl<'tcx> SpirvOperand<'tcx> {
    pub fn is_param(&self) -> bool {
        match self {
            &SpirvOperand::Consume(ref var) => var.is_param(),
            _ => false,
        }
    }
    pub fn expect_var(self) -> SpirvVar<'tcx> {
        match self {
            SpirvOperand::Consume(var) => var,
            _ => panic!("Expected var"),
        }
    }
    pub fn load_raw<'a, 'b>(self, ctx: &'b mut SpirvCtx<'a, 'tcx>, ty: SpirvTy) -> spirv::Word {
        match self {
            SpirvOperand::Consume(var) => if var.is_ptr() {
                // If the variable is a ptr, then we need to load the value
                ctx.builder
                    .load(ty.word, None, var.word, None, &[])
                    .expect("load")
            } else {
                // Otherwise we can just use the value
                var.word
            },
            SpirvOperand::ConstVal(expr) => expr.0,
        }
    }
    pub fn into_raw_word(self) -> spirv::Word {
        match self {
            SpirvOperand::Consume(var) => var.word,
            SpirvOperand::ConstVal(expr) => expr.0,
        }
    }
}

impl<'a, 'tcx> SpirvCtx<'a, 'tcx> {
    pub fn build_module(self) {
        use rspirv::binary::Assemble;
        use rspirv::binary::Disassemble;
        use std::mem::size_of;
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create("shader.spv").unwrap();
        let spirv_module = self.builder.module();
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
    pub fn from_ty(&mut self, ty: rustc::ty::Ty<'tcx>) -> SpirvTy {
        use rustc::ty::TypeVariants;
        let ty = match ty.sty {
            TypeVariants::TyRef(_, type_and_mut) => {
                let t = ty::TypeAndMut {
                    ty: type_and_mut.ty,
                    mutbl: rustc::hir::Mutability::MutMutable,
                };
                self.tcx.mk_ptr(t)
            }
            _ => ty,
        };
        if let Some(ty) = self.ty_cache.get(ty) {
            return *ty;
        }
        let spirv_type: SpirvTy = match ty.sty {
            TypeVariants::TyBool => self.builder.type_bool().into(),
            TypeVariants::TyUint(uint_ty) => self.builder
                .type_int(uint_ty.bit_width().unwrap() as u32, 0)
                .into(),
            TypeVariants::TyFloat(f_ty) => {
                use syntax::ast::FloatTy;
                match f_ty {
                    FloatTy::F32 => self.builder.type_float(32).into(),
                    FloatTy::F64 => self.builder.type_float(64).into(),
                }
            }
            TypeVariants::TyTuple(slice, _) if slice.len() == 0 => self.builder.type_void().into(),
            TypeVariants::TyFnPtr(sig) => {
                let ret_ty = self.from_ty(sig.output().skip_binder());
                let input_ty: Vec<_> = sig.inputs()
                    .skip_binder()
                    .iter()
                    .map(|ty| self.from_ty(ty).word)
                    .collect();
                self.builder.type_function(ret_ty.word, &input_ty).into()
            }
            TypeVariants::TyRawPtr(type_and_mut) => {
                let inner = self.from_ty(type_and_mut.ty);
                self.builder
                    .type_pointer(None, spirv::StorageClass::Function, inner.word)
                    .into()
            }
            TypeVariants::TyParam(ref param) => {
                let ty = param.to_ty(self.tcx);
                println!("param ty = {:?}", ty);
                self.from_ty(ty).into()
            }
            TypeVariants::TyAdt(adt, substs) => match adt.adt_kind() {
                ty::AdtKind::Struct => {
                    let node_id = self.tcx.hir.as_local_node_id(adt.did);
                    let node = node_id.and_then(|id| self.tcx.hir.find(id));
                    let item = node.and_then(|node| match node {
                        hir::map::Node::NodeItem(item) => Some(item),
                        _ => None,
                    });
                    let intrinsic = item.and_then(|item| {
                        item.attrs
                            .iter()
                            .filter_map(|attr| {
                                extract_spirv_attr(attr, &[], |s| match s {
                                    "Vec2" => Some(IntrinsicType::Vec(2)),
                                    _ => None,
                                })
                            })
                            .nth(0)
                    });

                    if let Some(intrinsic) = intrinsic {
                        let intrinsic_spirv = match intrinsic {
                            IntrinsicType::Vec(dim) => {
                                let field_ty = adt.all_fields()
                                    .nth(0)
                                    .map(|f| f.ty(self.tcx, substs))
                                    .expect("no field");
                                let spirv_ty = self.from_ty(field_ty);
                                self.builder.type_vector(spirv_ty.word, dim as u32).into()
                            }
                            ref r => unimplemented!("{:?}", r),
                        };
                        return intrinsic_spirv;
                    }
                    let field_ty_spirv: Vec<_> = adt.all_fields()
                        .map(|f| {
                            let ty = f.ty(self.tcx, substs);
                            self.from_ty(ty).word
                        })
                        .collect();
                    self.builder.type_struct(&field_ty_spirv).into()
                }
                ref r => unimplemented!("{:?}", r),
            },
            ref r => unimplemented!("{:?}", r),
        };
        self.ty_cache.insert(ty, spirv_type);
        spirv_type
    }

    pub fn from_ty_as_ptr<'gcx>(&mut self, ty: ty::Ty<'tcx>) -> SpirvTy {
        let t = ty::TypeAndMut {
            ty,
            mutbl: rustc::hir::Mutability::MutMutable,
        };
        let ty_ptr = self.tcx.mk_ptr(t);
        self.from_ty(ty_ptr)
    }
    pub fn new(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>) -> SpirvCtx<'a, 'tcx> {
        let mut builder = Builder::new();
        builder.capability(spirv::Capability::Shader);
        builder.ext_inst_import("GLSL.std.450");
        builder.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        SpirvCtx {
            builder,
            ty_cache: HashMap::new(),
            forward_fns: HashMap::new(),
            tcx,
        }
    }
}
#[derive(Copy, Clone, Debug)]
pub struct SpirvLabel(pub spirv::Word);
#[derive(Copy, Clone, Debug)]
pub struct SpirvFn(pub spirv::Word);
#[derive(Copy, Clone, Debug)]
pub struct SpirvVar<'tcx> {
    word: spirv::Word,
    is_param: bool,
    ty: ty::Ty<'tcx>,
}

impl<'tcx> SpirvVar<'tcx> {
    pub fn new(word: spirv::Word, is_param: bool, ty: ty::Ty<'tcx>) -> Self {
        SpirvVar { word, is_param, ty }
    }
    pub fn is_var(&self) -> bool {
        !self.is_param
    }
    pub fn is_param(&self) -> bool {
        self.is_param
    }
    pub fn is_ptr(&self) -> bool {
        if self.is_var() {
            true
        } else {
            self.ty.is_unsafe_ptr() || self.ty.is_mutable_pointer() || self.ty.is_region_ptr()
        }
    }
}
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

pub struct RlslVisitor<'b, 'a: 'b, 'tcx: 'a> {
    pub map: &'a hir::map::Map<'tcx>,
    current_table: Vec<&'a rustc::ty::TypeckTables<'tcx>>,
    pub mir: &'a mir::Mir<'tcx>,
    pub entry: Option<IntrinsicEntry>,
    pub def_id: hir::def_id::DefId,
    pub merge_collector: Option<MergeCollector>,
    pub constants: HashMap<mir::Constant<'tcx>, SpirvVar<'tcx>>,
    pub label_blocks: HashMap<mir::BasicBlock, SpirvLabel>,
    pub ctx: &'b mut SpirvCtx<'a, 'tcx>,
    pub vars: HashMap<mir::Local, SpirvVar<'tcx>>,
    pub exprs: HashMap<mir::Location, SpirvExpr>,
}

#[derive(Debug)]
pub enum IntrinsicType {
    Vec(usize),
}
#[derive(Debug)]
pub enum IntrinsicEntry {
    Vertex,
    Fragment,
}

pub fn extract_spirv_attr<'a, R, F>(
    attr: &'a syntax::ast::Attribute,
    keywords: &[&str],
    f: F,
) -> Option<R>
where
    F: Fn(&str) -> Option<R>,
{
    let meta_item_list = attr.name().and_then(|name| if name.as_str() == "spirv" {
        attr.meta_item_list()
    } else {
        None
    });
    if let Some(meta_item_list) = meta_item_list {
        let iter = keywords
            .iter()
            .zip(meta_item_list.iter())
            .filter_map(|(keyword, meta_item)| {
                if let syntax::ast::NestedMetaItemKind::MetaItem(ref meta) = meta_item.node {
                    if &meta.name().as_str() == keyword {
                        return Some(());
                    }
                }
                None
            });

        if iter.count() == keywords.len() {
            return meta_item_list.get(keywords.len()).and_then(|meta_item| {
                if let syntax::ast::NestedMetaItemKind::MetaItem(ref meta) = meta_item.node {
                    return f(&*meta.name().as_str());
                }
                None
            });
        }
    }
    None
}
pub struct MirContext<'a, 'tcx: 'a> {
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a mir::Mir<'tcx>,
    substs: &'tcx ty::subst::Substs<'tcx>,
}
impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn monomorphize<T>(&self, value: &T) -> T
    where
        T: rustc::infer::TransNormalize<'tcx>,
    {
        self.tcx.trans_apply_param_substs(self.substs, value)
    }
}


use std::fmt::{Debug, Error, Formatter};
impl<'a, 'tcx> Debug for MirContext<'a, 'tcx> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.write_fmt(format_args!("{:#?}", self.mir))
    }
}

pub struct CollectMirContext<'a, 'tcx: 'a> {
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    contexts: Vec<MirContext<'a, 'tcx>>,
}
pub fn collect_mir_context<'a, 'tcx>(
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a mir::Mir<'tcx>,
) -> Vec<MirContext<'a, 'tcx>> {
    let mut collector = CollectMirContext {
        tcx,
        contexts: Vec::new(),
    };
    collector.visit_mir(mir);
    collector.contexts
}
impl<'a, 'tcx> rustc::mir::visit::Visitor<'tcx> for CollectMirContext<'a, 'tcx> {
    fn visit_terminator_kind(
        &mut self,
        block: mir::BasicBlock,
        kind: &mir::TerminatorKind<'tcx>,
        location: mir::Location,
    ) {
        self.super_terminator_kind(block, kind, location);
        if let &mir::TerminatorKind::Call { ref func, .. } = kind {
            if let &mir::Operand::Constant(ref constant) = func {
                if let mir::Literal::Value { ref value } = constant.literal {
                    use rustc::middle::const_val::ConstVal;
                    if let &ConstVal::Function(def_id, substs) = value {
                        let resolve_def_id = resolve_fn_call(self.tcx, def_id, substs);
                        let mir = self.tcx.maybe_optimized_mir(resolve_def_id);
                        self.contexts.push(MirContext {
                            tcx: self.tcx,
                            mir: mir.expect("mir"),
                            substs,
                        });
                    }
                }
            }
        }
    }
}

pub fn trans_spirv<'a, 'tcx>(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>, items: &'a FxHashSet<TransItem<'tcx>>) {
    use rustc::mir::visit::Visitor;
    let mut ctx = SpirvCtx::new(tcx);
    for item in items {
        if let &TransItem::Fn(ref instance) = item {
            ctx.forward_fns
                .insert(instance.def_id(), SpirvFn(ctx.builder.id()));
        }
    }
    //    let entry_points = items
    //        .iter()
    //        .filter_map(|item| {
    //            if let &TransItem::Fn(ref instance) = item {
    //                let mir = tcx.maybe_optimized_mir(instance.def_id()).expect("mir");
    //                let map: &Map = &tcx.hir;
    //                if let Some( node_id ) = map.as_local_node_id(instance.def_id()){
    //                    let name = map.name(node_id);
    //                    let node = map.find(node_id).expect("node");
    //                    if let rustc::hir::map::Node::NodeItem(ref item) = node {
    //                        let intrinsic = item.attrs
    //                            .iter()
    //                            .filter_map(|attr| {
    //                                extract_spirv_attr(attr, &[], |s| match s {
    //                                    "vertex" => Some(IntrinsicEntry::Vertex),
    //                                    "fragment" => Some(IntrinsicEntry::Fragment),
    //                                    _ => None,
    //                                })
    //                            })
    //                            .nth(0);
    //                        if let Some(intrinsic) = intrinsic {
    //                            return Some(mir);
    //                        }
    //                    }
    //                }
    //            }
    //            None
    //        })
    //        .nth(0)
    //        .expect("enty point");
    //    let contexts = collect_mir_context(tcx, entry_points);
    //    for context in &contexts{
    //        let ty = context.mir.return_ty;
    //        let mono_ty = context.monomorphize(&ty);
    //        println!("ty = {:?}", ty);
    //        println!("mono_ty = {:?}", mono_ty);
    //    }
    for item in items {
        if let &TransItem::Fn(ref instance) = item {
            //let resolved_def_id = resolve_fn_call(tcx, )
            let mir = tcx.maybe_optimized_mir(instance.def_id());
            let map: &Map = &tcx.hir;
            //            let node_id = map.as_local_node_id(instance.def_id()).expect("node id");
            //            let name = map.name(node_id);
            //            let node = map.find(node_id).expect("node");
            //println!("node = {:#?}", node);
            if let Some(ref mir) = mir {
                let mir_ctx = MirContext {
                    tcx,
                    substs: instance.substs,
                    mir,
                };
                let ty = mir.return_ty;
                let mono_ty = mir_ctx.monomorphize(&ty);
                println!("ty = {:?}", ty);
                println!("mono_ty = {:?}", mono_ty);
                //println!("mir = {:#?}", mir);
                //let mut visitor = RlslVisitor::new(&mut ctx, tcx, mir, instance.def_id());
                //                if let rustc::hir::map::Node::NodeItem(ref item) = node {
                //                    let intrinsic = item.attrs
                //                        .iter()
                //                        .filter_map(|attr| {
                //                            extract_spirv_attr(attr, &[], |s| match s {
                //                                "vertex" => Some(IntrinsicEntry::Vertex),
                //                                "fragment" => Some(IntrinsicEntry::Fragment),
                //                                _ => None,
                //                            })
                //                        })
                //                        .nth(0);
                //                    //println!("intrinsic = {:?}", intrinsic);
                //                    //visitor.entry = intrinsic;
                //                }
                //visitor.merge_collector = Some(merge_collector(mir));
                //visitor.visit_mir(mir);
            }
        }
    }
    ctx.build_module();
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
    def_id
}
impl<'b, 'a, 'tcx: 'a> RlslVisitor<'b, 'a, 'tcx> {
    pub fn load_operand<'gcx>(&mut self, operand: &mir::Operand<'tcx>) -> SpirvOperand<'tcx> {
        let mir = self.mir;
        let local_decls = &mir.local_decls;
        let ty = operand.ty(local_decls, self.ctx.tcx);
        let spirv_ty = self.ctx.from_ty(ty);
        match operand {
            &mir::Operand::Consume(ref lvalue) => match lvalue {
                &mir::Lvalue::Local(local) => {
                    let local_decl = &local_decls[local];
                    let spirv_ty = self.ctx.from_ty(local_decl.ty);
                    let spirv_var = self.vars.get(&local).expect("local");
                    SpirvOperand::Consume(*spirv_var)
                }
                &mir::Lvalue::Projection(ref proj) => match &proj.elem {
                    &mir::ProjectionElem::Field(field, ty) => match &proj.base {
                        &mir::Lvalue::Local(local) => {
                            let var = self.vars.get(&local).expect("local");
                            let spirv_ty_ptr = self.ctx.from_ty_as_ptr(ty);
                            let index_ty = self.ctx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
                            let spirv_index_ty = self.ctx.from_ty(index_ty);
                            let index = self.ctx
                                .builder
                                .constant_u32(spirv_index_ty.word, field.index() as u32);
                            let field_access = self.ctx
                                .builder
                                .in_bounds_access_chain(spirv_ty_ptr.word, None, var.word, &[index])
                                .expect("access chain");

                            let spirv_var = SpirvVar::new(field_access, false, ty);
                            SpirvOperand::Consume(spirv_var)
                        }
                        rest => unimplemented!("{:?}", rest),
                    },
                    ref rest => unimplemented!("{:?}", rest),
                },
                ref rest => unimplemented!("{:?}", rest),
            },
            &mir::Operand::Constant(ref constant) => match constant.literal {
                mir::Literal::Value { ref value } => {
                    use rustc::middle::const_val::ConstVal;
                    let expr = match value {
                        &ConstVal::Float(f) => {
                            use syntax::ast::FloatTy;
                            match f.ty {
                                FloatTy::F32 => {
                                    let val: f32 = unsafe { ::std::mem::transmute(f.bits as u32) };
                                    let expr = SpirvExpr(
                                        self.ctx.builder.constant_f32(spirv_ty.word, val),
                                    );
                                    expr
                                }
                                _ => panic!("f64 not supported"),
                            }
                        }
                        ref rest => unimplemented!("{:?}", rest),
                    };
                    //                    if let Some(const_var) = self.constants.get(constant) {
                    //                        //                        let load_const = self.ctx
                    //                        //                            .builder
                    //                        //                            .load(spirv_ty.word, None, expr.0, None, &[])
                    //                        //                            .expect("load");
                    //                        self.ctx
                    //                            .builder
                    //                            .store(const_var.0, expr.0, None, &[])
                    //                            .expect("store");
                    //                        return SpirvOperand::Consume(*const_var);
                    //                    }
                    SpirvOperand::ConstVal(expr)
                }
                ref rest => unimplemented!("{:?}", rest),
            },
            ref rest => unimplemented!("{:?}", rest),
        }
    }
    pub fn get_table(&self) -> &'a ty::TypeckTables<'tcx> {
        self.current_table.last().expect("no table yet")
    }
    pub fn new(
        ctx: &'b mut SpirvCtx<'a, 'tcx>,
        tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
        mir: &'a mir::Mir<'tcx>,
        def_id: rustc::hir::def_id::DefId,
    ) -> Self {
        let mut visitor = RlslVisitor {
            map: &tcx.hir,
            current_table: Vec::new(),
            ctx,
            mir,
            entry: None,
            def_id,
            merge_collector: None,
            constants: HashMap::new(),
            label_blocks: HashMap::new(),
            exprs: HashMap::new(),
            vars: HashMap::new(),
        };
        visitor
    }
}
impl<'b, 'a, 'tcx: 'a> rustc::mir::visit::Visitor<'tcx> for RlslVisitor<'b, 'a, 'tcx> {
    fn visit_basic_block_data(&mut self, block: mir::BasicBlock, data: &mir::BasicBlockData<'tcx>) {
        println!("basic block");

        {
            let spirv_label = self.label_blocks.get(&block).expect("no spirv label");
            let label = self.ctx
                .builder
                .begin_basic_block(Some(spirv_label.0))
                .expect("begin block");
        }

        self.super_basic_block_data(block, data);
    }
    fn visit_statement(
        &mut self,
        block: mir::BasicBlock,
        statement: &mir::Statement<'tcx>,
        location: mir::Location,
    ) {
        self.super_statement(block, statement, location);
    }
    fn visit_mir(&mut self, mir: &mir::Mir<'tcx>) {
        for (block, _) in mir.basic_blocks().iter_enumerated() {
            self.label_blocks
                .insert(block, SpirvLabel(self.ctx.builder.id()));
        }
        let constants = collect_constants(mir);

        let ret_ty_spirv = self.ctx.from_ty(mir.return_ty);
        let args_ty: Vec<_> = mir.args_iter().map(|l| mir.local_decls[l].ty).collect();
        let fn_sig = self.ctx.tcx.mk_fn_sig(
            args_ty.into_iter(),
            mir.return_ty,
            false,
            hir::Unsafety::Normal,
            syntax::abi::Abi::Rust,
        );
        let fn_ty = self.ctx.tcx.mk_fn_ptr(ty::Binder(fn_sig));
        let fn_ty_spirv = self.ctx.from_ty(fn_ty);

        let def_id = self.def_id;
        let forward_fn = self.ctx
            .forward_fns
            .get(&def_id)
            .map(|f| f.0)
            .expect("forward");
        let spirv_function = self.ctx
            .builder
            .begin_function(
                ret_ty_spirv.word,
                Some(forward_fn),
                spirv::FunctionControl::empty(),
                fn_ty_spirv.word,
            )
            .expect("begin fn");
        //self.ctx.builder.begin_basic_block(None).expect("block");
        for local_arg in mir.args_iter() {
            let local_decl = &mir.local_decls[local_arg];
            let spirv_arg_ty = self.ctx.from_ty(local_decl.ty);
            let param = local_decl.ty.as_opt_param_ty();
            let spirv_param = self.ctx
                .builder
                .function_parameter(spirv_arg_ty.word)
                .expect("fn param");
            self.vars
                .insert(local_arg, SpirvVar::new(spirv_param, true, local_decl.ty));
        }
        self.ctx.builder.begin_basic_block(None).expect("block");
        for local_var in mir.vars_and_temps_iter() {
            let local_decl = &mir.local_decls[local_var];
            let lvalue_ty = local_decl.ty;
            let is_ptr = lvalue_ty.is_unsafe_ptr() || lvalue_ty.is_mutable_pointer() ||
                lvalue_ty.is_region_ptr();
            let spirv_var_ty = if is_ptr {
                //self.ctx.from_ty(local_decl.ty)
                self.ctx.from_ty_as_ptr(local_decl.ty)
            } else {
                self.ctx.from_ty_as_ptr(local_decl.ty)
            };
            let param = local_decl.ty.as_opt_param_ty();
            println!("param = {:?}", param);
            let spirv_var = self.ctx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.vars
                .insert(local_var, SpirvVar::new(spirv_var, false, local_decl.ty));
        }
        {
            use rustc_data_structures::indexed_vec::Idx;
            let local = mir::Local::new(0);
            let local_decl = &mir.local_decls[local];
            let spirv_var_ty = self.ctx.from_ty_as_ptr(local_decl.ty);
            let spirv_var = self.ctx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.vars
                .insert(local, SpirvVar::new(spirv_var, false, local_decl.ty));
            let spirv_label = self.label_blocks
                .get(&mir::BasicBlock::new(0))
                .expect("No first label");
            self.ctx.builder.branch(spirv_label.0).expect("branch");
        }
        self.super_mir(mir);
        self.ctx.builder.end_function().expect("end fn");
        if self.entry.is_some() {
            self.ctx
                .builder
                .entry_point(spirv::ExecutionModel::Vertex, spirv_function, "main", &[])
        }
    }
    fn visit_assign(
        &mut self,
        block: mir::BasicBlock,
        lvalue: &mir::Lvalue<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: mir::Location,
    ) {
        self.super_assign(block, lvalue, rvalue, location);
        let ty = rvalue.ty(&self.mir.local_decls, self.ctx.tcx);
        if let ty::TypeVariants::TyTuple(ref slice, _) = ty.sty {
            if slice.len() == 0 {
                return;
            }
        }
        let spirv_ty = self.ctx.from_ty(ty);
        let lvalue_ty = lvalue
            .ty(&self.mir.local_decls, self.ctx.tcx)
            .to_ty(self.ctx.tcx);
        let expr = match rvalue {
            &mir::Rvalue::BinaryOp(op, ref l, ref r) => {
                // TODO: Different types
                let l_load = self.load_operand(l).load_raw(&mut self.ctx, spirv_ty);
                let r_load = self.load_operand(r).load_raw(&mut self.ctx, spirv_ty);
                // TODO: Impl ops
                match op {
                    mir::BinOp::Add => {
                        let add = self.ctx
                            .builder
                            .fadd(spirv_ty.word, None, l_load, r_load)
                            .expect("fadd");
                        SpirvExpr(add)
                    }
                    mir::BinOp::Gt => {
                        let gt = self.ctx
                            .builder
                            .ugreater_than(spirv_ty.word, None, l_load, r_load)
                            .expect("g");
                        SpirvExpr(gt)
                    }
                    rest => unimplemented!("{:?}", rest),
                }
            }
            &mir::Rvalue::Use(ref operand) => {
                let is_ptr = lvalue_ty.is_unsafe_ptr() || lvalue_ty.is_mutable_pointer() ||
                    lvalue_ty.is_region_ptr();

                let operand = self.load_operand(operand);
                if is_ptr && operand.is_param() {
                    SpirvExpr(operand.expect_var().word)
                } else {
                    let load = operand.load_raw(&mut self.ctx, spirv_ty);
                    let expr = SpirvExpr(load);
                    expr
                }
            }

            //            &mir::Rvalue::NullaryOp(..) => {}
            //            &mir::Rvalue::CheckedBinaryOp(..) => {}
            //            &mir::Rvalue::Discriminant(..) => {}
            &mir::Rvalue::Aggregate(ref kind, ref operand) => {
                // TODO Aggregate
                println!("kind = {:?} {:?}", kind, operand);
                unimplemented!()
            }
            &mir::Rvalue::Ref(_, _, ref lvalue) => {
                let is_ptr = lvalue_ty.is_unsafe_ptr() || lvalue_ty.is_mutable_pointer() ||
                    lvalue_ty.is_region_ptr();
                match lvalue {
                    &mir::Lvalue::Local(local) => {
                        let var = self.vars.get(&local).expect("no local");
                        if is_ptr {
                            //self.ctx.builder.load(self.ctx.from_ty(var.ty))
                            // TODO
                            SpirvExpr(var.word)
                        } else {
                            SpirvExpr(var.word)
                        }
                    }
                    rest => unimplemented!("{:?}", rest),
                }
            }

            rest => unimplemented!("{:?}", rest),
        };

        match lvalue {
            &mir::Lvalue::Local(local) => {
                let var = self.vars.get(&local).expect("local");
                self.ctx
                    .builder
                    .store(var.word, expr.0, None, &[])
                    .expect("store");
            }
            &mir::Lvalue::Projection(ref proj) => match &proj.elem {
                &mir::ProjectionElem::Field(field, ty) => match &proj.base {
                    &mir::Lvalue::Local(local) => {
                        let var = self.vars.get(&local).expect("local");
                        let spirv_ty_ptr = self.ctx.from_ty_as_ptr(ty);
                        let index_ty = self.ctx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
                        let spirv_index_ty = self.ctx.from_ty(index_ty);
                        let index = self.ctx
                            .builder
                            .constant_u32(spirv_index_ty.word, field.index() as u32);
                        let field_access = self.ctx
                            .builder
                            .in_bounds_access_chain(spirv_ty_ptr.word, None, var.word, &[index])
                            .expect("access chain");
                        let spirv_ptr = self.ctx.from_ty(ty);
                        //let load = self.ctx.builder.load(spirv_ptr.word, None, field_access, None, &[]).expect("load");
                        self.ctx
                            .builder
                            .store(field_access, expr.0, None, &[])
                            .expect("store");
                    }
                    rest => unimplemented!("{:?}", rest),
                },
                rest => unimplemented!("{:?}", rest),
            },
            rest => unimplemented!("{:?}", rest),
        };
    }

    fn visit_lvalue(
        &mut self,
        lvalue: &mir::Lvalue<'tcx>,
        context: mir::visit::LvalueContext<'tcx>,
        location: mir::Location,
    ) {
        // TODO KERNEL NOT SUPPORTED :(
        //        if let &mir::Lvalue::Local(local) = lvalue{
        //            if context.is_storage_live_marker(){
        //                let var = self.vars.get(&local).expect("no local");
        //                self.ctx.builder.lifetime_start(var.word, 0);
        //            }
        //            if context.is_storage_dead_marker(){
        //                let var = self.vars.get(&local).expect("no local");
        //                self.ctx.builder.lifetime_stop(var.word, 0);
        //            }
        //        }
        self.super_lvalue(lvalue, context, location);
    }
    fn visit_terminator_kind(
        &mut self,
        block: mir::BasicBlock,
        kind: &mir::TerminatorKind<'tcx>,
        location: mir::Location,
    ) {
        self.super_terminator_kind(block, kind, location);
        let mir = self.mir;
        match kind {
            &mir::TerminatorKind::Return => {
                match mir.return_ty.sty {
                    ty::TypeVariants::TyTuple(ref slice, _) if slice.len() == 0 => {
                        self.ctx.builder.ret().expect("ret");
                    }
                    _ => {
                        use rustc_data_structures::indexed_vec::Idx;
                        let spirv_ty = { self.ctx.from_ty(mir.return_ty) };
                        let var = self.vars.get(&mir::Local::new(0)).unwrap();
                        let load = self.ctx
                            .builder
                            .load(spirv_ty.word, None, var.word, None, &[])
                            .expect("load");
                        self.ctx.builder.ret_value(load).expect("ret value");
                    }
                    _ => (),
                };
            }
            &mir::TerminatorKind::Goto { target } => {
                let label = self.label_blocks.get(&target).expect("no goto label");
                self.ctx.builder.branch(label.0);
            }
            &mir::TerminatorKind::SwitchInt {
                ref discr,
                switch_ty,
                ref targets,
                ..
            } => {
                use rustc_data_structures::control_flow_graph::ControlFlowGraph;
                let mir = self.mir;
                println!("targets = {:?}", targets);
                let spirv_operand = { self.load_operand(discr).into_raw_word() };
                let collector = self.merge_collector.as_ref().unwrap();
                let merge_block = collector
                    .get(&location)
                    .expect("Unable to find a merge block");
                let merge_block_label = self.label_blocks.get(merge_block).expect("no label");
                println!("merge_block = {:?}", merge_block);
                println!("switch_ty = {:?}", switch_ty);
                let spirv_ty = self.ctx.from_ty(switch_ty);
                let b = self.ctx.builder.constant_true(spirv_ty.word);
                self.ctx
                    .builder
                    .selection_merge(merge_block_label.0, spirv::SelectionControl::empty());
                let true_label = self.label_blocks.get(&targets[0]).expect("true label");
                let false_label = self.label_blocks.get(&targets[1]).expect("false label");
                self.ctx.builder.branch_conditional(
                    spirv_operand,
                    true_label.0,
                    false_label.0,
                    &[],
                );
            }
            &mir::TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                ..
            } => {
                let local_decls = &self.mir.local_decls;
                match func {
                    &mir::Operand::Constant(ref constant) => {
                        let ret_ty_binder = constant.ty.fn_sig(self.ctx.tcx).output();
                        let ret_ty = self.ctx
                            .tcx
                            .erase_late_bound_regions_and_normalize(&ret_ty_binder);
                        let spirv_ty = self.ctx.from_ty(ret_ty);
                        if let mir::Literal::Value { ref value } = constant.literal {
                            use rustc::middle::const_val::ConstVal;
                            if let &ConstVal::Function(def_id, ref subst) = value {
                                let resolve_fn_id = resolve_fn_call(self.ctx.tcx, def_id, subst);
                                let spirv_fn = self.ctx
                                    .forward_fns
                                    .get(&resolve_fn_id)
                                    .map(|v| *v)
                                    .expect("forward fn call");
                                let arg_operand_loads: Vec<_> = args.iter()
                                    .map(|arg| {
                                        let operand = self.load_operand(arg);
//                                        match operand {
//                                            SpirvOperand::Consume(var) => var.0,
//                                            SpirvOperand::ConstVal(constant) => panic!("no const"),
//                                        }
                                        let arg_ty = arg.ty(local_decls, self.ctx.tcx);
                                        let arg_ty_spirv = self.ctx.from_ty(arg_ty);
                                        operand.load_raw(&mut self.ctx, arg_ty_spirv)
                                    })
                                    .collect();
                                let spirv_fn_call = self.ctx
                                    .builder
                                    .function_call(
                                        spirv_ty.word,
                                        None,
                                        spirv_fn.0,
                                        &arg_operand_loads,
                                    )
                                    .expect("fn call");
                                if let &Some(ref dest) = destination {
                                    let &(ref lvalue, _) = dest;
                                    match lvalue {
                                        &mir::Lvalue::Local(local) => {
                                            let var = self.vars.get(&local).expect("local");
                                            self.ctx
                                                .builder
                                                .store(var.word, spirv_fn_call, None, &[])
                                                .expect("store");
                                        }
                                        rest => unimplemented!("{:?}", rest),
                                    };
                                }
                            }
                        }
                    }
                    _ => (),
                }
                let destination = destination.as_ref().expect("Fn call is diverging");
                let &(_, target_block) = destination;
                let target_label = self.label_blocks.get(&target_block).expect("no label");
                self.ctx.builder.branch(target_label.0).expect("label");
            }
            rest => unimplemented!("{:?}", rest),
        };
    }
    fn visit_local_decl(&mut self, local_decl: &mir::LocalDecl<'tcx>) {
        self.super_local_decl(local_decl);
    }
    //    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: mir::Location) {
    //        self.super_rvalue(rvalue, location);
    //        //println!("location = {:?}", location);
    //        let local_decls = &self.mir.local_decls;
    //        let ty = rvalue.ty(local_decls, self.ctx.tcx);
    //        let spirv_ty = self.ctx.from_ty(ty);
    //        match rvalue {
    //            &mir::Rvalue::BinaryOp(op, ref l, ref r) => {
    //                // TODO: Different types
    //                let l_load = self.load_operand(l).load_raw(&mut self.ctx, spirv_ty);
    //                let r_load = self.load_operand(r).load_raw(&mut self.ctx, spirv_ty);
    //                // TODO: Impl ops
    //                match op {
    //                    mir::BinOp::Add => {
    //                        let add = self.ctx
    //                            .builder
    //                            .fadd(spirv_ty.word, None, l_load, r_load)
    //                            .expect("fadd");
    //                        self.exprs.insert(location, SpirvExpr(add));
    //                    }
    //                    mir::BinOp::Gt => {
    //                        let gt = self.ctx
    //                            .builder
    //                            .ugreater_than(spirv_ty.word, None, l_load, r_load)
    //                            .expect("g");
    //                        self.exprs.insert(location, SpirvExpr(gt));
    //                    }
    //                    rest => unimplemented!("{:?}", rest),
    //                }
    //            }
    //            &mir::Rvalue::Use(ref operand) => {
    //                let load = self.load_operand(operand).load_raw(&mut self.ctx, spirv_ty);
    //                let expr = SpirvExpr(load);
    //                self.exprs.insert(location, expr);
    //            }
    //            //            &mir::Rvalue::NullaryOp(..) => {}
    //            //            &mir::Rvalue::CheckedBinaryOp(..) => {}
    //            //            &mir::Rvalue::Discriminant(..) => {}
    //            &mir::Rvalue::Aggregate(ref kind, ref operand) => {
    //                // TODO Aggregate
    //                println!("kind = {:?} {:?}", kind, operand);
    //            }
    //            &mir::Rvalue::Ref(_, _, ref lvalue) => {
    //                match lvalue{
    //                    &mir::Lvalue::Local(local) => {
    //                        let var = self.vars.get(&local).expect("no local");
    //                        self.exprs.insert(location, SpirvExpr(var.word));
    //                    }
    //                    rest => unimplemented!("{:?}", rest),
    //                }
    //            }
    //
    //            rest => unimplemented!("{:?}", rest),
    //        }
    //    }
}
