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
use self::hir::def_id::DefId;
pub enum IntrinsicFn {
    Dot,
}

fn intrinsic_fn(attrs: &[syntax::ast::Attribute]) -> Option<IntrinsicFn> {
    attrs
        .iter()
        .filter_map(|attr| {
            extract_spirv_attr(attr, &[], |s| match s {
                "dot" => Some(IntrinsicFn::Dot),
                _ => None,
            })
        })
        .nth(0)
}
pub fn trans_items<'a, 'tcx>(
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    analysis: ty::CrateAnalysis,
    output_filenames: &'a OutputFilenames,
) -> (SharedCrateContext<'a, 'tcx>, FxHashSet<TransItem<'tcx>>) {
    let ty::CrateAnalysis { reachable, .. } = analysis;
    let check_overflow = tcx.sess.overflow_checks();
    //let link_meta = link::build_link_meta(&incremental_hashes_map);
    let exported_symbol_node_ids = find_exported_symbols(tcx, &reachable);
    let shared_ccx = SharedCrateContext::new(tcx, check_overflow, output_filenames);
    let exported_symbols = ExportedSymbols::compute(tcx, &exported_symbol_node_ids);
    let (items, _) = rustc_trans::collector::collect_crate_translation_items(
        &shared_ccx,
        &exported_symbols,
        rustc_trans::collector::TransItemCollectionMode::Eager,
    );
    (shared_ccx, items)
}

pub struct SpirvCtx<'a, 'tcx: 'a> {
    pub tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    pub ccx: SharedCrateContext<'a, 'tcx>,
    pub builder: Builder,
    pub ty_cache: HashMap<ty::Ty<'tcx>, SpirvTy>,
    pub forward_fns: HashMap<hir::def_id::DefId, SpirvFn>,
    pub debug_symbols: bool,
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

use syntax_pos::symbol::InternedString;
impl<'a, 'tcx> SpirvCtx<'a, 'tcx> {
    fn attrs_from_def_id(&self, def_id: DefId) -> Option<&[syntax::ast::Attribute]> {
        let node_id = self.tcx.hir.as_local_node_id(def_id);
        let node = node_id.and_then(|id| self.tcx.hir.find(id));
        let item = node.and_then(|node| match node {
            hir::map::Node::NodeItem(item) => Some(item),
            _ => None,
        });
        item.map(|item| &*item.attrs)
    }
    pub fn name_from_def_id(&mut self, def_id: hir::def_id::DefId, id: spirv::Word) {
        let name = self.tcx
            .hir
            .as_local_node_id(def_id)
            .map(|node_id| self.tcx.hir.name(node_id).as_str());
        if let Some(name) = name {
            if self.debug_symbols {
                self.builder.name(id, name.as_ref());
            }
        }
    }
    pub fn name_from_str(&mut self, name: &str, id: spirv::Word) {
        if self.debug_symbols {
            self.builder.name(id, name);
        }
    }
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
    pub fn new(
        tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
        ccx: SharedCrateContext<'a, 'tcx>,
    ) -> SpirvCtx<'a, 'tcx> {
        let mut builder = Builder::new();
        builder.capability(spirv::Capability::Shader);
        builder.ext_inst_import("GLSL.std.450");
        builder.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        SpirvCtx {
            debug_symbols: true,
            builder,
            ty_cache: HashMap::new(),
            forward_fns: HashMap::new(),
            tcx,
            ccx,
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
    pub mtx: MirContext<'b, 'a, 'tcx>,
    pub entry: Option<IntrinsicEntry>,
    pub merge_collector: Option<MergeCollector>,
    pub constants: HashMap<mir::Constant<'tcx>, SpirvVar<'tcx>>,
    pub label_blocks: HashMap<mir::BasicBlock, SpirvLabel>,
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
pub struct MirContext<'b, 'a: 'b, 'tcx: 'a> {
    def_id: hir::def_id::DefId,
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a mir::Mir<'tcx>,
    substs: &'tcx ty::subst::Substs<'tcx>,
    pub stx: &'b mut SpirvCtx<'a, 'tcx>,
}
impl<'b, 'a, 'tcx> MirContext<'b, 'a, 'tcx> {
    pub fn monomorphize<T>(&self, value: &T) -> T
    where
        T: rustc::infer::TransNormalize<'tcx>,
    {
        self.tcx.trans_apply_param_substs(self.substs, value)
    }
    pub fn from_ty(&mut self, ty: rustc::ty::Ty<'tcx>) -> SpirvTy {
        use rustc::ty::TypeVariants;
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        use std::hash::Hasher;
        let ty = self.monomorphize(&ty);
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
        if let Some(ty) = self.stx.ty_cache.get(ty) {
            return *ty;
        }
        let spirv_type: SpirvTy = match ty.sty {
            TypeVariants::TyBool => self.stx.builder.type_bool().into(),
            TypeVariants::TyUint(uint_ty) => self.stx
                .builder
                .type_int(uint_ty.bit_width().unwrap() as u32, 0)
                .into(),
            TypeVariants::TyFloat(f_ty) => {
                use syntax::ast::FloatTy;
                match f_ty {
                    FloatTy::F32 => self.stx.builder.type_float(32).into(),
                    FloatTy::F64 => self.stx.builder.type_float(64).into(),
                }
            }
            TypeVariants::TyTuple(slice, _) if slice.len() == 0 => {
                self.stx.builder.type_void().into()
            }
            TypeVariants::TyFnPtr(sig) => {
                //let ret_ty = self.from_ty(sig.output().skip_binder());
                let ty = self.tcx
                    .erase_late_bound_regions_and_normalize(&sig.output());
                let ret_ty = self.from_ty(ty);
                // TODO: Skip binder?
                let input_ty: Vec<_> = sig.inputs()
                    .skip_binder()
                    .iter()
                    .map(|ty| self.from_ty(ty).word)
                    .collect();
                self.stx
                    .builder
                    .type_function(ret_ty.word, &input_ty)
                    .into()
            }
            TypeVariants::TyRawPtr(type_and_mut) => {
                let inner = self.from_ty(type_and_mut.ty);
                self.stx
                    .builder
                    .type_pointer(None, spirv::StorageClass::Function, inner.word)
                    .into()
            }
            TypeVariants::TyParam(ref param) => panic!("TyParam should have been monomorphized"),
            TypeVariants::TyAdt(adt, substs) => match adt.adt_kind() {
                ty::AdtKind::Struct => {
                    let attrs = self.tcx.get_attrs(adt.did);
                    let intrinsic = attrs
                        .iter()
                        .filter_map(|attr| {
                            extract_spirv_attr(attr, &[], |s| match s {
                                "Vec2" => Some(IntrinsicType::Vec(2)),
                                _ => None,
                            })
                        })
                        .nth(0);

                    if let Some(intrinsic) = intrinsic {
                        let intrinsic_spirv = match intrinsic {
                            IntrinsicType::Vec(dim) => {
                                let field_ty = adt.all_fields()
                                    .nth(0)
                                    .map(|f| f.ty(self.tcx, substs))
                                    .expect("no field");
                                let spirv_ty = self.from_ty(field_ty);
                                self.stx
                                    .builder
                                    .type_vector(spirv_ty.word, dim as u32)
                                    .into()
                            }
                            ref r => unimplemented!("{:?}", r),
                        };
                        intrinsic_spirv
                    } else {
                        let field_ty_spirv: Vec<_> = adt.all_fields()
                            .map(|f| {
                                let ty = f.ty(self.tcx, substs);
                                self.from_ty(ty).word
                            })
                            .collect();
                        let spirv_struct = self.stx.builder.type_struct(&field_ty_spirv);
                        self.stx.name_from_def_id(adt.did, spirv_struct);
                        spirv_struct.into()
                    }
                }
                ref r => unimplemented!("{:?}", r),
            },
            ref r => unimplemented!("{:?}", r),
        };
        self.stx.ty_cache.insert(ty, spirv_type);
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
}


pub struct CollectCrateItems<'a, 'tcx: 'a> {
    mir: &'a mir::Mir<'tcx>,
    ccx: &'a SharedCrateContext<'a, 'tcx>,
    items: Vec<TransItem<'tcx>>,
}
pub fn collect_crate_items<'a, 'b, 'tcx>(
    mir: &'a mir::Mir<'tcx>,
    ccx: &'a SharedCrateContext<'a, 'tcx>,
) -> Vec<TransItem<'tcx>> {
    let mut collector = CollectCrateItems {
        mir,
        ccx,
        items: Vec::new(),
    };
    collector.visit_mir(mir);
    collector.items
}
impl<'a, 'tcx> rustc::mir::visit::Visitor<'tcx> for CollectCrateItems<'a, 'tcx> {
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
                    if let &ConstVal::Function(def_id, ref substs) = value {
                        //let mono_substs = self.mtx.monomorphize(substs);
                        let instance = rustc_trans::resolve(&self.ccx, def_id, substs);
                        self.items.push(TransItem::Fn(instance));
                        //let mir = self.ccx.tcx().maybe_optimized_mir(resolve_def_id);
                        //                        self.contexts.push(MirContext {
                        //                            tcx: self.tcx,
                        //                            mir: mir.expect("mir"),
                        //                            substs,
                        //                        });
                    }
                }
            }
        }
    }
}

pub fn trans_all_items<'a, 'tcx>(
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    shared_ccx: &'a SharedCrateContext<'a, 'tcx>,
    start_items: &'a FxHashSet<TransItem<'tcx>>,
) -> FxHashSet<TransItem<'tcx>> {
    let mut hash_set = FxHashSet();
    let mut uncollected_items: Vec<Vec<TransItem<'tcx>>> = Vec::new();
    uncollected_items.push(start_items.iter().cloned().collect());
    while let Some(items) = uncollected_items.pop() {
        for item in &items {
            if let &TransItem::Fn(ref instance) = item {
                let mir = tcx.maybe_optimized_mir(instance.def_id());
                if let Some(mir) = mir {
                    let new_items = collect_crate_items(mir, shared_ccx);
                    if !new_items.is_empty() {
                        uncollected_items.push(new_items)
                    }
                    hash_set.insert(*item);
                }
            }
        }
    }
    hash_set
}

pub fn trans_spirv<'a, 'tcx>(
    tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    shared_ccx: SharedCrateContext<'a, 'tcx>,
    items: &'a FxHashSet<TransItem<'tcx>>,
) {
    use rustc::mir::visit::Visitor;
    let mut ctx = SpirvCtx::new(tcx, shared_ccx);
    for item in items {
        if let &TransItem::Fn(ref instance) = item {
            println!("instace = {:?}", instance.def_id());
            ctx.forward_fns
                .insert(instance.def_id(), SpirvFn(ctx.builder.id()));
            let has_mir = tcx.maybe_optimized_mir(instance.def_id()).is_some();
            println!("has_mir = {:?}", has_mir);
        }
    }
    println!("ctx.forward_fns = {:?}", ctx.forward_fns);
    for item in items {
        if let &TransItem::Fn(ref instance) = item {
            //let resolved_def_id = resolve_fn_call(tcx, )
            let mir = tcx.maybe_optimized_mir(instance.def_id());
            let map: &Map = &tcx.hir;
            if let Some(ref mir) = mir {
                let mir_ctx = MirContext {
                    def_id: instance.def_id(),
                    tcx,
                    substs: instance.substs,
                    mir,
                    stx: &mut ctx,
                };
                //println!("mir = {:#?}", mir);
                let mut visitor = RlslVisitor::new(tcx, mir_ctx);
                let node = map.as_local_node_id(instance.def_id())
                    .and_then(|node_id| map.find(node_id));
                if let Some(rustc::hir::map::Node::NodeItem(ref item)) = node {
                    let intrinsic = item.attrs
                        .iter()
                        .filter_map(|attr| {
                            extract_spirv_attr(attr, &[], |s| match s {
                                "vertex" => Some(IntrinsicEntry::Vertex),
                                "fragment" => Some(IntrinsicEntry::Fragment),
                                _ => None,
                            })
                        })
                        .nth(0);
                    //println!("intrinsic = {:?}", intrinsic);
                    visitor.entry = intrinsic;
                }
                visitor.merge_collector = Some(merge_collector(mir));
                visitor.visit_mir(mir);
            }
        }
    }
    ctx.build_module();
}
#[derive(Debug)]
struct AccessChain<'a, 'tcx: 'a> {
    base: &'a mir::Lvalue<'tcx>,
    indices: Vec<usize>,
}
impl<'b, 'a, 'tcx: 'a> RlslVisitor<'b, 'a, 'tcx> {
    fn access_chain<'r>(&mut self, lvalue: &'r mir::Lvalue<'tcx>) -> AccessChain<'r, 'tcx> {
        let mut indices = Vec::new();
        let lvalue = self.access_chain_indices(lvalue, &mut indices);
        AccessChain {
            base: lvalue,
            indices,
        }
    }
    fn access_chain_indices<'r>(
        &self,
        lvalue: &'r mir::Lvalue<'tcx>,
        indices: &mut Vec<usize>,
    ) -> &'r mir::Lvalue<'tcx> {
        if let &mir::Lvalue::Projection(ref proj) = lvalue {
            if let mir::ProjectionElem::Field(field, _) = proj.elem {
                let inner = self.access_chain_indices(&proj.base, indices);
                indices.push(field.index());
                return inner;
            }
        }
        lvalue
    }
    fn trans_lvalue(&mut self, lvalue: &mir::Lvalue<'tcx>) -> SpirvOperand<'tcx> {
        let local_decls = &self.mtx.mir.local_decls;
        match lvalue {
            &mir::Lvalue::Local(local) => {
                let local_decl = &local_decls[local];
                let spirv_ty = self.mtx.from_ty(local_decl.ty);
                let spirv_var = self.vars.get(&local).expect("local");
                SpirvOperand::Consume(*spirv_var)
            }
            &mir::Lvalue::Projection(ref proj) => match &proj.elem {
                &mir::ProjectionElem::Field(field, ty) => {
                    println!("{:?} field = {:?}", lvalue, field);
                    match &proj.base {
                        &mir::Lvalue::Local(local) => {
                            let var = self.vars.get(&local).expect("local");
                            let spirv_ty_ptr = self.mtx.from_ty_as_ptr(ty);
                            let index_ty = self.mtx.stx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
                            let spirv_index_ty = self.mtx.from_ty(index_ty);
                            let index = self.mtx
                                .stx
                                .builder
                                .constant_u32(spirv_index_ty.word, field.index() as u32);
                            let field_access = self.mtx
                                .stx
                                .builder
                                .in_bounds_access_chain(spirv_ty_ptr.word, None, var.word, &[index])
                                .expect("access chain");

                            let spirv_var = SpirvVar::new(field_access, false, ty);
                            SpirvOperand::Consume(spirv_var)
                        }
                        &mir::Lvalue::Projection(_) => self.trans_lvalue(&proj.base),
                        rest => unimplemented!("{:?}", rest),
                    }
                }
                &mir::ProjectionElem::Deref => match &proj.base {
                    &mir::Lvalue::Local(local) => {
                        let var = self.vars.get(&local).expect("local");
                        SpirvOperand::Consume(*var)
                    }
                    ref rest => unimplemented!("{:?}", rest),
                },
                ref rest => unimplemented!("{:?}", rest),
            },
            ref rest => unimplemented!("{:?}", rest),
        }
    }
    pub fn load_operand<'r>(&mut self, operand: &'r mir::Operand<'tcx>) -> SpirvOperand<'tcx> {
        let mir = self.mtx.mir;
        let local_decls = &mir.local_decls;
        let ty = operand.ty(local_decls, self.mtx.stx.tcx);
        let spirv_ty = self.mtx.from_ty(ty);
        match operand {
            &mir::Operand::Consume(ref lvalue) => {
                let access_chain = self.access_chain(lvalue);
                println!("access_chain = {:?}", access_chain);
                let spirv_var = match access_chain.base {
                    &mir::Lvalue::Local(local) => *self.vars.get(&local).expect("Local"),
                    _ => panic!("Should be local"),
                };
                if access_chain.indices.is_empty() {
                    SpirvOperand::Consume(spirv_var)
                } else {
                    let spirv_ty_ptr = self.mtx.from_ty_as_ptr(ty);
                    let index_ty = self.mtx.stx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
                    let spirv_index_ty = self.mtx.from_ty(index_ty);
                    let indices: Vec<_> = access_chain
                        .indices
                        .iter()
                        .map(|&i| {
                            self.mtx
                                .stx
                                .builder
                                .constant_u32(spirv_index_ty.word, i as u32)
                        })
                        .collect();
                    let access = self.mtx
                        .stx
                        .builder
                        .access_chain(spirv_ty_ptr.word, None, spirv_var.word, &indices)
                        .expect("access_chain");
                    let load = self.mtx
                        .stx
                        .builder
                        .load(spirv_ty.word, None, access, None, &[])
                        .expect("load");

                    SpirvOperand::ConstVal(SpirvExpr(load))
                }
                //                println!("access_chain = {:?}", access_chain);
                //                self.trans_lvalue(lvalue)
            }
            //            &mir::Operand::Consume(ref lvalue) => match lvalue {
            //                &mir::Lvalue::Local(local) => {
            //                    let local_decl = &local_decls[local];
            //                    let spirv_ty = self.mtx.from_ty(local_decl.ty);
            //                    let spirv_var = self.vars.get(&local).expect("local");
            //                    SpirvOperand::Consume(*spirv_var)
            //                }
            //                &mir::Lvalue::Projection(ref proj) => match &proj.elem {
            //                    &mir::ProjectionElem::Field(field, ty) => match &proj.base {
            //                        &mir::Lvalue::Local(local) => {
            //                            let var = self.vars.get(&local).expect("local");
            //                            let spirv_ty_ptr = self.mtx.from_ty_as_ptr(ty);
            //                            let index_ty = self.mtx.stx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
            //                            let spirv_index_ty = self.mtx.from_ty(index_ty);
            //                            let index = self.mtx
            //                                .stx
            //                                .builder
            //                                .constant_u32(spirv_index_ty.word, field.index() as u32);
            //                            let field_access = self.mtx
            //                                .stx
            //                                .builder
            //                                .in_bounds_access_chain(spirv_ty_ptr.word, None, var.word, &[index])
            //                                .expect("access chain");
            //
            //                            let spirv_var = SpirvVar::new(field_access, false, ty);
            //                            SpirvOperand::Consume(spirv_var)
            //                        }
            //                        rest => unimplemented!("{:?}", rest),
            //                    },
            //                    ref rest => unimplemented!("{:?}", rest),
            //                },
            //                ref rest => unimplemented!("{:?}", rest),
            //            },
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
                                        self.mtx.stx.builder.constant_f32(spirv_ty.word, val),
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
    pub fn new(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>, mir_ctx: MirContext<'b, 'a, 'tcx>) -> Self {
        let mut visitor = RlslVisitor {
            map: &tcx.hir,
            current_table: Vec::new(),
            mtx: mir_ctx,
            entry: None,
            merge_collector: None,
            constants: HashMap::new(),
            label_blocks: HashMap::new(),
            exprs: HashMap::new(),
            vars: HashMap::new(),
        };
        visitor
    }
}
fn is_ptr(ty: ty::Ty) -> bool {
    ty.is_unsafe_ptr() || ty.is_mutable_pointer() || ty.is_region_ptr()
}

impl<'b, 'a, 'tcx: 'a> rustc::mir::visit::Visitor<'tcx> for RlslVisitor<'b, 'a, 'tcx> {
    fn visit_basic_block_data(&mut self, block: mir::BasicBlock, data: &mir::BasicBlockData<'tcx>) {
        println!("basic block");

        {
            let spirv_label = self.label_blocks.get(&block).expect("no spirv label");
            let label = self.mtx
                .stx
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
        //        let inrinsic_fn = self.mtx
        //            .stx
        //            .attrs_from_def_id(self.mtx.def_id)
        //            .and_then(intrinsic_fn);
        //        match intrinsic_fn {
        //            IntrinsicFn::Dot => {
        //                let spirv_ret_ty = self.mtx.from_ty(mir.return_ty);
        //                let first_arg_local = mir.args_iter().nth(0).expect("arg");
        //                let local_decl = &mir.local_decls[first_arg_local];
        //                let spirv_arg_ty = self.mtx.from_ty(local_decl.ty);
        //
        //                //self.mtx.stx.builder.dot
        //            }
        //        }
        //let attr = self.mtx.tcx.get_attrs(self.mtx.def_id);

        //println!("attr = {:?}", attr);
        for (block, _) in mir.basic_blocks().iter_enumerated() {
            self.label_blocks
                .insert(block, SpirvLabel(self.mtx.stx.builder.id()));
        }
        let constants = collect_constants(mir);
        assert!(
            !is_ptr(mir.return_ty),
            "Functions are not allowed to return a ptr"
        );
        let ret_ty_spirv = self.mtx.from_ty(mir.return_ty);
        // If a param is not a ptr, we need to turn it into a ptr
        let args_ty: Vec<_> = mir.args_iter().map(|l| mir.local_decls[l].ty).collect();
        //        let args_ty: Vec<_> = mir.args_iter()
        //            .map(|l| {
        //                let ty = mir.local_decls[l].ty;
        //                if is_ptr(ty) {
        //                    ty
        //                } else {
        //                    let t = ty::TypeAndMut {
        //                        ty,
        //                        mutbl: rustc::hir::Mutability::MutMutable,
        //                    };
        //                    self.mtx.tcx.mk_ptr(t)
        //                }
        //            })
        //            .collect();
        let fn_sig = self.mtx.stx.tcx.mk_fn_sig(
            args_ty.into_iter(),
            mir.return_ty,
            false,
            hir::Unsafety::Normal,
            syntax::abi::Abi::Rust,
        );
        let fn_ty = self.mtx.stx.tcx.mk_fn_ptr(ty::Binder(fn_sig));
        println!("fn_ty = {:?}", fn_ty);
        let fn_ty_spirv = self.mtx.from_ty(fn_ty);

        let def_id = self.mtx.def_id;
        let forward_fn = self.mtx
            .stx
            .forward_fns
            .get(&def_id)
            .map(|f| f.0)
            .expect("forward");
        let spirv_function = self.mtx
            .stx
            .builder
            .begin_function(
                ret_ty_spirv.word,
                Some(forward_fn),
                spirv::FunctionControl::empty(),
                fn_ty_spirv.word,
            )
            .expect("begin fn");
        self.mtx.stx.name_from_def_id(def_id, spirv_function);
        let params: Vec<_> = mir.args_iter()
            .map(|local_arg| {
                let local_decl = &mir.local_decls[local_arg];
                let spirv_arg_ty = self.mtx.from_ty(local_decl.ty);
                let param = local_decl.ty.as_opt_param_ty();
                let spirv_param = self.mtx
                    .stx
                    .builder
                    .function_parameter(spirv_arg_ty.word)
                    .expect("fn param");

                if let Some(name) = local_decl.name {
                    self.mtx
                        .stx
                        .name_from_str(name.as_str().as_ref(), spirv_param);
                }
                spirv_param
            })
            .collect();
        self.mtx.stx.builder.begin_basic_block(None).expect("block");
        for (index, param) in params.into_iter().enumerate() {
            let local_arg = mir::Local::new(index + 1);
            let local_decl = &mir.local_decls[local_arg];
            let lvalue_ty = local_decl.ty;
            let is_ptr = lvalue_ty.is_unsafe_ptr() || lvalue_ty.is_mutable_pointer()
                || lvalue_ty.is_region_ptr();
            let spirv_var_ty = self.mtx.from_ty_as_ptr(local_decl.ty);
            let spirv_var = self.mtx.stx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.mtx.stx.builder.store(spirv_var, param, None, &[]);
            self.vars
                .insert(local_arg, SpirvVar::new(spirv_var, false, local_decl.ty));
        }
        for local_var in mir.vars_and_temps_iter() {
            let local_decl = &mir.local_decls[local_var];
            let lvalue_ty = local_decl.ty;
            let is_ptr = lvalue_ty.is_unsafe_ptr() || lvalue_ty.is_mutable_pointer()
                || lvalue_ty.is_region_ptr();
            let spirv_var_ty = self.mtx.from_ty_as_ptr(local_decl.ty);
            let spirv_var = self.mtx.stx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            if let Some(name) = local_decl.name {
                self.mtx
                    .stx
                    .name_from_str(name.as_str().as_ref(), spirv_var);
            }
            self.vars
                .insert(local_var, SpirvVar::new(spirv_var, false, local_decl.ty));
        }
        {
            use rustc_data_structures::indexed_vec::Idx;
            let local = mir::Local::new(0);
            let local_decl = &mir.local_decls[local];
            let spirv_var_ty = self.mtx.from_ty_as_ptr(local_decl.ty);
            let spirv_var = self.mtx.stx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.mtx.stx.name_from_str("retvar", spirv_var);
            self.vars
                .insert(local, SpirvVar::new(spirv_var, false, local_decl.ty));
            let spirv_label = self.label_blocks
                .get(&mir::BasicBlock::new(0))
                .expect("No first label");
            self.mtx.stx.builder.branch(spirv_label.0).expect("branch");
        }
        self.super_mir(mir);
        self.mtx.stx.builder.end_function().expect("end fn");
        if self.entry.is_some() {
            self.mtx.stx.builder.entry_point(
                spirv::ExecutionModel::Vertex,
                spirv_function,
                "main",
                &[],
            );
            self.mtx.stx.builder.execution_mode(
                spirv_function,
                spirv::ExecutionMode::OriginUpperLeft,
                &[],
            );
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
        let ty = rvalue.ty(&self.mtx.mir.local_decls, self.mtx.stx.tcx);
        if let ty::TypeVariants::TyTuple(ref slice, _) = ty.sty {
            if slice.len() == 0 {
                return;
            }
        }
        let spirv_ty = self.mtx.from_ty(ty);
        let lvalue_ty = lvalue
            .ty(&self.mtx.mir.local_decls, self.mtx.stx.tcx)
            .to_ty(self.mtx.stx.tcx);
        let expr = match rvalue {
            &mir::Rvalue::BinaryOp(op, ref l, ref r) => {
                // TODO: Different types
                let l_load = self.load_operand(l).load_raw(&mut self.mtx.stx, spirv_ty);
                let r_load = self.load_operand(r).load_raw(&mut self.mtx.stx, spirv_ty);
                // TODO: Impl ops
                match op {
                    mir::BinOp::Mul => {
                        let add = self.mtx
                            .stx
                            .builder
                            .fmul(spirv_ty.word, None, l_load, r_load)
                            .expect("fmul");
                        SpirvExpr(add)
                    }
                    mir::BinOp::Add => {
                        let add = self.mtx
                            .stx
                            .builder
                            .fadd(spirv_ty.word, None, l_load, r_load)
                            .expect("fadd");
                        SpirvExpr(add)
                    }
                    mir::BinOp::Gt => {
                        let gt = self.mtx
                            .stx
                            .builder
                            .ugreater_than(spirv_ty.word, None, l_load, r_load)
                            .expect("g");
                        SpirvExpr(gt)
                    }
                    rest => unimplemented!("{:?}", rest),
                }
            }
            &mir::Rvalue::Use(ref operand) => {
                let is_ptr = lvalue_ty.is_unsafe_ptr() || lvalue_ty.is_mutable_pointer()
                    || lvalue_ty.is_region_ptr();

                let operand = self.load_operand(operand);
                if is_ptr && operand.is_param() {
                    SpirvExpr(operand.expect_var().word)
                } else {
                    let load = operand.load_raw(&mut self.mtx.stx, spirv_ty);
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
                let is_ptr = lvalue_ty.is_unsafe_ptr() || lvalue_ty.is_mutable_pointer()
                    || lvalue_ty.is_region_ptr();
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
                self.mtx
                    .stx
                    .builder
                    .store(var.word, expr.0, None, &[])
                    .expect("store");
            }
            &mir::Lvalue::Projection(ref proj) => match &proj.elem {
                &mir::ProjectionElem::Field(field, ty) => match &proj.base {
                    &mir::Lvalue::Local(local) => {
                        let var = self.vars.get(&local).expect("local");
                        let spirv_ty_ptr = self.mtx.from_ty_as_ptr(ty);
                        let index_ty = self.mtx.stx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
                        let spirv_index_ty = self.mtx.from_ty(index_ty);
                        let index = self.mtx
                            .stx
                            .builder
                            .constant_u32(spirv_index_ty.word, field.index() as u32);
                        let field_access = self.mtx
                            .stx
                            .builder
                            .in_bounds_access_chain(spirv_ty_ptr.word, None, var.word, &[index])
                            .expect("access chain");
                        let spirv_ptr = self.mtx.from_ty(ty);
                        //let load = self.ctx.builder.load(spirv_ptr.word, None, field_access, None, &[]).expect("load");
                        self.mtx
                            .stx
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
        let mir = self.mtx.mir;
        match kind {
            &mir::TerminatorKind::Return => {
                match mir.return_ty.sty {
                    ty::TypeVariants::TyTuple(ref slice, _) if slice.len() == 0 => {
                        self.mtx.stx.builder.ret().expect("ret");
                    }
                    _ => {
                        use rustc_data_structures::indexed_vec::Idx;
                        let spirv_ty = { self.mtx.from_ty(mir.return_ty) };
                        let var = self.vars.get(&mir::Local::new(0)).unwrap();
                        let load = self.mtx
                            .stx
                            .builder
                            .load(spirv_ty.word, None, var.word, None, &[])
                            .expect("load");
                        self.mtx.stx.builder.ret_value(load).expect("ret value");
                    }
                    _ => (),
                };
            }
            &mir::TerminatorKind::Goto { target } => {
                let label = self.label_blocks.get(&target).expect("no goto label");
                self.mtx.stx.builder.branch(label.0);
            }
            &mir::TerminatorKind::SwitchInt {
                ref discr,
                switch_ty,
                ref targets,
                ..
            } => {
                use rustc_data_structures::control_flow_graph::ControlFlowGraph;
                let mir = self.mtx.mir;
                println!("targets = {:?}", targets);
                let spirv_operand = { self.load_operand(discr).into_raw_word() };
                let collector = self.merge_collector.as_ref().unwrap();
                let merge_block = collector
                    .get(&location)
                    .expect("Unable to find a merge block");
                let merge_block_label = self.label_blocks.get(merge_block).expect("no label");
                println!("merge_block = {:?}", merge_block);
                println!("switch_ty = {:?}", switch_ty);
                let spirv_ty = self.mtx.from_ty(switch_ty);
                let b = self.mtx.stx.builder.constant_true(spirv_ty.word);
                self.mtx
                    .stx
                    .builder
                    .selection_merge(merge_block_label.0, spirv::SelectionControl::empty());
                let true_label = self.label_blocks.get(&targets[0]).expect("true label");
                let false_label = self.label_blocks.get(&targets[1]).expect("false label");
                self.mtx.stx.builder.branch_conditional(
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
                let local_decls = &self.mtx.mir.local_decls;
                match func {
                    &mir::Operand::Constant(ref constant) => {
                        let ret_ty_binder = constant.ty.fn_sig(self.mtx.stx.tcx).output();
                        //                        let ret_ty = self.mtx.stx
                        //                            .tcx
                        //                            .erase_late_bound_regions_and_normalize(&ret_ty_binder);
                        let ret_ty = ret_ty_binder.skip_binder();
                        let spirv_ty = self.mtx.from_ty(ret_ty);
                        if let mir::Literal::Value { ref value } = constant.literal {
                            use rustc::middle::const_val::ConstVal;
                            if let &ConstVal::Function(def_id, ref substs) = value {
                                //let resolve_fn_id = resolve_fn_call(self.mtx.tcx, def_id, substs);
                                println!("fn call def_id = {:?}", def_id);
                                let mono_substs = self.mtx.monomorphize(substs);
                                let resolve_fn_id =
                                    rustc_trans::resolve(&self.mtx.stx.ccx, def_id, &mono_substs)
                                        .def_id();
                                //                                let resolve_fn_id =
                                //                                    rustc_trans::resolve(&self.mtx.stx.ccx, def_id, self.mtx.substs)
                                //                                        .def_id();
                                let spirv_fn = self.mtx
                                    .stx
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
                                        let arg_ty = arg.ty(local_decls, self.mtx.stx.tcx);
                                        let arg_ty_spirv = self.mtx.from_ty(arg_ty);
                                        operand.load_raw(&mut self.mtx.stx, arg_ty_spirv)
                                        //operand.into_raw_word()
                                    })
                                    .collect();
                                let spirv_fn_call = self.mtx
                                    .stx
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
                                            self.mtx
                                                .stx
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
                self.mtx.stx.builder.branch(target_label.0).expect("label");
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
