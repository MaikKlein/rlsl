pub mod terminator;
pub mod rvalue;
pub mod tycache;

use rustc_data_structures::control_flow_graph::ControlFlowGraph;
use rustc::mir::visit::Visitor;
use rustc::hir::map::Map;
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

pub struct SpirvCtx<'tcx> {
    pub builder: Builder,
    pub ty_cache: tycache::SpirvTyCache<'tcx>,
    pub fns: HashMap<hir::def_id::DefId, SpirvFn>,
    pub forward_fns: HashMap<hir::def_id::DefId, SpirvFn>,
    pub vars: HashMap<mir::Local, SpirvVar>,
    pub exprs: HashMap<mir::Location, SpirvExpr>,
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


//impl<'a, 'tcx> rustc::mir::visit::Visitor<'tcx> for MergeCollector<'a, 'tcx> {
//    fn visit_terminator_kind(
//        &mut self,
//        block: mir::BasicBlock,
//        kind: &mir::TerminatorKind<'tcx>,
//        location: mir::Location,
//    ) {
//        self.super_terminator_kind(block, kind, location);
//        match kind {
//            &mir::TerminatorKind::SwitchInt { ref targets, .. } =>  {
//                targets.iter().find(|&&block| {
//                    self.mir.successors(block).filter_map(|block|{
//                        let block_data = &self.mir.basic_blocks()[block];
//                        match block_data.terminator().kind{
//                            mir::TerminatorKind::Goto {target} => {
//                                Some(target)
//                            }
//                            _ => None
//                        }
//                    });
//
//                    true
//                });
//            },
//            _ => (),
//        };
//    }
//}
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
pub enum SpirvOperand {
    Consume(SpirvVar),
    ConstVal(SpirvExpr),
}
impl SpirvOperand {
    pub fn expect_var(self) -> SpirvVar {
        match self {
            SpirvOperand::Consume(var) => var,
            _ => panic!("Expected var"),
        }
    }
    pub fn load_raw(self, ctx: &mut SpirvCtx, ty: SpirvTy) -> spirv::Word {
        match self {
            SpirvOperand::Consume(var) => ctx.builder
                .load(ty.word, None, var.0, None, &[])
                .expect("load"),
            SpirvOperand::ConstVal(expr) => expr.0,
        }
    }
    pub fn into_raw_word(self) -> spirv::Word {
        match self {
            SpirvOperand::Consume(var) => var.0,
            SpirvOperand::ConstVal(expr) => expr.0,
        }
    }
}

impl<'tcx> SpirvCtx<'tcx> {
    pub fn from_ty(&mut self, ty: ty::Ty<'tcx>) -> SpirvTy {
        self.ty_cache.from_ty(&mut self.builder, ty)
    }
    pub fn from_ty_as_ptr<'a, 'gcx>(
        &mut self,
        tcx: ty::TyCtxt<'a, 'gcx, 'tcx>,
        ty: ty::Ty<'tcx>,
    ) -> SpirvTy {
        self.ty_cache.from_ty_as_ptr(&mut self.builder, tcx, ty)
    }
    pub fn new() -> Self {
        SpirvCtx {
            builder: Builder::new(),
            ty_cache: tycache::SpirvTyCache::new(),
            fns: HashMap::new(),
            forward_fns: HashMap::new(),
            vars: HashMap::new(),
            exprs: HashMap::new(),
        }
    }
}
#[derive(Copy, Clone, Debug)]
pub struct SpirvLabel(pub spirv::Word);
#[derive(Copy, Clone, Debug)]
pub struct SpirvFn(pub spirv::Word);
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
    pub entry: Option<IntrinsicEntry>,
    pub def_id: Option<hir::def_id::DefId>,
    pub merge_collector: Option<MergeCollector>,
    pub constants: HashMap<mir::Constant<'tcx>, SpirvVar>,
    pub forward_labels: Vec<SpirvLabel>,
    pub label_blocks: HashMap<mir::BasicBlock, SpirvLabel>,
    pub ctx: SpirvCtx<'tcx>,
}

#[derive(Debug)]
pub enum IntrinsicVec {
    Vec(u32),
}
#[derive(Debug)]
pub enum IntrinsicEntry {
    Vertex,
    Fragment,
}
pub fn extract_intrinsic_entry(attr: &[syntax::ast::Attribute]) -> Option<IntrinsicEntry> {
    let spirv = attr.iter()
        .filter_map(|a| a.meta())
        .find(|meta| meta.name.as_str() == "entry");
    if let Some(spirv) = spirv {
        let list = spirv.meta_item_list().and_then(|nested_list| {
            nested_list
                .iter()
                .map(|nested_meta| match nested_meta.node {
                    syntax::ast::NestedMetaItemKind::MetaItem(ref meta) => {
                        match &*meta.name.as_str() {
                            "vertex" => IntrinsicEntry::Vertex,
                            "fragment" => IntrinsicEntry::Fragment,
                            ref rest => unimplemented!("{:?}", rest),
                        }
                    }
                    ref rest => unimplemented!("{:?}", rest),
                })
                .nth(0)
        });
        return list;
    }
    None
}
pub fn extract_intrinsic_vec(attr: &[syntax::ast::Attribute]) -> Option<Vec<IntrinsicVec>> {
    let spirv = attr.iter()
        .filter_map(|a| a.meta())
        .find(|meta| meta.name.as_str() == "spirv");
    if let Some(spirv) = spirv {
        let list = spirv.meta_item_list().map(|nested_list| {
            nested_list
                .iter()
                .map(|nested_meta| match nested_meta.node {
                    syntax::ast::NestedMetaItemKind::Literal(ref lit) => match lit.node {
                        syntax::ast::LitKind::Str(ref sym, _) => IntrinsicVec::Vec(2),
                        ref rest => unimplemented!("{:?}", rest),
                    },
                    syntax::ast::NestedMetaItemKind::MetaItem(ref meta) => {
                        match &*meta.name.as_str() {
                            "Vec2" => IntrinsicVec::Vec(2),
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
            let map: &Map = &tcx.hir;
            let node_id = map.as_local_node_id(instance.def_id()).expect("node id");
            let name = map.name(node_id);
            println!("name = {:?}", name);
            let node = map.find(node_id).expect("node");
            if let rustc::hir::map::Node::NodeItem(ref item) = node {
                visitor.entry = extract_intrinsic_entry(&*item.attrs);
            }
            //println!("node = {:#?}", node);
            if let Some(ref mir) = mir {
                visitor.def_id = Some(instance.def_id());
                visitor.mir = Some(mir);
                visitor.merge_collector = Some(merge_collector(mir));
                visitor.visit_mir(mir);
                visitor.mir = None;
            }

            println!("instance = {:?}", instance);
        }
    }
    //visitor.build_module();
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
impl<'a, 'tcx: 'a> RlslVisitor<'a, 'tcx> {
    pub fn load_operand<'gcx>(&mut self, operand: &mir::Operand<'tcx>) -> SpirvOperand {
        let mir = self.mir.unwrap();
        let local_decls = &mir.local_decls;
        let ty = operand.ty(local_decls, self.ty_ctx);
        let spirv_ty = self.ctx.from_ty(ty);
        match operand {
            &mir::Operand::Consume(ref lvalue) => match lvalue {
                &mir::Lvalue::Local(local) => {
                    let local_decl = &local_decls[local];
                    let spirv_ty = self.ctx.from_ty(local_decl.ty);
                    let spirv_var = self.ctx.vars.get(&local).expect("local");
                    SpirvOperand::Consume(*spirv_var)
                }
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
                    if let Some(const_var) = self.constants.get(constant) {
                        //                        let load_const = self.ctx
                        //                            .builder
                        //                            .load(spirv_ty.word, None, expr.0, None, &[])
                        //                            .expect("load");
                        self.ctx
                            .builder
                            .store(const_var.0, expr.0, None, &[])
                            .expect("store");
                        return SpirvOperand::Consume(*const_var);
                    }
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
            entry: None,
            def_id: None,
            merge_collector: None,
            constants: HashMap::new(),
            label_blocks: HashMap::new(),
            forward_labels: Vec::new(),
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
    fn visit_basic_block_data(&mut self, block: mir::BasicBlock, data: &mir::BasicBlockData<'tcx>) {
        println!("basic block");
        let label = self.ctx
            .builder
            .begin_basic_block(self.forward_labels.pop().map(|i| i.0))
            .expect("begin block");
        let spirv_label = SpirvLabel(label);
        self.label_blocks.insert(block, spirv_label);

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
        let constants = collect_constants(mir);
        let ret_ty_spirv = self.ctx.from_ty(mir.return_ty);
        let args_ty: Vec<_> = mir.args_iter()
            .map(|l| {
                let ty = mir.local_decls[l].ty;
                let t = ty::TypeAndMut {
                    ty,
                    mutbl: rustc::hir::Mutability::MutMutable,
                };
                self.ty_ctx.mk_ptr(t)
            })
            .collect();
        let fn_sig = self.ty_ctx.mk_fn_sig(
            args_ty.into_iter(),
            mir.return_ty,
            false,
            hir::Unsafety::Normal,
            syntax::abi::Abi::Rust,
        );
        let fn_ty = self.ty_ctx.mk_fn_ptr(ty::Binder(fn_sig));
        let fn_ty_spirv = self.ctx.from_ty(fn_ty);

        let def_id = self.def_id.unwrap();
        let forward_fn = self.ctx.forward_fns.get(&def_id).map(|f| f.0);
        let spirv_function = self.ctx
            .builder
            .begin_function(
                ret_ty_spirv.word,
                forward_fn,
                spirv::FunctionControl::empty(),
                fn_ty_spirv.word,
            )
            .expect("begin fn");
        let def_id = self.def_id.unwrap();
        self.ctx.fns.insert(def_id, SpirvFn(spirv_function));
        //self.ctx.builder.begin_basic_block(None).expect("block");
        for local_arg in mir.args_iter() {
            let local_decl = &mir.local_decls[local_arg];
            let spirv_arg_ty = self.ctx.from_ty_as_ptr(self.ty_ctx, local_decl.ty);
            let spirv_param = self.ctx
                .builder
                .function_parameter(spirv_arg_ty.word)
                .expect("fn param");
            self.ctx.vars.insert(local_arg, SpirvVar(spirv_param));
        }
        //        macro_rules! basic_blocks {
        //                    (mut) => (mir.basic_blocks_mut().iter_enumerated_mut());
        //                    () => (mir.basic_blocks().iter_enumerated());
        //                };
        //        for (bb, data) in basic_blocks!() {
        //            self.visit_basic_block_data(bb, data);
        //        }
        self.ctx.builder.begin_basic_block(None).expect("block");
        for local_var in mir.vars_and_temps_iter() {
            let local_decl = &mir.local_decls[local_var];
            let spirv_var_ty = self.ctx.from_ty_as_ptr(self.ty_ctx, local_decl.ty);
            let spirv_var = self.ctx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.ctx.vars.insert(local_var, SpirvVar(spirv_var));
        }
        for constant in constants {
            let spirv_var_ty = self.ctx.from_ty_as_ptr(self.ty_ctx, constant.ty);
            let spirv_var = self.ctx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.constants.insert(constant, SpirvVar(spirv_var));
        }
        {
            use rustc_data_structures::indexed_vec::Idx;
            let local = mir::Local::new(0);
            let local_decl = &mir.local_decls[local];
            let spirv_var_ty = self.ctx.from_ty_as_ptr(self.ty_ctx, local_decl.ty);
            let spirv_var = self.ctx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.ctx.vars.insert(local, SpirvVar(spirv_var));
        }
        let next_label = SpirvLabel(self.ctx.builder.id());
        self.forward_labels.push(next_label);
        self.ctx.builder.branch(next_label.0);
        self.super_mir(mir);
        // TODO: Other cases
        //        for scope in &mir.visibility_scopes {
        //            self.visit_visibility_scope_data(scope);
        //        }
        //        let lookup = mir::visit::Lookup::Src(mir::SourceInfo {
        //            span: mir.span,
        //            scope: mir::ARGUMENT_VISIBILITY_SCOPE,
        //        });
        //        for local_decl in &mir.local_decls {
        //            self.visit_local_decl(local_decl);
        //        }
        //        self.visit_ty(&mir.return_ty, lookup);
        //
        //        self.visit_span(&mir.span);
        //println!("mir = {:#?}", mir);
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
        let ty = rvalue.ty(&self.mir.unwrap().local_decls, self.ty_ctx);
        if let ty::TypeVariants::TyTuple(ref slice, _) = ty.sty {
            if slice.len() == 0 {
                return;
            }
        }
        match lvalue {
            &mir::Lvalue::Local(local) => {
                let expr = self.ctx.exprs.get(&location).expect("expr");
                let var = self.ctx.vars.get(&local).expect("local");
                self.ctx
                    .builder
                    .store(var.0, expr.0, None, &[])
                    .expect("store");
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
        let mir = self.mir.unwrap();
        match kind {
            &mir::TerminatorKind::Return => {
                match mir.return_ty.sty {
                    ty::TypeVariants::TyTuple(ref slice, _) if slice.len() == 0 => {
                        self.ctx.builder.ret().expect("ret");
                    }
                    _ => {
                        use rustc_data_structures::indexed_vec::Idx;
                        let spirv_ty = { self.ctx.from_ty(mir.return_ty) };
                        let var = self.ctx.vars.get(&mir::Local::new(0)).unwrap();
                        let load = self.ctx
                            .builder
                            .load(spirv_ty.word, None, var.0, None, &[])
                            .expect("load");
                        self.ctx.builder.ret_value(load).expect("ret value");
                    }
                    _ => (),
                };
            }
            &mir::TerminatorKind::SwitchInt { ref targets, .. } => {
                use rustc_data_structures::control_flow_graph::ControlFlowGraph;
                let mir = self.mir.unwrap();
                println!("targets = {:?}", targets);
                let collector = self.merge_collector.as_ref().unwrap();
                let merge_block = collector.get(&location).expect("Unable to find a merge block");
                println!("merge_block = {:?}", merge_block);
//                let label = self.ctx.builder.id();
//                let spirv_label = SpirvLabel(label);
//                self.forward_labels.push(spirv_label);
//                self.ctx.builder.branch(label).expect("label");
            }
            &mir::TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                ..
            } => {
                let destionation = destination.as_ref().expect("Fn call is diverging");
                let local_decls = &self.mir.unwrap().local_decls;
                match func {
                    &mir::Operand::Constant(ref constant) => {
                        let ret_ty_binder = constant.ty.fn_sig(self.ty_ctx).output();
                        let ret_ty = self.ty_ctx
                            .erase_late_bound_regions_and_normalize(&ret_ty_binder);
                        let spirv_ty = self.ctx.from_ty(ret_ty);
                        if let mir::Literal::Value { ref value } = constant.literal {
                            use rustc::middle::const_val::ConstVal;
                            if let &ConstVal::Function(def_id, ref subst) = value {
                                let resolve_fn_id = resolve_fn_call(self.ty_ctx, def_id, subst);
                                let spirv_fn =
                                    self.ctx.fns.get(&resolve_fn_id).map(|v| *v).unwrap_or_else(
                                        || {
                                            let forward_id = SpirvFn(self.ctx.builder.id());
                                            self.ctx.forward_fns.insert(resolve_fn_id, forward_id);
                                            forward_id
                                        },
                                    );
                                let arg_operand_loads: Vec<_> = args.iter()
                                    .map(|arg| {
                                        let operand = self.load_operand(arg);
                                        match operand {
                                            SpirvOperand::Consume(var) => var.0,
                                            SpirvOperand::ConstVal(constant) => panic!(""),
                                        }
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
                                            let var = self.ctx.vars.get(&local).expect("local");
                                            self.ctx
                                                .builder
                                                .store(var.0, spirv_fn_call, None, &[])
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
                let label = self.ctx.builder.id();
                let spirv_label = SpirvLabel(label);
                self.forward_labels.push(spirv_label);
                self.ctx.builder.branch(label).expect("label");
            }
            rest => unimplemented!("{:?}", rest),
        };
    }
    fn visit_local_decl(&mut self, local_decl: &mir::LocalDecl<'tcx>) {
        self.super_local_decl(local_decl);
    }
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: mir::Location) {
        use rustc::mir::HasLocalDecls;
        self.super_rvalue(rvalue, location);
        //println!("location = {:?}", location);
        let local_decls = &self.mir.unwrap().local_decls;
        let spirv_ty = self.ctx.from_ty(rvalue.ty(local_decls, self.ty_ctx));
        match rvalue {
            &mir::Rvalue::BinaryOp(op, ref l, ref r) => {
                // TODO: Different types
                let l_load = self.load_operand(l).load_raw(&mut self.ctx, spirv_ty);
                let r_load = self.load_operand(r).load_raw(&mut self.ctx, spirv_ty);
                let add = self.ctx
                    .builder
                    .fadd(spirv_ty.word, None, l_load, r_load)
                    .expect("fadd");
                self.ctx.exprs.insert(location, SpirvExpr(add));
            }
            &mir::Rvalue::Use(ref operand) => {
                let load = self.load_operand(operand).load_raw(&mut self.ctx, spirv_ty);
                let expr = SpirvExpr(load);
                self.ctx.exprs.insert(location, expr);
            }
            &mir::Rvalue::NullaryOp(..) => {}
            &mir::Rvalue::CheckedBinaryOp(..) => {}
            &mir::Rvalue::Discriminant(..) => {}
            &mir::Rvalue::Aggregate(..) => {}
            rest => unimplemented!("{:?}", rest),
        }
    }
}
