#![feature(rustc_private)]
#![feature(box_syntax)]
#![feature(try_from)]

extern crate arena;
extern crate env_logger;
extern crate getopts;
extern crate log;
extern crate rspirv;
extern crate rustc;
extern crate rustc_borrowck;
extern crate rustc_const_math;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_incremental;
extern crate rustc_mir;
extern crate rustc_passes;
extern crate rustc_plugin;
extern crate rustc_resolve;
extern crate rustc_trans;
extern crate spirv_headers as spirv;
extern crate syntax;
extern crate syntax_pos;
pub mod trans;
use rustc::ty::layout::{HasDataLayout, LayoutOf, TargetDataLayout, TyLayout};
use rustc_data_structures::indexed_vec::Idx;
use std::collections::HashMap;
use rustc::{hir, mir};
use rustc_data_structures::fx::FxHashSet;
use rustc::mir::mono::MonoItem;
use rustc::ty::{Binder, Instance, ParamEnv, Ty, TyCtxt, TypeVariants, TypeckTables};

pub mod context;
pub mod ty;
pub mod collector;

use self::context::{MirContext, SpirvCtx};
use self::ty::*;

pub enum IntrinsicFn {
    Dot,
}

#[derive(Debug)]
pub struct Entry {
    next_id: usize,
    locations: HashMap<SpirvTy, usize>,
    input_vars: HashMap<usize, spirv::Word>,
}

impl Entry {
    pub fn new() -> Self {
        Entry {
            next_id: 0,
            locations: HashMap::new(),
            input_vars: HashMap::new(),
        }
    }

    fn insert(&mut self, ty: SpirvTy) {
        if !self.locations.contains_key(&ty) {
            self.locations.insert(ty, self.next_id);
            self.next_id += 1;
        }
    }

    pub fn get_location(&self, ty: SpirvTy) -> Option<usize> {
        self.locations.get(&ty).map(|l| *l)
    }
    pub fn get_input_var(&self, ty: SpirvTy) -> Option<spirv::Word> {
        self.get_location(ty)
            .and_then(|l| self.input_vars.get(&l))
            .map(|l| *l)
    }
}

fn intrinsic_fn(attrs: &[syntax::ast::Attribute]) -> Option<IntrinsicFn> {
    attrs
        .iter()
        .filter_map(|attr| {
            extract_attr(attr, &[], |s| match s {
                "dot" => Some(IntrinsicFn::Dot),
                _ => None,
            })
        })
        .nth(0)
}

#[derive(Debug, Copy, Clone)]
pub enum Intrinsic {
    GlslExt(spirv::Word),
    Abort,
}

pub struct RlslVisitor<'b, 'a: 'b, 'tcx: 'a> {
    current_table: Vec<&'a TypeckTables<'tcx>>,
    pub mcx: MirContext<'a, 'tcx>,
    pub scx: &'b mut SpirvCtx<'a, 'tcx>,
    pub entry: Option<IntrinsicEntry>,
    pub constants: HashMap<mir::Constant<'tcx>, SpirvVar<'tcx>>,
    pub label_blocks: HashMap<mir::BasicBlock, SpirvLabel>,
    pub vars: HashMap<mir::Local, SpirvVar<'tcx>>,
    pub entry_point: &'b Entry,
}

#[derive(Debug)]
pub enum IntrinsicType {
    Vec(usize),
}
impl IntrinsicType {
    pub fn from_attr(attrs: &[syntax::ast::Attribute]) -> Option<Self> {
        attrs
            .iter()
            .filter_map(|attr| {
                extract_attr(attr, &["spirv", "ty"], |s| match s {
                    "Vec2" => Some(IntrinsicType::Vec(2)),
                    _ => None,
                })
            })
            .nth(0)
    }
}
#[derive(Debug)]
pub enum IntrinsicEntry {
    Vertex,
    Fragment,
}

pub fn extract_attr_impl<'a, R, F>(
    meta_item: &'a syntax::ast::MetaItem,
    keywords: &[&str],
    f: F,
) -> Option<R>
where
    F: Fn(&str) -> Option<R>,
{
    if keywords.is_empty() {
        return f(&*meta_item.name().as_str());
    }
    let meta_item_list = &meta_item.meta_item_list()?[0];
    if meta_item.name() == keywords[0] {
        if let syntax::ast::NestedMetaItemKind::MetaItem(ref meta) = meta_item_list.node {
            return extract_attr_impl(meta, &keywords[1..], f);
        }
    }
    None
}
pub fn extract_attr<'a, R, F>(
    attr: &'a syntax::ast::Attribute,
    keywords: &[&str],
    f: F,
) -> Option<R>
where
    F: Fn(&str) -> Option<R>,
{
    if let Some(ref meta) = attr.meta() {
        return extract_attr_impl(meta, keywords, f);
    }
    None
}

#[repr(u32)]
pub enum GlslExtId {
    Round = 1,
    Sqrt = 31,
    Sin = 13,
    Cos = 14,
}

pub enum SpirvFunctionCall {
    Function(SpirvFn),
    Intrinsic(Intrinsic),
}

pub fn trans_spirv<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, items: &'a FxHashSet<MonoItem<'tcx>>) {
    use rustc::mir::visit::Visitor;
    let entry_fn = tcx.sess
        .entry_fn
        .borrow()
        .map(|(node_id, _)| tcx.hir.local_def_id(node_id))
        .expect("entry");
    let mut ctx = SpirvCtx::new(tcx);
    items
        .iter()
        .filter_map(|item| {
            if let &MonoItem::Fn(ref instance) = item {
                return Some(instance);
            }
            None
        })
        .for_each(|instance| {
            //            let a = tcx.def_path(instance.def_id())
            //                .data
            //                .into_iter()
            //                .map(|elem| elem.data.to_string())
            //                .collect::<Vec<_>>();
            //let a = hir.as_local_node_id(instance.def_id()).map(|n| tcx.node_path_str(n));
            //println!("a = {:?}", a);
            if tcx.is_foreign_item(instance.def_id()) {
                let intrinsic_name = &*tcx.item_name(instance.def_id());
                let id = match intrinsic_name {
                    "sqrtf32" => Some(GlslExtId::Sqrt),
                    "sinf32" => Some(GlslExtId::Sin),
                    "cosf32" => Some(GlslExtId::Cos),
                    _ => None,
                };
                if let Some(id) = id {
                    ctx.intrinsic_fns
                        .insert(instance.def_id(), Intrinsic::GlslExt(id as u32));
                }
                let abort = match intrinsic_name {
                    "abort" => Some(Intrinsic::Abort),
                    _ => None,
                };
                if let Some(abort) = abort {
                    ctx.intrinsic_fns.insert(instance.def_id(), abort);
                }
            } else {
                ctx.forward_fns
                    .insert(instance.def_id(), SpirvFn(ctx.builder.id()));
            }
        });
    println!("entry {:?}", entry_fn);
    let instances: Vec<_> = items
        .iter()
        .filter_map(|item| {
            if let &MonoItem::Fn(ref instance) = item {
                if let Some(mir) = tcx.maybe_optimized_mir(instance.def_id()) {
                    return Some(MirContext {
                        mir,
                        def_id: instance.def_id(),
                        substs: instance.substs,
                        tcx,
                    });
                }
            }
            None
        })
        .collect();
    let entry_instances = instances.iter().filter(|&&mtx| {
        let intrinsic = tcx.get_attrs(mtx.def_id)
            .iter()
            .filter_map(|attr| {
                extract_attr(attr, &["spirv"], |s| match s {
                    "vertex" => Some(IntrinsicEntry::Vertex),
                    "fragment" => Some(IntrinsicEntry::Fragment),
                    _ => None,
                })
            })
            .nth(0);
        intrinsic.is_some()
    });
    let mut entry = Entry::new();
    for &mtx in entry_instances {
        for arg in mtx.mir.args_iter() {
            let ty = mtx.mir.local_decls[arg].ty;
            let spirv_ty = ctx.to_ty_as_ptr(ty, mtx, spirv::StorageClass::Input);
            entry.insert(spirv_ty);
        }
    }

    for (ty, &location) in &entry.locations {
        let var = ctx.builder
            .variable(ty.word, None, spirv::StorageClass::Input, None);
        entry.input_vars.insert(location, var);
    }

    instances
        .iter()
        .filter(|mtx| mtx.def_id != entry_fn && tcx.lang_items().start_fn() != Some(mtx.def_id))
        .for_each(|&mtx| {
            let mut visitor = RlslVisitor::new(mtx, &entry, &mut ctx);
            let intrinsic = tcx.get_attrs(mtx.def_id)
                .iter()
                .filter_map(|attr| {
                    extract_attr(attr, &["spirv"], |s| match s {
                        "vertex" => Some(IntrinsicEntry::Vertex),
                        "fragment" => Some(IntrinsicEntry::Fragment),
                        _ => None,
                    })
                })
                .nth(0);
            visitor.entry = intrinsic;
            visitor.visit_mir(mtx.mir);
        });
    ctx.build_module();
}
#[derive(Debug)]
pub struct AccessChain {
    pub base: mir::Local,
    pub indices: Vec<usize>,
}

impl AccessChain {
    pub fn compute<'r, 'tcx>(lvalue: &'r mir::Place<'tcx>) -> Self {
        fn access_chain_indices<'r, 'tcx>(
            lvalue: &'r mir::Place<'tcx>,
            mut indices: Vec<usize>,
        ) -> (&'r mir::Place<'tcx>, Vec<usize>) {
            if let &mir::Place::Projection(ref proj) = lvalue {
                if let mir::ProjectionElem::Field(field, _) = proj.elem {
                    indices.push(field.index());
                    return access_chain_indices(&proj.base, indices);
                }
                if let mir::ProjectionElem::Downcast(_, id) = proj.elem {
                    indices.push(id);
                    return access_chain_indices(&proj.base, indices);
                }
            }
            (lvalue, indices)
        }
        let indices = Vec::new();
        let (base, mut indices) = access_chain_indices(lvalue, indices);
        let local = match base {
            &mir::Place::Local(local) => local,
            _ => panic!("Should be local"),
        };
        indices.reverse();
        AccessChain {
            base: local,
            indices,
        }
    }
}

impl<'b, 'a, 'tcx: 'a> RlslVisitor<'b, 'a, 'tcx> {
    pub fn to_ty(&mut self, ty: Ty<'tcx>, storage_class: spirv::StorageClass) -> SpirvTy {
        self.scx.to_ty(ty, self.mcx, storage_class)
    }
    pub fn to_ty_as_ptr(&mut self, ty: Ty<'tcx>, storage_class: spirv::StorageClass) -> SpirvTy {
        self.scx.to_ty_as_ptr(ty, self.mcx, storage_class)
    }
    pub fn to_ty_fn(&mut self, ty: Ty<'tcx>) -> SpirvTy {
        self.scx.to_ty(ty, self.mcx, spirv::StorageClass::Function)
    }
    pub fn to_ty_as_ptr_fn(&mut self, ty: Ty<'tcx>) -> SpirvTy {
        self.scx
            .to_ty_as_ptr(ty, self.mcx, spirv::StorageClass::Function)
    }
    pub fn constant(&mut self, val: SpirvConstVal) -> SpirvValue {
        self.scx.constant(self.mcx, val)
    }
    pub fn constant_f32(&mut self, value: f32) -> SpirvValue {
        self.scx.constant_f32(self.mcx, value)
    }
    pub fn constant_u32(&mut self, value: u32) -> SpirvValue {
        self.scx.constant_u32(self.mcx, value)
    }

    pub fn get_table(&self) -> &'a TypeckTables<'tcx> {
        self.current_table.last().expect("no table yet")
    }
    pub fn new(
        mir_ctx: MirContext<'a, 'tcx>,
        entry_point: &'b Entry,
        stx: &'b mut SpirvCtx<'a, 'tcx>,
    ) -> Self {
        let visitor = RlslVisitor {
            scx: stx,
            entry_point,
            current_table: Vec::new(),
            mcx: mir_ctx,
            entry: None,
            constants: HashMap::new(),
            label_blocks: HashMap::new(),
            vars: HashMap::new(),
        };
        visitor
    }
}
fn is_ptr(ty: Ty) -> bool {
    ty.is_unsafe_ptr() || ty.is_mutable_pointer() || ty.is_region_ptr()
}
// TODO: More than two cases
pub fn find_merge_block(
    mir: &mir::Mir,
    root: mir::BasicBlock,
    targets: &[mir::BasicBlock],
) -> Option<mir::BasicBlock> {
    use rustc_data_structures::control_flow_graph::iterate::post_order_from;
    use rustc_data_structures::control_flow_graph::dominators::dominators;
    use std::collections::HashSet;
    let dominators = dominators(mir);
    let true_order: HashSet<_> = post_order_from(mir, targets[0]).into_iter().collect();
    let false_order: HashSet<_> = post_order_from(mir, targets[1]).into_iter().collect();
    true_order
        .intersection(&false_order)
        .filter(|&&target| dominators.is_dominated_by(target, root))
        .last()
        .map(|b| *b)
}

pub struct Enum<'tcx> {
    pub discr_ty: Ty<'tcx>,
    pub index: usize,
}

impl<'tcx> Enum<'tcx> {
    pub fn from_ty<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: Ty<'tcx>) -> Option<Enum<'tcx>> {
        use rustc::ty::layout::Variants;
        let e = (tcx, ParamEnv::empty(rustc::traits::Reveal::All))
            .layout_of(ty)
            .ok()
            .and_then(|layout| {
                if let Variants::Tagged {
                    ref discr,
                    ref variants,
                } = layout.variants
                {
                    Some((discr.value.to_ty(tcx), variants.len()))
                } else {
                    None
                }
            });
        if let Some((discr_ty, index)) = e {
            Some(Enum { discr_ty, index })
        } else {
            None
        }
    }
}

impl<'b, 'a, 'tcx: 'a> rustc::mir::visit::Visitor<'tcx> for RlslVisitor<'b, 'a, 'tcx> {
    fn visit_basic_block_data(&mut self, block: mir::BasicBlock, data: &mir::BasicBlockData<'tcx>) {
        {
            let spirv_label = self.label_blocks.get(&block).expect("no spirv label");
            let label = self.scx
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
        trans::statement::trans_statement(self, block, statement, location);
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
        use rustc_data_structures::control_flow_graph::iterate::reverse_post_order;
        for (block, _) in mir.basic_blocks().iter_enumerated() {
            self.label_blocks
                .insert(block, SpirvLabel(self.scx.builder.id()));
        }
        assert!(
            !is_ptr(mir.return_ty()),
            "Functions are not allowed to return a ptr"
        );
        let def_id = self.mcx.def_id;
        let spirv_function = if !self.entry.is_some() {
            let ret_ty_spirv = self.to_ty_fn(mir.return_ty());
            // If a param is not a ptr, we need to turn it into a ptr
            let args_ty: Vec<_> = mir.args_iter().map(|l| mir.local_decls[l].ty).collect();
            let fn_sig = self.scx.tcx.mk_fn_sig(
                args_ty.into_iter(),
                mir.return_ty(),
                false,
                hir::Unsafety::Normal,
                syntax::abi::Abi::Rust,
            );
            let fn_ty = self.scx.tcx.mk_fn_ptr(Binder(fn_sig));
            let fn_ty_spirv = self.to_ty_fn(fn_ty);

            let forward_fn = self.scx
                .forward_fns
                .get(&def_id)
                .map(|f| f.0)
                .expect("forward");
            let spirv_function = self.scx
                .builder
                .begin_function(
                    ret_ty_spirv.word,
                    Some(forward_fn),
                    spirv::FunctionControl::empty(),
                    fn_ty_spirv.word,
                )
                .expect("begin fn");
            spirv_function
        } else {
            let void = self.scx.tcx.mk_nil();
            let fn_sig = self.scx.tcx.mk_fn_sig(
                [].into_iter(),
                &void,
                false,
                hir::Unsafety::Normal,
                syntax::abi::Abi::Rust,
            );
            let void_spirv = self.to_ty_fn(void);
            let fn_ty = self.scx.tcx.mk_fn_ptr(Binder(fn_sig));
            let fn_ty_spirv = self.to_ty_fn(fn_ty);
            let forward_fn = self.scx
                .forward_fns
                .get(&def_id)
                .map(|f| f.0)
                .expect("forward");
            let spirv_function = self.scx
                .builder
                .begin_function(
                    void_spirv.word,
                    Some(forward_fn),
                    spirv::FunctionControl::empty(),
                    fn_ty_spirv.word,
                )
                .expect("begin fn");
            spirv_function
        };
        if self.entry.is_none() {
            self.scx.name_from_def_id(def_id, spirv_function);
            let params: Vec<_> = mir.args_iter()
                .map(|local_arg| {
                    let local_decl = &mir.local_decls[local_arg];
                    let spirv_arg_ty = self.to_ty_fn(local_decl.ty);
                    let spirv_param = self.scx
                        .builder
                        .function_parameter(spirv_arg_ty.word)
                        .expect("fn param");

                    if let Some(name) = local_decl.name {
                        self.scx.name_from_str(name.as_str().as_ref(), spirv_param);
                    }
                    spirv_param
                })
                .collect();
            self.scx.builder.begin_basic_block(None).expect("block");
            for (index, param) in params.into_iter().enumerate() {
                let local_arg = mir::Local::new(index + 1);
                let local_decl = &mir.local_decls[local_arg];
                let spirv_var_ty = self.to_ty_as_ptr_fn(local_decl.ty);
                let spirv_var = self.scx.builder.variable(
                    spirv_var_ty.word,
                    None,
                    spirv::StorageClass::Function,
                    None,
                );
                self.scx
                    .builder
                    .store(spirv_var, param, None, &[])
                    .expect("store");
                self.vars
                    .insert(local_arg, SpirvVar::new(spirv_var, false, local_decl.ty));
            }
        } else {
            self.scx.builder.begin_basic_block(None).expect("block");
            for arg in mir.args_iter() {
                let ty = &mir.local_decls[arg].ty;
                let spirv_ty = self.to_ty_as_ptr_fn(ty);
                let var = self.entry_point.get_input_var(spirv_ty).expect("input");
                if let Some(name) = mir.local_decls[arg].name {
                    self.scx.name_from_str(name.as_str().as_ref(), var);
                }
                self.vars.insert(arg, SpirvVar::new(var, false, ty));
            }
        }
        for local_var in mir.vars_and_temps_iter() {
            let local_decl = &mir.local_decls[local_var];
            let spirv_var_ty = self.to_ty_as_ptr_fn(local_decl.ty);
            let spirv_var = self.scx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            if let Some(name) = local_decl.name {
                self.scx.name_from_str(name.as_str().as_ref(), spirv_var);
            }
            self.vars
                .insert(local_var, SpirvVar::new(spirv_var, false, local_decl.ty));
        }
        {
            use rustc_data_structures::indexed_vec::Idx;
            let local = mir::Local::new(0);
            let local_decl = &mir.local_decls[local];
            let spirv_var_ty = self.to_ty_as_ptr_fn(local_decl.ty);
            let spirv_var = self.scx.builder.variable(
                spirv_var_ty.word,
                None,
                spirv::StorageClass::Function,
                None,
            );
            self.scx.name_from_str("retvar", spirv_var);
            self.vars
                .insert(local, SpirvVar::new(spirv_var, false, local_decl.ty));
            let spirv_label = self.label_blocks
                .get(&mir::BasicBlock::new(0))
                .expect("No first label");
            self.scx.builder.branch(spirv_label.0).expect("branch");
        }
        //self.super_mir(mir);
        let order = reverse_post_order(mir, mir::BasicBlock::new(0));
        for bb in order {
            self.visit_basic_block_data(bb, &mir.basic_blocks()[bb]);
        }

        for scope in &mir.visibility_scopes {
            self.visit_visibility_scope_data(scope);
        }

        let lookup = mir::visit::TyContext::ReturnTy(mir::SourceInfo {
            span: mir.span,
            scope: mir::ARGUMENT_VISIBILITY_SCOPE,
        });
        self.visit_ty(&mir.return_ty(), lookup);

        for (local, local_decl) in mir.local_decls.iter_enumerated() {
            self.visit_local_decl(local, local_decl);
        }

        self.visit_span(&mir.span);
        self.scx.builder.end_function().expect("end fn");
        if self.entry.is_some() {
            let inputs: Vec<_> = mir.args_iter()
                .map(|arg| {
                    let ty = &mir.local_decls[arg].ty;
                    let spirv_ty = self.to_ty_as_ptr_fn(ty);
                    let var = self.entry_point.get_input_var(spirv_ty).expect("input");
                    var
                })
                .collect();
            let name = self.mcx.tcx.item_name(self.mcx.def_id);
            self.scx.builder.entry_point(
                spirv::ExecutionModel::Vertex,
                spirv_function,
                name.as_ref(),
                &inputs,
            );
            self.scx.builder.execution_mode(
                spirv_function,
                spirv::ExecutionMode::OriginUpperLeft,
                &[],
            );
        }
    }
    fn visit_assign(
        &mut self,
        block: mir::BasicBlock,
        lvalue: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: mir::Location,
    ) {
        self.super_assign(block, lvalue, rvalue, location);
        let ty = rvalue.ty(&self.mcx.mir.local_decls, self.scx.tcx);
        if let TypeVariants::TyTuple(ref slice, _) = ty.sty {
            if slice.len() == 0 {
                return;
            }
        }
        let spirv_ty = self.to_ty_fn(ty);
        let lvalue_ty = lvalue
            .ty(&self.mcx.mir.local_decls, self.scx.tcx)
            .to_ty(self.scx.tcx);
        let lvalue_ty_spirv = self.to_ty_fn(lvalue_ty);
        let expr = match rvalue {
            &mir::Rvalue::BinaryOp(op, ref l, ref r)
            | &mir::Rvalue::CheckedBinaryOp(op, ref l, ref r) => {
                self.scx.binary_op(self.mcx, &self.vars, spirv_ty, op, l, r)
            }
            &mir::Rvalue::Use(ref operand) => {
                let is_ptr = lvalue_ty.is_unsafe_ptr() || lvalue_ty.is_mutable_pointer()
                    || lvalue_ty.is_region_ptr();

                let operand = self.scx.load_operand(self.mcx, &self.vars, operand);
                if is_ptr && operand.is_param() {
                    SpirvValue(operand.expect_var().word)
                } else {
                    let load = operand.load_raw(&mut self.scx, spirv_ty);
                    let expr = SpirvValue(load);
                    expr
                }
            }

            //            &mir::Rvalue::NullaryOp(..) => {}
            //            &mir::Rvalue::CheckedBinaryOp(..) => {}
            //            &mir::Rvalue::Discriminant(..) => {}
            &mir::Rvalue::Aggregate(_, ref operands) => {
                // If there are no operands, then it should be 0 sized and we can
                // abort.
                if operands.is_empty() {
                    return;
                }

                let spirv_operands: Vec<_> = operands
                    .iter()
                    .map(|op| {
                        let ty = op.ty(&self.mcx.mir.local_decls, self.mcx.tcx);
                        let spirv_ty = self.to_ty_fn(ty);
                        self.scx
                            .load_operand(self.mcx, &self.vars, op)
                            .load_raw(self.scx, spirv_ty)
                    })
                    .collect();
                SpirvValue(
                    self.scx
                        .builder
                        .composite_construct(lvalue_ty_spirv.word, None, &spirv_operands)
                        .expect("composite"),
                )
            }
            &mir::Rvalue::Ref(_, _, ref lvalue) => {
                let is_ptr = lvalue_ty.is_unsafe_ptr() || lvalue_ty.is_mutable_pointer()
                    || lvalue_ty.is_region_ptr();
                match lvalue {
                    &mir::Place::Local(local) => {
                        let var = self.vars.get(&local).expect("no local");
                        if is_ptr {
                            //self.ctx.builder.load(self.ctx.from_ty(var.ty))
                            // TODO
                            SpirvValue(var.word)
                        } else {
                            SpirvValue(var.word)
                        }
                    }
                    rest => unimplemented!("{:?}", rest),
                }
            }
            &mir::Rvalue::Discriminant(ref lvalue) => {
                let local = match lvalue {
                    &mir::Place::Local(local) => local,
                    _ => panic!("Should be local"),
                };
                let var = *self.vars.get(&local).expect("local");
                let ty = self.mcx.mir.local_decls[local].ty;
                let enum_data = Enum::from_ty(self.mcx.tcx, ty).expect("enum");
                let discr_ty_spirv = self.to_ty_fn(enum_data.discr_ty);
                let discr_ty_spirv_ptr = self.to_ty_as_ptr_fn(enum_data.discr_ty);
                let index = self.constant_u32(enum_data.index as u32).0;
                let access = self.scx
                    .builder
                    .access_chain(discr_ty_spirv_ptr.word, None, var.word, &[index])
                    .expect("access");
                let load = self.scx
                    .builder
                    .load(discr_ty_spirv.word, None, access, None, &[])
                    .expect("load");
                let target_ty_spirv = self.to_ty_fn(lvalue_ty);
                let cast = self.scx
                    .builder
                    .bitcast(target_ty_spirv.word, None, load)
                    .expect("bitcast");

                SpirvValue(cast)
            }
            &mir::Rvalue::Cast(_, ref op, ty) => {
                println!("op = {:?}", op);
                println!("ty = {:?}", ty);
                let op_ty = op.ty(&self.mcx.mir.local_decls, self.mcx.tcx);
                println!("op_ty = {:?}", op_ty);
                unimplemented!("cast")
            }

            rest => unimplemented!("{:?}", rest),
        };

        let access_chain = AccessChain::compute(lvalue);
        let spirv_var = *self.vars.get(&access_chain.base).expect("Local");
        let store = if access_chain.indices.is_empty() {
            spirv_var.word
        } else {
            let spirv_ty_ptr = self.to_ty_as_ptr_fn(ty);
            let indices: Vec<_> = access_chain
                .indices
                .iter()
                .map(|&i| self.constant_u32(i as u32).0)
                .collect();
            let access = self.scx
                .builder
                .access_chain(spirv_ty_ptr.word, None, spirv_var.word, &indices)
                .expect("access_chain");
            access
        };
        self.scx
            .builder
            .store(store, expr.0, None, &[])
            .expect("store");
        //        match lvalue {
        //            &mir::Place::Local(local) => {
        //                let var = self.vars.get(&local).expect("local");
        //                self.mtx
        //                    .stx
        //                    .builder
        //                    .store(var.word, expr.0, None, &[])
        //                    .expect("store");
        //            }
        //            &mir::Place::Projection(ref proj) => match &proj.elem {
        //                &mir::ProjectionElem::Field(field, ty) => match &proj.base {
        //                    &mir::Place::Local(local) => {
        //                        let var = self.vars.get(&local).expect("local");
        //                        let spirv_ty_ptr = self.mtx.from_ty_as_ptr(ty);
        //                        let index_ty = self.mtx.stx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
        //                        let spirv_index_ty = self.mtx.from_ty(index_ty);
        //                        let index = self.mtx
        //                            .stx
        //                            .builder
        //                            .constant_u32(spirv_index_ty.word, field.index() as u32);
        //                        let field_access = self.mtx
        //                            .stx
        //                            .builder
        //                            .in_bounds_access_chain(spirv_ty_ptr.word, None, var.word, &[index])
        //                            .expect("access chain");
        //                        let spirv_ptr = self.mtx.from_ty(ty);
        //                        //let load = self.ctx.builder.load(spirv_ptr.word, None, field_access, None, &[]).expect("load");
        //                        self.mtx
        //                            .stx
        //                            .builder
        //                            .store(field_access, expr.0, None, &[])
        //                            .expect("store");
        //                    }
        //                    rest => unimplemented!("{:?}", rest),
        //                },
        //                rest => unimplemented!("{:?}", rest),
        //            },
        //            rest => unimplemented!("{:?}", rest),
        //        };
    }

    fn visit_terminator_kind(
        &mut self,
        block: mir::BasicBlock,
        kind: &mir::TerminatorKind<'tcx>,
        location: mir::Location,
    ) {
        self.super_terminator_kind(block, kind, location);
        let mir = self.mcx.mir;
        match kind {
            &mir::TerminatorKind::Return => {
                match mir.return_ty().sty {
                    TypeVariants::TyTuple(ref slice, _) if slice.len() == 0 => {
                        self.scx.builder.ret().expect("ret");
                    }
                    _ => {
                        use rustc_data_structures::indexed_vec::Idx;
                        let spirv_ty = { self.to_ty_fn(mir.return_ty()) };
                        let var = self.vars.get(&mir::Local::new(0)).unwrap();
                        let load = self.scx
                            .builder
                            .load(spirv_ty.word, None, var.word, None, &[])
                            .expect("load");
                        self.scx.builder.ret_value(load).expect("ret value");
                    }
                };
            }
            &mir::TerminatorKind::Goto { target } => {
                let label = self.label_blocks.get(&target).expect("no goto label");
                self.scx.builder.branch(label.0).expect("branch");
            }
            &mir::TerminatorKind::SwitchInt {
                ref discr,
                switch_ty,
                ref targets,
                ..
            } => {
                let mir = self.mcx.mir;
                let spirv_ty = self.to_ty_fn(switch_ty);
                let selector = if switch_ty.is_bool() {
                    let load = self.scx
                        .load_operand(self.mcx, &self.vars, discr)
                        .load_raw(self.scx, spirv_ty);
                    let target_ty = self.mcx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
                    let target_ty_spirv = self.to_ty_fn(target_ty);
                    self.scx
                        .builder
                        .bitcast(target_ty_spirv.word, None, load)
                        .expect("bitcast")
                } else {
                    self.scx
                        .load_operand(self.mcx, &self.vars, discr)
                        .load_raw(self.scx, spirv_ty)
                };
                let default_label = *self.label_blocks
                    .get(targets.last().unwrap())
                    .expect("default label");
                let labels: Vec<_> = targets
                    .iter()
                    .take(targets.len() - 1)
                    .enumerate()
                    .map(|(index, target)| {
                        let label = self.label_blocks.get(&target).expect("label");
                        (index as u32, label.0)
                    })
                    .collect();
                {
                    let merge_block =
                        find_merge_block(mir, block, targets).expect("no merge block");
                    let merge_block_label = self.label_blocks.get(&merge_block).expect("no label");
                    self.scx
                        .builder
                        .selection_merge(merge_block_label.0, spirv::SelectionControl::empty())
                        .expect("selection merge");
                }
                self.scx
                    .builder
                    .switch(selector, default_label.0, &labels)
                    .expect("switch");
            }
            &mir::TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                ..
            } => {
                let local_decls = &self.mcx.mir.local_decls;
                match func {
                    &mir::Operand::Constant(ref constant) => {
                        let ret_ty_binder = constant.ty.fn_sig(self.scx.tcx).output();
                        let ret_ty = ret_ty_binder.skip_binder();
                        let spirv_ty = self.to_ty_fn(ret_ty);
                        if let mir::Literal::Value { ref value } = constant.literal {
                            use rustc::middle::const_val::ConstVal;
                            if let ConstVal::Function(def_id, ref substs) = value.val {
                                let mono_substs = self.mcx.monomorphize(substs);
                                let resolve_fn_id = Instance::resolve(
                                    self.scx.tcx,
                                    ParamEnv::empty(rustc::traits::Reveal::All),
                                    def_id,
                                    &mono_substs,
                                ).unwrap()
                                    .def_id();
                                let arg_operand_loads: Vec<_> = args.iter()
                                    .map(|arg| {
                                        let operand =
                                            self.scx.load_operand(self.mcx, &self.vars, arg);
                                        let arg_ty = arg.ty(local_decls, self.scx.tcx);
                                        let arg_ty_spirv = self.to_ty_fn(arg_ty);
                                        operand.load_raw(&mut self.scx, arg_ty_spirv)
                                    })
                                    .collect();
                                let fn_call = self.scx
                                    .get_function_call(resolve_fn_id)
                                    .expect("function call");
                                let spirv_fn_call = match fn_call {
                                    SpirvFunctionCall::Function(_) => {
                                        let spirv_fn = self.scx
                                            .forward_fns
                                            .get(&resolve_fn_id)
                                            .map(|v| *v)
                                            .expect("forward fn call");
                                        let fn_call = self.scx
                                            .builder
                                            .function_call(
                                                spirv_ty.word,
                                                None,
                                                spirv_fn.0,
                                                &arg_operand_loads,
                                            )
                                            .expect("fn call");
                                        Some(fn_call)
                                    }
                                    SpirvFunctionCall::Intrinsic(intrinsic) => match intrinsic {
                                        Intrinsic::GlslExt(id) => {
                                            let ext_fn = self.scx
                                                .builder
                                                .ext_inst(
                                                    spirv_ty.word,
                                                    None,
                                                    self.scx.glsl_ext_id,
                                                    id,
                                                    &arg_operand_loads,
                                                )
                                                .expect("ext instr");
                                            Some(ext_fn)
                                        }
                                        Intrinsic::Abort => {
                                            self.scx.builder.unreachable().expect("unreachable");
                                            None
                                        }
                                    },
                                };
                                if let (&Some(ref dest), Some(spirv_fn_call)) =
                                    (destination, spirv_fn_call)
                                {
                                    let &(ref lvalue, _) = dest;
                                    match lvalue {
                                        &mir::Place::Local(local) => {
                                            let var = self.vars.get(&local).expect("local");
                                            self.scx
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
                self.scx.builder.branch(target_label.0).expect("label");
            }
            &mir::TerminatorKind::Assert { target, .. } => {
                let target_label = self.label_blocks.get(&target).expect("no label");
                self.scx.builder.branch(target_label.0).expect("label");
            }
            rest => unimplemented!("{:?}", rest),
        };
    }
}
