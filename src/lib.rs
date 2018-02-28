#![feature(rustc_private)]
#![feature(box_syntax)]
#![feature(try_from)]
#![feature(conservative_impl_trait)]
#![feature(rustc_diagnostic_macros)]

extern crate byteorder;

extern crate arena;
extern crate env_logger;
extern crate getopts;
extern crate itertools;
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
extern crate rustc_typeck;
extern crate spirv_headers as spirv;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
pub mod trans;
use rustc::ty::layout::LayoutOf;
use rustc_data_structures::indexed_vec::Idx;
use rustc::{hir, mir};
use rustc_data_structures::fx::FxHashSet;
use rustc::mir::mono::MonoItem;
use rustc::ty::{Binder, Instance, ParamEnv, TyCtxt, TypeVariants, TypeckTables};
use rustc::mir::visit::{TyContext, Visitor};
pub mod context;
pub mod collector;
pub mod typ;
use rustc::ty;
use self::context::{CodegenCx, MirContext};
use self::typ::*;
use rustc::ty::subst::Substs;
use std::collections::HashMap;

use itertools::{Either, Itertools};
#[derive(Copy, Clone, Debug)]
pub enum IntrinsicFn {
    Dot,
}
register_diagnostics! {
    E1337,
}

pub fn remove_ptr_ty<'tcx>(ty: ty::Ty<'tcx>) -> ty::Ty<'tcx> {
    match ty.sty {
        TypeVariants::TyRef(_, type_and_mut) => remove_ptr_ty(type_and_mut.ty),
        _ => ty,
    }
}

pub struct TyErrorVisitor {
    has_error: bool,
}
impl TyErrorVisitor {
    pub fn has_error(mcxs: &[MirContext]) -> bool {
        let mut visitor = TyErrorVisitor { has_error: false };
        for mcx in mcxs {
            if visitor.has_error {
                return true;
            } else {
                visitor.visit_mir(mcx.mir);
            }
        }
        false
    }
}
impl<'tcx> rustc::mir::visit::Visitor<'tcx> for TyErrorVisitor {
    fn visit_ty(&mut self, ty: &ty::Ty<'tcx>, _: TyContext) {
        self.super_ty(ty);
        if let TypeVariants::TyError = ty.sty {
            self.has_error = true;
        }
    }
}

pub fn extract_location<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<u32> {
    if let TypeVariants::TyAdt(_, substs) = ty.sty {
        assert!(substs.len() == 1, "Len should be 1");
        let inner_ty = substs[0].as_type().expect("Should be ty");
        let location_id = inner_ty.ty_to_def_id().expect("id location");
        let attrs = tcx.get_attrs(location_id);
        let val = ::extract_attr(&attrs, "spirv", |s| match s {
            "Const0" => Some(0u32),
            "Const1" => Some(1),
            "Const2" => Some(2),
            "Const3" => Some(3),
            "Const4" => Some(4),
            "Const5" => Some(5),
            _ => None,
        });
        return val.get(0).map(|&i| i);
    }
    None
}

pub struct EntryPoint<'a, 'tcx: 'a> {
    pub entry_type: IntrinsicEntry,
    pub mcx: MirContext<'a, 'tcx>,
}

impl<'a, 'tcx> EntryPoint<'a, 'tcx> {
    pub fn input_iter(&'a self) -> impl Iterator<Item = Input<'tcx>> + 'a {
        self.mcx.mir.args_iter().filter_map(move |local| {
            let ty = self.mcx.mir.local_decls[local].ty;
            Input::new(self, ty)
        })
    }

    pub fn descriptor_iter(&'a self) -> impl Iterator<Item = Descriptor<'tcx>> + 'a {
        self.mcx.mir.args_iter().filter_map(move |local| {
            let ty = self.mcx.mir.local_decls[local].ty;
            Descriptor::new(self.mcx.tcx, ty)
        })
    }

    pub fn output_iter(&'a self) -> impl Iterator<Item = Output<'tcx>> + 'a {
        use std::iter::once;
        once(self.mcx.mir.return_ty()).filter_map(move |ty| Output::new(self.mcx.tcx, ty))
    }

    pub fn args(&self) -> Vec<mir::Local> {
        match self.entry_type {
            IntrinsicEntry::Vertex => self.mcx.mir.args_iter().skip(1).collect(),
            IntrinsicEntry::Fragment => self.mcx.mir.args_iter().collect(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GlobalVar<'a> {
    pub ty: rustc::ty::Ty<'a>,
    pub var: spirv::Word,
    pub location: u32,
    pub storage_class: spirv::StorageClass,
}
fn count_types<'a>(tys: &[rustc::ty::Ty<'a>]) -> HashMap<rustc::ty::Ty<'a>, usize> {
    tys.iter().fold(HashMap::new(), |mut map, ty| {
        {
            let count = map.entry(ty).or_insert(0);
            *count += 1;
        }
        map
    })
}

impl<'a> GlobalVar<'a> {}

fn is_per_vertex<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> bool {
    if let TypeVariants::TyRef(_, ty_and_mut) = ty.sty {
        if let TypeVariants::TyAdt(adt, substs) = ty_and_mut.ty.sty {
            let attrs = tcx.get_attrs(adt.did);
            return extract_attr(&attrs, "spirv", |s| match s {
                "PerVertex" => Some(true),
                _ => None,
            }).get(0)
                .is_some();
        }
    }
    false
}
pub type TyMap<'a> = HashMap<rustc::ty::Ty<'a>, GlobalVar<'a>>;

use std::hash::Hash;
#[derive(Debug)]
pub struct Entry<'tcx, T: Hash + Eq> {
    global_vars: HashMap<T, GlobalVar<'tcx>>,
}

use std::collections::HashSet;
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Input<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub location: u32,
}
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Output<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub location: u32,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Descriptor<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub set: u32,
    pub binding: u32,
}

impl<'tcx> Output<'tcx> {
    fn new<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        let (adt, substs) = get_builtin_adt(tcx, ty, "Output")?;
        let f = adt.all_fields().collect_vec();
        let fields: Vec<_> = adt.all_fields()
            .map(|field| field.ty(tcx, substs))
            .collect();
        assert!(fields.len() == 2, "Output should have two fields");
        let location_ty = fields[1];
        let location = extract_location(tcx, location_ty).expect("Unable to extract location");
        Some(Output { ty, location })
    }
}

impl<'tcx> Descriptor<'tcx> {
    fn new<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        let (adt, substs) = get_builtin_adt(tcx, ty, "Descriptor")?;
        let f = adt.all_fields().collect_vec();
        let fields: Vec<_> = adt.all_fields()
            .map(|field| field.ty(tcx, substs))
            .collect();
        assert_eq!(fields.len(), 3, "Descriptor should have 3 fields");
        let binding_ty = fields[1];
        let set_ty = fields[2];
        let binding = extract_location(tcx, binding_ty).expect("Unable to extract location");
        let set = extract_location(tcx, set_ty).expect("Unable to extract location");
        Some(Descriptor { ty, binding, set })
    }
}

use rustc::ty::AdtDef;
fn get_builtin_adt<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ty: ty::Ty<'tcx>,
    attribute: &str,
) -> Option<(&'tcx AdtDef, &'tcx Substs<'tcx>)> {
    if let TypeVariants::TyAdt(adt, substs) = ty.sty {
        let attrs = tcx.get_attrs(adt.did);
        extract_attr(&attrs, "spirv", |s| {
            if attribute == s {
                Some((adt, substs))
            } else {
                None
            }
        }).get(0)
            .map(|i| *i)
    } else {
        None
    }
}
impl<'tcx> Input<'tcx> {
    fn new<'a>(entry_point: &EntryPoint<'a, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        let (adt, substs) = get_builtin_adt(entry_point.mcx.tcx, ty, "Input")?;
        let f = adt.all_fields().collect_vec();
        let fields: Vec<_> = adt.all_fields()
            .map(|field| field.ty(entry_point.mcx.tcx, substs))
            .collect();
        assert!(fields.len() == 2, "Input should have two fields");
        let location_ty = fields[1];
        let location =
            extract_location(entry_point.mcx.tcx, location_ty).expect("Unable to extract location");
        Some(Input { ty, location })
    }
}

impl<'tcx> Global<'tcx> for Input<'tcx> {
    fn ty(&self) -> ty::Ty<'tcx> {
        self.ty
    }
}

impl<'tcx> Global<'tcx> for Output<'tcx> {
    fn ty(&self) -> ty::Ty<'tcx> {
        self.ty
    }
}
impl<'tcx> Global<'tcx> for Descriptor<'tcx> {
    fn ty(&self) -> ty::Ty<'tcx> {
        self.ty
    }
}

pub trait Global<'tcx>: Hash + Eq {
    fn ty(&self) -> ty::Ty<'tcx>;
}
impl<'tcx> Entry<'tcx, Input<'tcx>> {
    pub fn input<'a>(entry_points: &[EntryPoint<'a, 'tcx>], stx: &mut CodegenCx<'a, 'tcx>) -> Self {
        let set: HashSet<_> = entry_points
            .iter()
            .flat_map(EntryPoint::input_iter)
            .collect();
        Self::create(set, stx, spirv::StorageClass::Input)
    }

    fn variable_iter<'borrow, 'a>(
        &'borrow self,
        entry: &'borrow EntryPoint<'a, 'tcx>,
    ) -> impl Iterator<Item = (mir::Local, GlobalVar<'tcx>)> + 'borrow {
        entry
            .mcx
            .mir
            .args_iter()
            .filter_map(move |local| {
                let ty = entry.mcx.mir.local_decls[local].ty;
                Input::new(entry, ty).map(|input| (local, input))
            })
            .map(move |(local, input)| {
                (local, *self.global_vars.get(&input).expect("Entry compute"))
            })
    }
}

#[derive(Debug, Clone)]
pub enum Layout<'tcx> {
    Single(SingleLayout<'tcx>),
    Composition(Vec<Box<Layout<'tcx>>>),
}

impl<'tcx> Layout<'tcx> {
    pub fn size(&self) -> usize {
        self.size_impl().0
    }

    pub fn offsets(&self) -> Vec<usize> {
        self.size_impl().1
    }

    fn size_impl(&self) -> (usize, Vec<usize>) {
        match *self {
            Layout::Single(single) => (single.size, Vec::new()),
            Layout::Composition(ref comp) => {
                let mut offset = 0;
                let offsets = comp.iter()
                    .map(|layout| {
                        let align = layout.align();
                        let aligned_offset = (align - (offset % align)) % align + offset;
                        offset = aligned_offset + layout.size();
                        aligned_offset
                    })
                    .collect_vec();
                (offset, offsets)
            }
        }
    }

    pub fn align(&self) -> usize {
        match *self {
            Layout::Single(single) => single.align,
            Layout::Composition(ref comp) => {
                let max_align = comp.iter().map(|layout| layout.align()).max().unwrap_or(0);
                max_align.max(16).min(16)
            }
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SingleLayout<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub size: usize,
    pub align: usize,
}
use syntax::ast;
pub fn std140_layout<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ty: ty::Ty<'tcx>,
) -> Option<Layout<'tcx>> {
    if let Some(intrinsic) = IntrinsicType::from_ty(tcx, ty) {
        match intrinsic {
            IntrinsicType::TyVec(ty_vec) => {
                let multiplier = match ty_vec.dim {
                    2 => 2,
                    3 => 4,
                    4 => 4,
                    _ => unreachable!(),
                };
                return std140_layout(tcx, ty_vec.ty).map(|inner_layout| {
                    let single = SingleLayout {
                        ty,
                        size: inner_layout.size() * ty_vec.dim,
                        align: inner_layout.align() * multiplier,
                    };
                    Layout::Single(single)
                });
            }
        }
    }

    match ty.sty {
        TypeVariants::TyFloat(float_ty) => {
            assert!(float_ty == ast::FloatTy::F32, "F64 is not supported");
            let single = SingleLayout {
                ty,
                size: 4,
                align: 4,
            };
            Some(Layout::Single(single))
        }
        TypeVariants::TyAdt(adt, substs) => {
            if adt.is_struct() {
                let comp = adt.all_fields()
                    .map(|field| field.ty(tcx, substs))
                    .filter(|ty| !ty.is_phantom_data())
                    .map(|ty| {
                        Box::new(std140_layout(tcx, ty).expect("No layout inside Composition"))
                    })
                    .collect_vec();
                Some(Layout::Composition(comp))
            } else {
                None
            }
        }
        _ => None,
    }
}
impl<'tcx> Entry<'tcx, Descriptor<'tcx>> {
    pub fn descriptor<'a>(
        entry_points: &[EntryPoint<'a, 'tcx>],
        stx: &mut CodegenCx<'a, 'tcx>,
    ) -> Self {
        let set: HashSet<_> = entry_points
            .iter()
            .flat_map(EntryPoint::descriptor_iter)
            .collect();
        Self::create(set, stx, spirv::StorageClass::Uniform)
    }

    fn variable_iter<'borrow, 'a>(
        &'borrow self,
        entry: &'borrow EntryPoint<'a, 'tcx>,
    ) -> impl Iterator<Item = (mir::Local, GlobalVar<'tcx>)> + 'borrow {
        entry
            .mcx
            .mir
            .args_iter()
            .filter_map(move |local| {
                let ty = entry.mcx.mir.local_decls[local].ty;
                Descriptor::new(entry.mcx.tcx, ty).map(|input| (local, input))
            })
            .map(move |(local, descriptor)| {
                (
                    local,
                    *self.global_vars.get(&descriptor).expect("Entry compute"),
                )
            })
    }
}

impl<'tcx, T> Entry<'tcx, T>
where
    T: Global<'tcx>,
{
    pub fn create<'a>(
        set: HashSet<T>,
        stx: &mut CodegenCx<'a, 'tcx>,
        storage_class: spirv::StorageClass,
    ) -> Self {
        let global_vars: HashMap<_, _> = set.into_iter()
            .map(|global| {
                let spirv_ty = stx.to_ty_as_ptr(global.ty(), storage_class);
                let var = stx.builder
                    .variable(spirv_ty.word, None, storage_class, None);
                let global_var = GlobalVar {
                    var,
                    ty: global.ty(),
                    storage_class: storage_class,
                    location: 0,
                };
                (global, global_var)
            })
            .collect();
        Entry { global_vars }
    }
}

impl<'tcx> Entry<'tcx, Output<'tcx>> {
    pub fn output<'a>(
        entry_points: &[EntryPoint<'a, 'tcx>],
        stx: &mut CodegenCx<'a, 'tcx>,
    ) -> Self {
        let set: HashSet<_> = entry_points
            .iter()
            .flat_map(EntryPoint::output_iter)
            .collect();
        Self::create(set, stx, spirv::StorageClass::Output)
    }

    fn variable_iter<'borrow, 'a>(
        &'borrow self,
        entry: &'borrow EntryPoint<'a, 'tcx>,
    ) -> impl Iterator<Item = (mir::Local, GlobalVar<'tcx>)> + 'borrow {
        let ty = entry.mcx.mir.return_ty();
        let output = Output::new(entry.mcx.tcx, ty).expect("Should be output");
        Some((
            mir::Local::new(0),
            *self.global_vars.get(&output).expect("Entry compute"),
        )).into_iter()
    }
}

fn intrinsic_fn(attrs: &[syntax::ast::Attribute]) -> Option<IntrinsicFn> {
    extract_attr(attrs, "spirv", |s| match s {
        "dot" => Some(IntrinsicFn::Dot),
        _ => None,
    }).get(0)
        .map(|&i| i)
}

#[derive(Debug, Copy, Clone)]
pub enum Intrinsic {
    GlslExt(spirv::Word),
    Abort,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum InstanceType {
    Entry(IntrinsicEntry),
    Fn,
}

pub struct FunctionCx<'b, 'a: 'b, 'tcx: 'a> {
    current_table: Vec<&'a TypeckTables<'tcx>>,
    pub mcx: MirContext<'a, 'tcx>,
    pub scx: &'b mut CodegenCx<'a, 'tcx>,
    pub constants: HashMap<mir::Constant<'tcx>, Variable<'tcx>>,
    pub label_blocks: HashMap<mir::BasicBlock, Label>,
    pub vars: HashMap<mir::Local, Variable<'tcx>>,
    pub references: HashMap<mir::Place<'tcx>, mir::Place<'tcx>>,
    pub instance_ty: InstanceType,
}

#[derive(Debug, Copy, Clone)]
pub struct TyVec<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub dim: usize,
}

#[derive(Debug, Copy, Clone)]
pub enum IntrinsicType<'tcx> {
    TyVec(TyVec<'tcx>),
}
impl<'tcx> IntrinsicType<'tcx> {
    pub fn from_ty<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        TyVec::from_ty(tcx, ty).map(IntrinsicType::TyVec)
    }
}
impl<'tcx> TyVec<'tcx> {
    pub fn from_ty<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        if let TypeVariants::TyAdt(adt, substs) = ty.sty {
            let attrs = tcx.get_attrs(adt.did);
            let dim = extract_attr(&attrs, "spirv", |s| match s {
                "Vec2" => Some(2),
                "Vec3" => Some(3),
                "Vec4" => Some(4),
                _ => None,
            }).get(0)
                .map(|&i| i)?;
            assert!(adt.is_struct(), "A Vec should be a struct");
            let field = adt.all_fields()
                .nth(0)
                .expect("A Vec should have at least one field");
            let field_ty = field.ty(tcx, substs);
            Some(TyVec { ty: field_ty, dim })
        } else {
            None
        }
    }
}
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum IntrinsicEntry {
    Vertex,
    Fragment,
}

pub fn extract_attr_impl<R, F>(
    meta_item: &syntax::ast::MetaItem,
    keywords: &[&str],
    f: &F,
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
// TODO: Better API
pub fn extract_attr<R, F>(attrs: &[syntax::ast::Attribute], keyword: &str, f: F) -> Vec<R>
where
    F: Fn(&str) -> Option<R>,
{
    attrs
        .iter()
        .filter_map(|attr| {
            attr.meta()
                .and_then(|meta| extract_attr_impl(&meta, &[keyword], &f))
        })
        .collect::<Vec<_>>()
}

pub enum FunctionCall {
    Function(Function),
    Intrinsic(Intrinsic),
}

pub fn trans_spirv<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, items: &'a FxHashSet<MonoItem<'tcx>>) {
    //struct_span_err!(tcx.sess, DUMMY_SP, E1337, "Test not allowed").emit();

    let mut ctx = CodegenCx::new(tcx);
    items
        .iter()
        .filter_map(|item| {
            if let &MonoItem::Fn(ref instance) = item {
                return Some(instance);
            }
            None
        })
        .for_each(|instance| {
            if tcx.is_foreign_item(instance.def_id()) {
                let intrinsic_name = &*tcx.item_name(instance.def_id());
                use spirv::GLOp::*;
                let id = match intrinsic_name {
                    "sqrtf32" => Some(Sqrt),
                    "sinf32" => Some(Sin),
                    "cosf32" => Some(Cos),
                    "absf32" => Some(FAbs),
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
            }
        });

    let instances: Vec<MirContext> = items
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
    // write_dot(&instances);
    instances.iter().for_each(|mcx| {
        //println!("{:?}", mcx.def_id);
        // println!("{}", mcx.tcx.def_symbol_name(mcx.def_id));
        // println!("{}", mcx.tcx.item_name(mcx.def_id));
        //println!("{:#?}", mcx.mir);
    });

    if TyErrorVisitor::has_error(&instances) {
        return;
    }
    for mcx in &instances {
        ctx.forward_fns
            .insert((mcx.def_id, mcx.substs), Function(ctx.builder.id()));
    }
    let (entry_instances, fn_instances): (Vec<_>, Vec<_>) =
        instances.iter().partition_map(|&mcx| {
            let attrs = tcx.get_attrs(mcx.def_id);
            let entry = extract_attr(&attrs, "spirv", |s| match s {
                "vertex" => Some(IntrinsicEntry::Vertex),
                "fragment" => Some(IntrinsicEntry::Fragment),
                _ => None,
            }).iter()
                .nth(0)
                .map(|&entry_type| EntryPoint { mcx, entry_type });
            if let Some(entry_point) = entry {
                Either::Left(entry_point)
            } else {
                Either::Right(mcx)
            }
        });
    let entry_input = Entry::input(&entry_instances, &mut ctx);
    let entry_output = Entry::output(&entry_instances, &mut ctx);
    let entry_descriptor = Entry::descriptor(&entry_instances, &mut ctx);

    entry_instances.iter().for_each(|e| {
        FunctionCx::trans_entry(&e, &entry_input, &entry_output, &entry_descriptor, &mut ctx);
    });
    fn_instances
        .iter()
        //.filter(|mcx| mcx.def_id != entry_fn && tcx.lang_items().start_fn() != Some(mcx.def_id))
        .for_each(|&mcx| {
            FunctionCx::trans_fn(mcx, &mut ctx);
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
                match proj.elem {
                    mir::ProjectionElem::Field(field, _) => {
                        indices.push(field.index());
                        access_chain_indices(&proj.base, indices)
                    }
                    mir::ProjectionElem::Downcast(_, id) => {
                        indices.push(id);
                        access_chain_indices(&proj.base, indices)
                    }
                    // TODO: Is this actually correct?
                    _ => access_chain_indices(&proj.base, indices),
                }
            } else {
                (lvalue, indices)
            }
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

fn write_dot(mcxs: &[MirContext]) {
    use std::fs::File;
    use rustc_mir::util::write_mir_fn_graphviz;
    let path = std::env::current_dir().expect("dir").join("shader.dot");
    let mut file = File::create(&path).expect("file");
    for mcx in mcxs {
        write_mir_fn_graphviz(mcx.tcx, mcx.def_id, mcx.mir, &mut file);
    }
}

use rustc::middle::const_val::ConstVal;
impl<'b, 'a, 'tcx> FunctionCx<'b, 'a, 'tcx> {
    pub fn load_operand<'r>(&mut self, operand: &'r mir::Operand<'tcx>) -> Operand<'tcx> {
        let mir = self.mcx.mir;
        let mcx = self.mcx;
        let local_decls = &mir.local_decls;
        let ty = operand.ty(local_decls, self.mcx.tcx);
        let ty = mcx.monomorphize(&ty);
        let spirv_ty = self.to_ty_fn(ty);
        match operand {
            &mir::Operand::Copy(ref place) | &mir::Operand::Move(ref place) => {
                if let Some(ref_place) = self.references.get(place).map(|p| p.clone()) {
                    let access_chain = AccessChain::compute(&ref_place);
                    assert!(
                        access_chain.indices.len() == 0,
                        "Access chain should be empty"
                    );
                    let spirv_var = *self.vars.get(&access_chain.base).expect("Local");
                    Operand::new(ty, OperandVariant::Variable(spirv_var))
                } else {
                    let access_chain = AccessChain::compute(place);
                    let spirv_var = *self.vars.get(&access_chain.base).expect("Local");
                    if access_chain.indices.is_empty() {
                        Operand::new(ty, OperandVariant::Variable(spirv_var))
                    } else {
                        let spirv_ty_ptr = self.scx.to_ty_as_ptr(ty, spirv_var.storage_class);
                        let indices: Vec<_> = access_chain
                            .indices
                            .iter()
                            .map(|&i| self.scx.constant_u32(mcx, i as u32).word)
                            .collect();
                        let access = self.scx
                            .builder
                            .access_chain(spirv_ty_ptr.word, None, spirv_var.word, &indices)
                            .expect("access_chain");
                        let load = self.scx
                            .builder
                            .load(spirv_ty.word, None, access, None, &[])
                            .expect("load");
                        Operand::new(ty, OperandVariant::Value(Value::new(load)))
                    }
                }
            }
            &mir::Operand::Constant(ref constant) => match constant.literal {
                mir::Literal::Value { ref value } => {
                    let expr = match value.val {
                        ConstVal::Float(f) => {
                            let val = ConstValue::Float(f);
                            self.scx.constant(mcx, val)
                        }
                        ConstVal::Integral(int) => {
                            let val = ConstValue::Integer(int);
                            self.scx.constant(mcx, val)
                        }
                        ConstVal::Bool(b) => {
                            let val = ConstValue::Bool(b);
                            self.scx.constant(mcx, val)
                        }
                        ref rest => unimplemented!("{:?}", rest),
                    };
                    Operand::new(ty, OperandVariant::Value(expr))
                }
                ref rest => unimplemented!("{:?}", rest),
            },
        }
    }
    pub fn trans_fn(mcx: MirContext<'a, 'tcx>, scx: &mut CodegenCx<'a, 'tcx>) {
        use mir::visit::Visitor;
        let ret_ty = mcx.monomorphize(&mcx.mir.return_ty());
        let ret_ty_spirv = scx.to_ty_fn(ret_ty);
        let def_id = mcx.def_id;

        let args_ty: Vec<_> = mcx.mir
            .args_iter()
            .map(|l| mcx.monomorphize(&mcx.mir.local_decls[l].ty))
            .collect();
        let fn_sig = scx.tcx.mk_fn_sig(
            args_ty.into_iter(),
            ret_ty,
            false,
            hir::Unsafety::Normal,
            syntax::abi::Abi::Rust,
        );
        let fn_ty = scx.tcx.mk_fn_ptr(Binder(fn_sig));
        let fn_ty_spirv = scx.to_ty_fn(fn_ty);

        let forward_fn = scx.forward_fns
            .get(&(def_id, mcx.substs))
            .map(|f| f.0)
            .expect("forward");
        let spirv_function = scx.builder
            .begin_function(
                ret_ty_spirv.word,
                Some(forward_fn),
                spirv::FunctionControl::empty(),
                fn_ty_spirv.word,
            )
            .expect("begin fn");

        scx.name_from_def_id(def_id, spirv_function);
        let params: Vec<_> = mcx.mir
            .args_iter()
            .map(|local_arg| {
                let local_decl = &mcx.mir.local_decls[local_arg];
                let local_ty = mcx.monomorphize(&local_decl.ty);
                Param::alloca(scx, local_ty)
            })
            .collect();
        scx.builder.begin_basic_block(None).expect("block");
        let mut args_map: HashMap<_, _> = params
            .into_iter()
            .enumerate()
            .map(|(index, param)| {
                let local_arg = mir::Local::new(index + 1);
                let local_decl = &mcx.mir.local_decls[local_arg];
                let local_ty = mcx.monomorphize(&local_decl.ty);
                (
                    local_arg,
                    param.to_variable(scx, spirv::StorageClass::Function),
                )
            })
            .collect();
        {
            use rustc_data_structures::indexed_vec::Idx;
            let local = mir::Local::new(0);
            let local_decl = &mcx.mir.local_decls[local];
            let ty = mcx.monomorphize(&local_decl.ty);
            let variable = Variable::alloca(scx, ty, spirv::StorageClass::Function);
            // TODO DEBUG
            //scx.name_from_str("retvar", spirv_var);
            args_map.insert(local, variable);
        }
        FunctionCx::new(InstanceType::Fn, mcx, args_map, scx).visit_mir(mcx.mir);
    }
    pub fn trans_entry(
        entry_point: &EntryPoint<'a, 'tcx>,
        entry_input: &Entry<'tcx, Input<'tcx>>,
        entry_output: &Entry<'tcx, Output<'tcx>>,
        entry_descriptor: &Entry<'tcx, Descriptor<'tcx>>,
        scx: &mut CodegenCx<'a, 'tcx>,
    ) {
        use mir::visit::Visitor;
        let def_id = entry_point.mcx.def_id;
        let mir = entry_point.mcx.mir;
        // TODO: Fix properly
        if entry_point.entry_type == IntrinsicEntry::Vertex {
            let first_local = mir::Local::new(1);
            let per_vertex = scx.get_per_vertex(mir.local_decls[first_local].ty);
        }
        if entry_point.entry_type == IntrinsicEntry::Fragment {
            let first_local = mir::Local::new(1);
            let per_fragment = scx.get_per_fragment(mir.local_decls[first_local].ty);
        }
        let void = scx.tcx.mk_nil();
        let fn_sig = scx.tcx.mk_fn_sig(
            [].into_iter(),
            &void,
            false,
            hir::Unsafety::Normal,
            syntax::abi::Abi::Rust,
        );
        let void_spirv = scx.to_ty_fn(void);
        let fn_ty = scx.tcx.mk_fn_ptr(Binder(fn_sig));
        let fn_ty_spirv = scx.to_ty_fn(fn_ty);
        let forward_fn = scx.forward_fns
            .get(&(def_id, entry_point.mcx.substs))
            .map(|f| f.0)
            .expect("forward");
        let spirv_function = scx.builder
            .begin_function(
                void_spirv.word,
                Some(forward_fn),
                spirv::FunctionControl::empty(),
                fn_ty_spirv.word,
            )
            .expect("begin fn");
        scx.builder.begin_basic_block(None).expect("block");
        entry_point.descriptor_iter().for_each(|input| {
            if let TypeVariants::TyAdt(adt, substs) = input.ty.sty {
                let ty = adt.all_fields()
                    .nth(0)
                    .expect("field")
                    .ty(entry_point.mcx.tcx, substs);
                let layout = ::std140_layout(entry_point.mcx.tcx, ty);
            }
        });
        let inputs_iter = entry_input.variable_iter(&entry_point);
        let output_iter = entry_output.variable_iter(&entry_point);
        let descriptor_iter = entry_descriptor.variable_iter(&entry_point);
        entry_descriptor
            .global_vars
            .iter()
            .for_each(|(descriptor, global)| {
                scx.builder.decorate(
                    global.var,
                    spirv::Decoration::DescriptorSet,
                    &[rspirv::mr::Operand::LiteralInt32(descriptor.set)],
                );
                scx.builder.decorate(
                    global.var,
                    spirv::Decoration::Binding,
                    &[rspirv::mr::Operand::LiteralInt32(descriptor.binding)],
                );
            });
        let mut variable_map: HashMap<mir::Local, Variable<'tcx>> = inputs_iter
            .chain(output_iter)
            .chain(descriptor_iter)
            .map(|(local, global)| {
                (
                    local,
                    Variable {
                        word: global.var,
                        ty: global.ty,
                        storage_class: global.storage_class,
                    },
                )
            })
            .collect();

        // Only add the per vertex variable in the vertex shader
        if entry_point.entry_type == IntrinsicEntry::Vertex {
            let first_local = mir::Local::new(1);
            let per_vertex = scx.get_per_vertex(mir.local_decls[first_local].ty);
            variable_map.insert(first_local, per_vertex);
        }
        if entry_point.entry_type == IntrinsicEntry::Fragment {
            let first_local = mir::Local::new(1);
            let per_fragment = scx.get_per_fragment(mir.local_decls[first_local].ty);
            variable_map.insert(first_local, per_fragment);
        }
        let outputs = entry_output
            .variable_iter(entry_point)
            .map(|(_, gv)| gv)
            .collect_vec();
        let output_var = outputs[0];
        // Insert the return variable
        variable_map.insert(
            mir::Local::new(0),
            Variable {
                word: output_var.var,
                ty: mir.return_ty(),
                storage_class: output_var.storage_class,
            },
        );

        FunctionCx::new(
            InstanceType::Entry(entry_point.entry_type),
            entry_point.mcx,
            variable_map,
            scx,
        ).visit_mir(entry_point.mcx.mir);
        let mut inputs_raw = entry_input
            .variable_iter(entry_point)
            .map(|(local, gv)| gv.var)
            .collect_vec();
        inputs_raw.extend(outputs.iter().map(|gv| gv.var));
        let name = entry_point.mcx.tcx.item_name(def_id);
        let model = match entry_point.entry_type {
            IntrinsicEntry::Vertex => spirv::ExecutionModel::Vertex,
            IntrinsicEntry::Fragment => spirv::ExecutionModel::Fragment,
        };
        scx.builder
            .entry_point(model, spirv_function, name.as_ref(), inputs_raw);
        scx.builder
            .execution_mode(spirv_function, spirv::ExecutionMode::OriginUpperLeft, &[]);
    }
    pub fn to_ty(&mut self, ty: ty::Ty<'tcx>, storage_class: spirv::StorageClass) -> Ty<'tcx> {
        let ty = self.mcx.monomorphize(&ty);
        self.scx.to_ty(ty, storage_class)
    }
    pub fn to_ty_as_ptr(
        &mut self,
        ty: ty::Ty<'tcx>,
        storage_class: spirv::StorageClass,
    ) -> Ty<'tcx> {
        let ty = self.mcx.monomorphize(&ty);
        self.scx.to_ty_as_ptr(ty, storage_class)
    }
    pub fn to_ty_fn(&mut self, ty: ty::Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.mcx.monomorphize(&ty);
        self.scx.to_ty(ty, spirv::StorageClass::Function)
    }
    pub fn to_ty_as_ptr_fn(&mut self, ty: ty::Ty<'tcx>) -> Ty<'tcx> {
        let ty = self.mcx.monomorphize(&ty);
        self.scx.to_ty_as_ptr(ty, spirv::StorageClass::Function)
    }
    pub fn constant(&mut self, val: ConstValue) -> Value {
        self.scx.constant(self.mcx, val)
    }
    pub fn constant_f32(&mut self, value: f32) -> Value {
        self.scx.constant_f32(self.mcx, value)
    }
    pub fn constant_u32(&mut self, value: u32) -> Value {
        self.scx.constant_u32(self.mcx, value)
    }

    pub fn get_table(&self) -> &'a TypeckTables<'tcx> {
        self.current_table.last().expect("no table yet")
    }
    pub fn new(
        instance_ty: InstanceType,
        mcx: MirContext<'a, 'tcx>,
        mut variable_map: HashMap<mir::Local, Variable<'tcx>>,
        scx: &'b mut CodegenCx<'a, 'tcx>,
    ) -> Self {
        let label_blocks: HashMap<_, _> = mcx.mir
            .basic_blocks()
            .iter_enumerated()
            .map(|(block, _)| (block, Label(scx.builder.id())))
            .collect();
        let mut local_vars: HashMap<_, _> = mcx.mir
            .vars_and_temps_iter()
            .filter_map(|local_var| {
                // Don't generate variables for ptrs
                let local_decl = &mcx.mir.local_decls[local_var];
                let local_ty = mcx.monomorphize(&local_decl.ty);
                if is_ptr(local_ty) {
                    return None;
                }
                let local_ty = remove_ptr_ty(local_ty);
                let variable = Variable::alloca(scx, local_ty, spirv::StorageClass::Function);
                Some((local_var, variable))
            })
            .collect();
        {
            let spirv_label = label_blocks
                .get(&mir::BasicBlock::new(0))
                .expect("No first label");
            scx.builder.branch(spirv_label.0).expect("branch");
        }
        variable_map.extend(local_vars.into_iter());
        let mut visitor = FunctionCx {
            instance_ty,
            scx,
            current_table: Vec::new(),
            mcx,
            constants: HashMap::new(),
            label_blocks,
            vars: variable_map,
            references: HashMap::new(),
        };
        visitor
    }
}
fn is_ptr(ty: ty::Ty) -> bool {
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
    pub discr_ty: ty::Ty<'tcx>,
    pub index: usize,
}

impl<'tcx> Enum<'tcx> {
    pub fn from_ty<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Enum<'tcx>> {
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
pub trait Monomorphize<'tcx> {
    fn mono<'a>(self, mtx: MirContext<'a, 'tcx>) -> ty::Ty<'tcx>;
}

impl<'tcx> Monomorphize<'tcx> for ty::Ty<'tcx> {
    fn mono<'a>(self, mtx: MirContext<'a, 'tcx>) -> ty::Ty<'tcx> {
        mtx.monomorphize(&self)
    }
}

impl<'b, 'a, 'tcx> rustc::mir::visit::Visitor<'tcx> for FunctionCx<'b, 'a, 'tcx> {
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
    fn visit_ty(&mut self, ty: &ty::Ty<'tcx>, _: TyContext) {
        self.super_ty(ty);
        //println!("{:?}", ty);
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
        use mir::traversal::reverse_postorder;
        let order = reverse_postorder(mir);
        for (bb, data) in order {
            self.visit_basic_block_data(bb, &data);
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
    }
    fn visit_assign(
        &mut self,
        block: mir::BasicBlock,
        lvalue: &mir::Place<'tcx>,
        rvalue: &mir::Rvalue<'tcx>,
        location: mir::Location,
    ) {
        self.super_assign(block, lvalue, rvalue, location);
        let lvalue_ty = lvalue
            .ty(&self.mcx.mir.local_decls, self.scx.tcx)
            .to_ty(self.scx.tcx);
        let lvalue_ty = self.mcx.monomorphize(&lvalue_ty);
        if lvalue_ty.is_phantom_data() || lvalue_ty.is_nil() {
            return;
        }
        let ty = rvalue.ty(&self.mcx.mir.local_decls, self.scx.tcx);
        if let TypeVariants::TyTuple(ref slice, _) = ty.sty {
            if slice.len() == 0 {
                return;
            }
        }
        let ty = self.mcx.monomorphize(&ty);
        let spirv_ty = self.to_ty_fn(ty);
        let lvalue_ty_spirv = self.to_ty_fn(lvalue_ty);
        if let &mir::Rvalue::Ref(_, _, ref place) = rvalue {
            assert!(is_ptr(lvalue_ty), "LValue should be a ptr");
            self.references.insert(lvalue.clone(), place.clone());
            // Don't do anything
            return;
        }
        let expr = match rvalue {
            &mir::Rvalue::BinaryOp(op, ref l, ref r)
            | &mir::Rvalue::CheckedBinaryOp(op, ref l, ref r) => self.binary_op(spirv_ty, op, l, r),
            &mir::Rvalue::Use(ref operand) => {
                let load = self.load_operand(operand).load(self.scx);
                let expr = Value::new(load.word);
                expr
            }

            &mir::Rvalue::Aggregate(_, ref operands) => {
                // If there are no operands, then it should be 0 sized and we can
                // abort.
                if operands.is_empty() {
                    return;
                }

                let spirv_operands: Vec<_> = operands
                    .iter()
                    .filter_map(|op| {
                        let ty = op.ty(&self.mcx.mir.local_decls, self.mcx.tcx);
                        let ty = self.mcx.monomorphize(&ty);
                        if ty.is_phantom_data() {
                            None
                        } else {
                            let spirv_ty = self.to_ty_fn(ty);
                            Some(self.load_operand(op).load(self.scx).word)
                        }
                    })
                    .collect();
                Value::new(
                    self.scx
                        .builder
                        .composite_construct(lvalue_ty_spirv.word, None, &spirv_operands)
                        .expect("composite"),
                )
            }
            &mir::Rvalue::Discriminant(ref lvalue) => {
                let local = match lvalue {
                    &mir::Place::Local(local) => local,
                    _ => panic!("Should be local"),
                };
                let var = *self.vars.get(&local).expect("local");
                let ty = self.mcx.mir.local_decls[local].ty;
                let enum_data = Enum::from_ty(self.mcx.tcx, ty).expect("enum");
                let discr_ty = self.mcx.monomorphize(&enum_data.discr_ty);
                let discr_ty_spirv = self.to_ty_fn(discr_ty);
                let discr_ty_spirv_ptr = self.to_ty_as_ptr_fn(discr_ty);
                let index = self.constant_u32(enum_data.index as u32).word;
                let access = self.scx
                    .builder
                    .access_chain(discr_ty_spirv_ptr.word, None, var.word, &[index])
                    .expect("access");
                let load = self.scx
                    .builder
                    .load(discr_ty_spirv.word, None, access, None, &[])
                    .expect("load");
                let lvalue_ty = self.mcx.monomorphize(&lvalue_ty);
                let target_ty_spirv = self.to_ty_fn(lvalue_ty);
                let cast = self.scx
                    .builder
                    .bitcast(target_ty_spirv.word, None, load)
                    .expect("bitcast");

                Value::new(cast)
            }
            &mir::Rvalue::Cast(_, ref op, ty) => {
                let op_ty = op.ty(&self.mcx.mir.local_decls, self.mcx.tcx);
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
            let var = *self.vars
                .get(&access_chain.base)
                .expect("access chain local");
            // TODO: Better way to get the correct storage class
            let spirv_ty_ptr = self.to_ty_as_ptr(ty, var.storage_class);
            let indices: Vec<_> = access_chain
                .indices
                .iter()
                .map(|&i| self.constant_u32(i as u32).word)
                .collect();
            let access = self.scx
                .builder
                .access_chain(spirv_ty_ptr.word, None, spirv_var.word, &indices)
                .expect("access_chain");
            access
        };
        self.scx
            .builder
            .store(store, expr.word, None, &[])
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
                // If we are inside an entry, we just return void
                if self.instance_ty != InstanceType::Fn {
                    return self.scx.builder.ret().expect("ret");
                } else {
                    match mir.return_ty().sty {
                        TypeVariants::TyTuple(ref slice, _) if slice.len() == 0 => {
                            self.scx.builder.ret().expect("ret");
                        }
                        _ => {
                            use rustc_data_structures::indexed_vec::Idx;
                            let ty = self.mcx.monomorphize(&mir.return_ty());
                            let spirv_ty = { self.to_ty_fn(ty) };
                            let var = self.vars.get(&mir::Local::new(0)).unwrap();
                            let load = self.scx
                                .builder
                                .load(spirv_ty.word, None, var.word, None, &[])
                                .expect("load");
                            self.scx.builder.ret_value(load).expect("ret value");
                        }
                    };
                }
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
                    // TODO bitcast api
                    let load = self.load_operand(discr).load(self.scx).word;
                    let target_ty = self.mcx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
                    let target_ty_spirv = self.to_ty_fn(target_ty);
                    self.scx
                        .builder
                        .bitcast(target_ty_spirv.word, None, load)
                        .expect("bitcast")
                } else {
                    self.load_operand(discr).load(self.scx).word
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
                use syntax::abi::Abi;
                let local_decls = &self.mcx.mir.local_decls;
                match func {
                    &mir::Operand::Constant(ref constant) => {
                        let fn_sig = constant.ty.fn_sig(self.scx.tcx);
                        let abi = fn_sig.abi();
                        let ret_ty_binder = fn_sig.output();
                        let ret_ty = ret_ty_binder.skip_binder();
                        let ret_ty = self.mcx.monomorphize(ret_ty);
                        let spirv_ty = self.to_ty_fn(ret_ty);
                        if let mir::Literal::Value { ref value } = constant.literal {
                            use rustc::middle::const_val::ConstVal;
                            if let ConstVal::Function(def_id, ref substs) = value.val {
                                let mono_substs = self.mcx.monomorphize(substs);
                                let resolve_instance = Instance::resolve(
                                    self.scx.tcx,
                                    ParamEnv::empty(rustc::traits::Reveal::All),
                                    def_id,
                                    &mono_substs,
                                ).expect("resolve instance call");
                                let fn_call = self.scx
                                    .get_function_call(
                                        resolve_instance.def_id(),
                                        resolve_instance.substs,
                                    )
                                    .expect("function call");
                                let args_ty = args.iter()
                                    .map(|arg| {
                                        let ty = arg.ty(local_decls, self.mcx.tcx);
                                        ty
                                    })
                                    .collect_vec();
                                // Split the rust-call tupled arguments off.
                                let (first_args, untuple) =
                                    if abi == Abi::RustCall && !args.is_empty() {
                                        let (tup, args) = args.split_last().unwrap();
                                        (args, Some(tup))
                                    } else {
                                        (&args[..], None)
                                    };
                                let mut arg_operand_loads: Vec<_> = first_args
                                    .iter()
                                    .map(|operand| {
                                        let ty = operand.ty(local_decls, self.mcx.tcx);
                                        let ty = self.mcx.monomorphize(&ty);
                                        let spirv_operand = self.load_operand(operand);
                                        if is_ptr(ty) {
                                            spirv_operand
                                                .to_variable()
                                                .expect("should be a variable")
                                                .word
                                        } else {
                                            spirv_operand.load(self.scx).word
                                        }
                                    })
                                    .collect();
                                if let Some(tup) = untuple {
                                    let ty = tup.ty(local_decls, self.mcx.tcx);
                                    let ty = self.mcx.monomorphize(&ty);
                                    let spirv_ty_ptr = self.to_ty_as_ptr_fn(ty);
                                    let tuple_var = self.load_operand(tup)
                                        .to_variable()
                                        .expect("Should be a variable");
                                    match ty.sty {
                                        TypeVariants::TyTuple(slice, b) => {
                                            let tuple_iter =
                                                slice.iter().enumerate().map(|(idx, field_ty)| {
                                                    let field_ty_spv = self.to_ty_fn(field_ty);
                                                    let field_ty_ptr_spv =
                                                        self.to_ty_as_ptr_fn(field_ty);
                                                    let spirv_idx =
                                                        self.constant_u32(idx as u32).word;
                                                    let field = self.scx
                                                        .builder
                                                        .access_chain(
                                                            field_ty_ptr_spv.word,
                                                            None,
                                                            tuple_var.word,
                                                            &[spirv_idx],
                                                        )
                                                        .expect("access chain");
                                                    self.scx
                                                        .builder
                                                        .load(
                                                            field_ty_spv.word,
                                                            None,
                                                            field,
                                                            None,
                                                            &[],
                                                        )
                                                        .expect("load")
                                                });
                                            arg_operand_loads.extend(tuple_iter);
                                        }
                                        _ => panic!("tup"),
                                    }
                                }
                                // println!("{:?}", args_ty);
                                // println!("{:?}", arg_operand_loads);
                                let spirv_fn_call = match fn_call {
                                    FunctionCall::Function(spirv_fn) => {
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
                                    FunctionCall::Intrinsic(intrinsic) => match intrinsic {
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
                                // only write op store if the result is not nil
                                if !ret_ty.is_nil() {
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
            &mir::TerminatorKind::Drop {
                ref location,
                target,
                ..
            } => {
                let target_label = self.label_blocks.get(&target).expect("no label");
                self.scx.builder.branch(target_label.0).expect("label");
            }
            &mir::TerminatorKind::Resume | &mir::TerminatorKind::Unreachable => {
                self.scx.builder.unreachable();
            }
            rest => unimplemented!("{:?}", rest),
        };
    }
}

#[derive(Debug, Clone)]
pub enum SpirvRvalue {
}

impl<'b, 'a, 'tcx> FunctionCx<'b, 'a, 'tcx> {
    pub fn binary_op(
        &mut self,
        spirv_ty: Ty,
        op: mir::BinOp,
        l: &mir::Operand<'tcx>,
        r: &mir::Operand<'tcx>,
    ) -> Value {
        // TODO: Different types
        let left = self.load_operand(l).load(self.scx).word;
        let right = self.load_operand(r).load(self.scx).word;
        // TODO: Impl ops
        match op {
            mir::BinOp::Mul => {
                let add = self.scx
                    .builder
                    .fmul(spirv_ty.word, None, left, right)
                    .expect("fmul");
                Value::new(add)
            }
            mir::BinOp::Add => {
                let add = self.scx
                    .builder
                    .fadd(spirv_ty.word, None, left, right)
                    .expect("fadd");
                Value::new(add)
            }
            mir::BinOp::Sub => {
                let add = self.scx
                    .builder
                    .fsub(spirv_ty.word, None, left, right)
                    .expect("fsub");
                Value::new(add)
            }
            mir::BinOp::Div => {
                let add = self.scx
                    .builder
                    .fdiv(spirv_ty.word, None, left, right)
                    .expect("fsub");
                Value::new(add)
            }
            mir::BinOp::Gt => {
                let gt = self.scx
                    .builder
                    .ugreater_than(spirv_ty.word, None, left, right)
                    .expect("g");
                Value::new(gt)
            }
            mir::BinOp::Shl => {
                let shl = self.scx
                    .builder
                    .shift_left_logical(spirv_ty.word, None, left, right)
                    .expect("shl");
                Value::new(shl)
            }
            mir::BinOp::BitOr => {
                let bit_or = self.scx
                    .builder
                    .bitwise_or(spirv_ty.word, None, left, right)
                    .expect("bitwise or");
                Value::new(bit_or)
            }
            rest => unimplemented!("{:?}", rest),
        }
    }
}
