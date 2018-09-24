#![feature(rustc_private)]
#![feature(box_syntax)]
#![feature(try_from)]
#![feature(rustc_diagnostic_macros)]

extern crate byteorder;
extern crate petgraph;

extern crate arena;
extern crate env_logger;
extern crate getopts;
extern crate itertools;
extern crate log;
extern crate rspirv;
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
extern crate rustc_target;
extern crate rustc_typeck;
extern crate spirv_headers as spirv;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
pub mod trans;
use rustc::mir::mono::MonoItem;
use rustc::mir::visit::{TyContext, Visitor};
use rustc::ty::{Binder, Instance, TyCtxt, TypeVariants, TypeckTables};
use rustc::{hir, mir};
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::Idx;
use rustc_target::spec::abi::Abi;
pub mod collector;
pub mod context;
pub mod graph;
pub mod iterate;
pub mod typ;
use self::context::{CodegenCx, MirContext, SpirvMir};
use self::typ::*;
use itertools::{Either, Itertools};
use rustc::ty;
use rustc::ty::subst::Substs;
use std::collections::HashMap;
use std::path::Path;
#[derive(Copy, Clone, Debug)]
pub enum IntrinsicFn {
    Dot,
}
register_diagnostics! {
    E1337,
}

pub fn remove_ptr_ty<'tcx>(ty: ty::Ty<'tcx>) -> ty::Ty<'tcx> {
    match ty.sty {
        TypeVariants::TyRef(_, ty, _) => remove_ptr_ty(ty),
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
                visitor.visit_mir(mcx.mir());
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
        //assert!(substs.len() == 1, "Len should be 1");
        let inner_ty = substs.type_at(0);
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
    pub mcx: SpirvMir<'a, 'tcx>,
}

impl<'a, 'tcx> EntryPoint<'a, 'tcx> {
    pub fn input_iter(&'a self) -> impl Iterator<Item = Input<'tcx>> + 'a {
        self.mcx.mir().args_iter().filter_map(move |local| {
            let ty = self.mcx.mir().local_decls[local].ty;
            Input::new(self, ty)
        })
    }

    pub fn uniform_iter(&'a self) -> impl Iterator<Item = Uniform<'tcx>> + 'a {
        self.mcx.mir().args_iter().filter_map(move |local| {
            let ty = self.mcx.mir().local_decls[local].ty;
            Uniform::new(self.mcx.tcx, ty)
        })
    }

    pub fn buffer_iter(&'a self) -> impl Iterator<Item = Buffer<'tcx>> + 'a {
        self.mcx.mir().args_iter().filter_map(move |local| {
            let ty = self.mcx.mir().local_decls[local].ty;
            Buffer::new(self.mcx.tcx, ty)
        })
    }

    pub fn output_iter(&'a self) -> impl Iterator<Item = Output<'tcx>> + 'a {
        use std::iter::once;
        once(self.mcx.mir().return_ty()).filter_map(move |ty| Output::new(self.mcx.tcx, ty))
    }

    pub fn args(&self) -> Vec<mir::Local> {
        match self.entry_type {
            // Need to skip?
            IntrinsicEntry::Vertex => self.mcx.mir().args_iter().skip(1).collect(),
            IntrinsicEntry::Fragment => self.mcx.mir().args_iter().collect(),
            IntrinsicEntry::Compute => self.mcx.mir().args_iter().collect(),
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

fn count_types<'a>(tys: &[ty::Ty<'a>]) -> HashMap<ty::Ty<'a>, usize> {
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
    if let TypeVariants::TyRef(_, ty_and_mut, _) = ty.sty {
        if let TypeVariants::TyAdt(adt, _substs) = ty_and_mut.sty {
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

impl<'tcx> Input<'tcx> {
    fn new<'a>(entry_point: &EntryPoint<'a, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        let (adt, substs) = get_builtin_adt(entry_point.mcx.tcx, ty, "Input")?;
        let fields: Vec<_> = adt
            .all_fields()
            .map(|field| field.ty(entry_point.mcx.tcx, substs))
            .collect();
        assert!(fields.len() == 2, "Input should have two fields");
        let location_ty = fields[1];
        let location =
            extract_location(entry_point.mcx.tcx, location_ty).expect("Unable to extract location");
        Some(Input { ty, location })
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Output<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub location: u32,
}

impl<'tcx> Output<'tcx> {
    fn new<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        let (adt, substs) = get_builtin_adt(tcx, ty, "Output")?;
        let fields: Vec<_> = adt
            .all_fields()
            .map(|field| field.ty(tcx, substs))
            .collect();
        assert!(fields.len() == 2, "Output should have two fields");
        let location_ty = fields[1];
        let location = extract_location(tcx, location_ty).expect("Unable to extract location");
        Some(Output { ty, location })
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Uniform<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub set: u32,
    pub binding: u32,
}

impl<'tcx> Uniform<'tcx> {
    fn new<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        let (adt, substs) = get_builtin_adt(tcx, ty, "Uniform")?;
        let fields: Vec<_> = adt
            .all_fields()
            .map(|field| field.ty(tcx, substs))
            .collect();
        assert_eq!(fields.len(), 3, "Uniform should have 3 fields");
        let binding_ty = fields[1];
        let set_ty = fields[2];
        let binding = extract_location(tcx, binding_ty).expect("Unable to extract location");
        let set = extract_location(tcx, set_ty).expect("Unable to extract location");
        Some(Uniform { ty, binding, set })
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Buffer<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub set: u32,
    pub binding: u32,
}

impl<'tcx> Buffer<'tcx> {
    fn new<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        let (adt, substs) = get_builtin_adt(tcx, ty, "Buffer")?;
        let fields: Vec<_> = adt
            .all_fields()
            .map(|field| field.ty(tcx, substs))
            .collect();
        assert_eq!(fields.len(), 3, "Buffer should have 3 fields");
        let binding_ty = fields[1];
        let set_ty = fields[2];
        let binding = extract_location(tcx, binding_ty).expect("Unable to extract location");
        let set = extract_location(tcx, set_ty).expect("Unable to extract location");
        Some(Buffer { ty, binding, set })
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

impl<'tcx> Global<'tcx> for Input<'tcx> {
    fn ty(&self) -> ty::Ty<'tcx> {
        self.ty
    }
}
impl<'tcx> Global<'tcx> for Buffer<'tcx> {
    fn ty(&self) -> ty::Ty<'tcx> {
        self.ty
    }
}

impl<'tcx> Global<'tcx> for Output<'tcx> {
    fn ty(&self) -> ty::Ty<'tcx> {
        self.ty
    }
}
impl<'tcx> Global<'tcx> for Uniform<'tcx> {
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
            .flat_map(|entry| EntryPoint::input_iter(entry))
            .collect();
        Self::create(set, stx, spirv::StorageClass::Input)
    }

    fn variable_iter<'borrow, 'a>(
        &'borrow self,
        entry: &'borrow EntryPoint<'a, 'tcx>,
    ) -> impl Iterator<Item = (mir::Local, GlobalVar<'tcx>)> + 'borrow {
        entry
            .mcx
            .mir()
            .args_iter()
            .filter_map(move |local| {
                let ty = entry.mcx.mir().local_decls[local].ty;
                Input::new(entry, ty).map(|input| (local, input))
            }).map(move |(local, input)| {
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
                let offsets = comp
                    .iter()
                    .map(|layout| {
                        let align = layout.align();
                        let aligned_offset = (align - (offset % align)) % align + offset;
                        offset = aligned_offset + layout.size();
                        aligned_offset
                    }).collect_vec();
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
pub fn std430_layout<'a, 'tcx>(
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
                return std430_layout(tcx, ty_vec.ty).map(|inner_layout| {
                    let single = SingleLayout {
                        ty,
                        size: inner_layout.size() * ty_vec.dim,
                        align: inner_layout.align() * multiplier,
                    };
                    Layout::Single(single)
                });
            }
            IntrinsicType::RuntimeArray(rt_array) => {
                return std430_layout(tcx, rt_array.ty).map(|inner_layout| {
                    let single = SingleLayout {
                        ty,
                        size: 0,
                        // [FIXME] proper alignment for arrays
                        align: inner_layout.align() % 16,
                    };
                    Layout::Single(single)
                });
            }
            _ => unimplemented!("Intrinsic std140"),
        }
    }

    match ty.sty {
        // TODO: Other variants
        TypeVariants::TyUint(uint_ty) => {
            assert!(uint_ty == ast::UintTy::U32, "only u32");
            let single = SingleLayout {
                ty,
                size: 4,
                align: 4,
            };
            Some(Layout::Single(single))
        }
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
                let comp = adt
                    .all_fields()
                    .map(|field| field.ty(tcx, substs))
                    .filter(|ty| !ty.is_phantom_data())
                    .map(|ty| {
                        Box::new(std430_layout(tcx, ty).expect("No layout inside Composition"))
                    }).collect_vec();
                Some(Layout::Composition(comp))
            } else {
                None
            }
        }
        _ => None,
    }
}

impl<'tcx> Entry<'tcx, Buffer<'tcx>> {
    pub fn buffer<'a>(
        entry_points: &[EntryPoint<'a, 'tcx>],
        stx: &mut CodegenCx<'a, 'tcx>,
    ) -> Self {
        let set: HashSet<_> = entry_points
            .iter()
            .flat_map(|entry| EntryPoint::buffer_iter(entry))
            .collect();
        Self::create(set, stx, spirv::StorageClass::StorageBuffer)
    }

    fn variable_iter<'borrow, 'a>(
        &'borrow self,
        entry: &'borrow EntryPoint<'a, 'tcx>,
    ) -> impl Iterator<Item = (mir::Local, GlobalVar<'tcx>)> + 'borrow {
        entry
            .mcx
            .mir()
            .args_iter()
            .filter_map(move |local| {
                let ty = entry.mcx.mir().local_decls[local].ty;
                Buffer::new(entry.mcx.tcx, ty).map(|input| (local, input))
            }).map(move |(local, uniform)| {
                (
                    local,
                    *self.global_vars.get(&uniform).expect("Entry compute"),
                )
            })
    }
}
impl<'tcx> Entry<'tcx, Uniform<'tcx>> {
    pub fn uniform<'a>(
        entry_points: &[EntryPoint<'a, 'tcx>],
        stx: &mut CodegenCx<'a, 'tcx>,
    ) -> Self {
        let set: HashSet<_> = entry_points
            .iter()
            .flat_map(|entry| EntryPoint::uniform_iter(entry))
            .collect();
        Self::create(set, stx, spirv::StorageClass::Uniform)
    }

    fn variable_iter<'borrow, 'a>(
        &'borrow self,
        entry: &'borrow EntryPoint<'a, 'tcx>,
    ) -> impl Iterator<Item = (mir::Local, GlobalVar<'tcx>)> + 'borrow {
        entry
            .mcx
            .mir()
            .args_iter()
            .filter_map(move |local| {
                let ty = entry.mcx.mir().local_decls[local].ty;
                Uniform::new(entry.mcx.tcx, ty).map(|input| (local, input))
            }).map(move |(local, uniform)| {
                (
                    local,
                    *self.global_vars.get(&uniform).expect("Entry compute"),
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
        let global_vars: HashMap<_, _> = set
            .into_iter()
            .map(|global| {
                let spirv_ty = stx.to_ty_as_ptr(global.ty(), storage_class);
                let var = stx
                    .builder
                    .variable(spirv_ty.word, None, storage_class, None);
                let global_var = GlobalVar {
                    var,
                    ty: global.ty(),
                    storage_class: storage_class,
                    location: 0,
                };
                (global, global_var)
            }).collect();
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
            .flat_map(|entry| EntryPoint::output_iter(entry))
            .collect();
        Self::create(set, stx, spirv::StorageClass::Output)
    }

    fn variable_iter<'borrow, 'a>(
        &'borrow self,
        entry: &'borrow EntryPoint<'a, 'tcx>,
    ) -> impl Iterator<Item = (mir::Local, GlobalVar<'tcx>)> + 'borrow {
        let ty = entry.mcx.mir().return_ty();
        if ty.is_nil() {
            return None.into_iter();
        }
        let output = Output::new(entry.mcx.tcx, ty).expect("Should be output");
        Some((
            mir::Local::new(0),
            *self.global_vars.get(&output).expect("Entry compute"),
        )).into_iter()
    }
}

// fn intrinsic_fn(attrs: &[syntax::ast::Attribute]) -> Option<IntrinsicFn> {
//     extract_attr(attrs, "spirv", |s| match s {
//         "dot" => Some(IntrinsicFn::Dot),
//         _ => None,
//     }).get(0)
//         .map(|&i| i)
// }

#[derive(Debug, Copy, Clone)]
pub enum RuntimeArrayIntrinsic {
    Get,
    Store,
}
#[derive(Debug, Copy, Clone)]
pub enum Intrinsic {
    GlslExt(spirv::Word),
    Abort,
    Discard,
    RuntimeArray(RuntimeArrayIntrinsic),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum InstanceType {
    Entry(IntrinsicEntry),
    Fn,
}

pub struct FunctionCx<'b, 'a: 'b, 'tcx: 'a> {
    current_table: Vec<&'a TypeckTables<'tcx>>,
    pub mcx: &'b context::SpirvMir<'a, 'tcx>,
    pub scx: &'b mut CodegenCx<'a, 'tcx>,
    pub constants: HashMap<mir::Constant<'tcx>, Variable<'tcx>>,
    pub label_blocks: HashMap<mir::BasicBlock, Label>,
    pub merge_blocks: HashMap<mir::BasicBlock, Label>,
    pub vars: HashMap<mir::Local, Variable<'tcx>>,
    pub references: HashMap<mir::Place<'tcx>, mir::Place<'tcx>>,
    pub instance_ty: InstanceType,
    pub resume_at: Option<mir::BasicBlock>,
}

#[derive(Debug, Copy, Clone)]
pub enum IntrinsicType<'tcx> {
    TyVec(TyVec<'tcx>),
    RuntimeArray(RuntimeArray<'tcx>),
}
impl<'tcx> IntrinsicType<'tcx> {
    pub fn from_ty<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        TyVec::from_ty(tcx, ty)
            .map(IntrinsicType::TyVec)
            .or_else(|| RuntimeArray::from_ty(tcx, ty).map(IntrinsicType::RuntimeArray))
    }
    pub fn contruct_ty<'a>(
        &self,
        storage_class: spirv::StorageClass,
        cx: &mut CodegenCx<'a, 'tcx>,
    ) -> Ty<'tcx> {
        use typ::ConstructTy;
        match self {
            IntrinsicType::TyVec(ty_vec) => {
                let spirv_ty = cx.to_ty(ty_vec.ty, storage_class);
                let ty = cx.builder.type_vector(spirv_ty.word, ty_vec.dim as u32);
                ty.construct_ty(ty_vec.ty)
            }
            IntrinsicType::RuntimeArray(rt_array) => {
                let spirv_ty = cx.to_ty(rt_array.ty, storage_class);
                let ty: spirv::Word = cx.builder.type_runtime_array(spirv_ty.word);
                let layout = std430_layout(cx.tcx, rt_array.ty).expect("Should have layout");
                cx.builder.decorate(
                    ty,
                    spirv::Decoration::ArrayStride,
                    &[rspirv::mr::Operand::LiteralInt32(layout.size() as u32)],
                );
                Ty::new(ty, rt_array.ty)
            }
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct RuntimeArray<'tcx> {
    pub ty: ty::Ty<'tcx>,
}
impl<'tcx> RuntimeArray<'tcx> {
    pub fn from_ty<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Self> {
        if let TypeVariants::TyAdt(adt, substs) = ty.sty {
            let attrs = tcx.get_attrs(adt.did);
            let _ = extract_attr(&attrs, "spirv", |s| match s {
                "RuntimeArray" => Some(()),
                _ => None,
            }).first()
            .cloned()?;
            assert!(adt.is_struct(), "A RuntimeArray should be a struct");
            let field = adt
                .all_fields()
                .nth(0)
                .expect("A Vec should have at least one field");
            let field_ty = field.ty(tcx, substs);
            if let TypeVariants::TyAdt(_, substs) = field_ty.sty {
                substs.types().nth(0).map(|ty| RuntimeArray { ty })
            } else {
                None
            }
        } else {
            None
        }
    }
}
#[derive(Debug, Copy, Clone)]
pub struct TyVec<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub dim: usize,
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
            .cloned()?;
            assert!(adt.is_struct(), "A Vec should be a struct");
            let field = adt
                .all_fields()
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
    Compute,
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
        }).collect::<Vec<_>>()
}

pub enum FunctionCall {
    Function(Function),
    Intrinsic(Intrinsic),
}

pub fn find_ref_functions<'borrow, 'a, 'tcx>(
    items: &'borrow [MirContext<'a, 'tcx>],
) -> impl Iterator<Item = &'borrow MirContext<'a, 'tcx>> {
    items.iter().filter(|mcx| is_ptr(mcx.mir.return_ty()))
}
pub fn trans_spirv<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, items: &'a FxHashSet<MonoItem<'tcx>>) {
    //struct_span_err!(tcx.sess, DUMMY_SP, E1337, "Test not allowed").emit();

    let mut ctx = CodegenCx::new(tcx);

    let mut instances: Vec<MirContext> = items
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
        }).collect();
    let mut file =
        std::fs::File::create("/home/maik/projects/rlsl/issues/mir/mir.dot").expect("graph");
    for (id, mcx) in instances.iter().enumerate() {
        let graph = graph::PetMir::from_mir(mcx.mir);
        graph.export(&mut file);
    }
    // let mut mir_before =
    //     std::fs::File::create("/home/maik/projects/rlsl/issues/mir/mir_before.dot").expect("graph");
    // for mcx in &instances {
    //     rustc_mir::util::write_mir_fn_graphviz(tcx, mcx.def_id, &mcx.mir, &mut mir_before);
    // }
    items.iter().for_each(|item| {
        use spirv::GLOp::*;
        if let &MonoItem::Fn(ref instance) = item {
            let def_id = instance.def_id();
            if tcx.is_foreign_item(def_id) {
                let intrinsic_name: String = tcx.item_name(def_id).into();
                let id = match intrinsic_name.as_str() {
                    "sqrtf32" => Some(Sqrt),
                    "sinf32" => Some(Sin),
                    "cosf32" => Some(Cos),
                    "tanf32" => Some(Tan),
                    "absf32" => Some(FAbs),
                    "fractf32" => Some(Fract),
                    "minf32" => Some(FMin),
                    "floorf32" => Some(Floor),
                    _ => None,
                };
                if let Some(id) = id {
                    ctx.intrinsic_fns
                        .insert(def_id, Intrinsic::GlslExt(id as u32));
                }
                let abort = match intrinsic_name.as_str() {
                    "abort" => Some(Intrinsic::Abort),
                    "spirv_discard" => Some(Intrinsic::Discard),
                    _ => None,
                };
                if let Some(abort) = abort {
                    ctx.intrinsic_fns.insert(def_id, abort);
                }
            }
        }
    });

    // Insert all the generic intrinsics, that can't be defined in an extern
    // block
    instances.iter().for_each(|mcx| {
        let attrs = tcx.get_attrs(mcx.def_id);
        let intrinsic = extract_attr(&attrs, "spirv", |s| match s {
            "runtime_array_get" => Some(Intrinsic::RuntimeArray(RuntimeArrayIntrinsic::Get)),
            "runtime_array_store" => Some(Intrinsic::RuntimeArray(RuntimeArrayIntrinsic::Store)),
            _ => None,
        }).first()
        .cloned();
        if let Some(intrinsic) = intrinsic {
            ctx.intrinsic_fns.insert(mcx.def_id, intrinsic);
        }
    });

    if TyErrorVisitor::has_error(&instances) {
        return;
    }
    use rustc_mir::transform::inline::Inline;
    use rustc_mir::transform::{MirPass, MirSource};
    let i: Vec<_> = find_ref_functions(&instances).map(|m| m.def_id).collect();

    let entry_fn_node_id = tcx.sess.entry_fn.borrow().expect("entry").0;
    let entry_fn = tcx.hir.local_def_id(entry_fn_node_id);
    // instances
    //     .iter()
    //     .filter(|mcx| mcx.def_id != entry_fn && tcx.lang_items().start_fn() != Some(mcx.def_id))
    //     .for_each(|mcx| {
    //         println!("{:#?}", mcx.def_id);
    //         println!("{:#?}", mcx.mir);
    //     });
    instances
        .iter()
        .filter(|mcx| mcx.def_id != entry_fn && tcx.lang_items().start_fn() != Some(mcx.def_id))
        .for_each(|mcx| {});
    let mut spirv_instances: Vec<_> = instances
        .iter()
        .filter(|mcx| mcx.def_id != entry_fn && tcx.lang_items().start_fn() != Some(mcx.def_id))
        .map(|mcx| context::SpirvMir::from_mir(mcx))
        .collect();
    let mut mir_after =
        std::fs::File::create("/home/maik/projects/rlsl/issues/mir/mir_after.dot").expect("graph");
    for (id, mcx) in spirv_instances.iter().enumerate() {
        let graph = graph::PetMir::from_mir(&mcx.mir);
        graph.export(&mut mir_after);
    }
    // let mut mir_after_orig =
    //     std::fs::File::create("/home/maik/projects/rlsl/issues/mir/mir_after_orig.dot")
    //         .expect("graph");
    // for mcx in &spirv_instances {
    //     rustc_mir::util::write_mir_fn_graphviz(tcx, mcx.def_id, &mcx.mir, &mut mir_after_orig);
    // }
    // spirv_instances.iter().for_each(|scx| {
    //     println!("{:#?}", scx.def_id);
    //     println!("{:#?}", scx.mir);
    // });

    // Finds functions that return a reference
    let fn_refs_def_id: Vec<_> = spirv_instances
        .iter()
        .filter(|scx| is_ptr(scx.mir.return_ty()))
        .map(|scx| scx.def_id)
        .collect();
    // Inline all functions calls of functions that return a reference
    fn_refs_def_id.iter().for_each(|&def_id| {
        spirv_instances.iter_mut().for_each(|scx| {
            let mir_source = MirSource {
                def_id,
                promoted: None,
            };
            Inline.run_pass(scx.tcx, mir_source, &mut scx.mir);
        });
    });

    // delete all mir for functions that return a reference because
    // they can not be expressed in SPIR-V and they have been inlined.
    fn_refs_def_id.iter().for_each(|&def_id| {
        spirv_instances.retain(|scx| scx.def_id != def_id);
    });

    //println!("{:#?}", ctx.intrinsic_fns);
    // write_dot(&instances);
    //spirv_instances.iter().for_each(|mcx| {
    //    //println!("{:#?}", mcx.mir());
    //    // let attrs = tcx.get_attrs(mcx.def_id);
    //    // println!("{:#?}", attrs);
    //    // println!("{}", mcx.tcx.def_symbol_name(mcx.def_id));
    //    // println!("{}", mcx.tcx.item_name(mcx.def_id));
    //});

    // Remove all the items that have been marked as an intrinsic. We
    // don't want to generate code for those fns.
    ctx.intrinsic_fns.keys().for_each(|&def_id| {
        spirv_instances.retain(|mcx| mcx.def_id != def_id);
    });

    for mcx in &spirv_instances {
        ctx.forward_fns
            .insert((mcx.def_id, mcx.substs), Function(ctx.builder.id()));
    }

    //println!("instances {:#?}", spirv_instances.iter().map(|m|m.def_id).collect::<Vec<_>>());
    let (entry_instances, fn_instances): (Vec<_>, Vec<_>) =
        spirv_instances.into_iter().partition_map(|mcx| {
            let attrs = tcx.get_attrs(mcx.def_id);
            let entry = extract_attr(&attrs, "spirv", |s| match s {
                "vertex" => Some(IntrinsicEntry::Vertex),
                "fragment" => Some(IntrinsicEntry::Fragment),
                "compute" => Some(IntrinsicEntry::Compute),
                _ => None,
            }).iter()
            .nth(0)
            .map(|&entry_type| EntryPoint {
                mcx: mcx.clone(),
                entry_type,
            });
            if let Some(entry_point) = entry {
                Either::Left(entry_point)
            } else {
                Either::Right(mcx)
            }
        });
    let entry_input = Entry::input(&entry_instances, &mut ctx);
    let entry_output = Entry::output(&entry_instances, &mut ctx);
    let entry_descriptor = Entry::uniform(&entry_instances, &mut ctx);
    let entry_buffer = Entry::buffer(&entry_instances, &mut ctx);

    entry_instances.iter().for_each(|e| {
        FunctionCx::trans_entry(
            e,
            &entry_input,
            &entry_output,
            &entry_descriptor,
            &entry_buffer,
            &mut ctx,
        );
    });
    fn_instances.iter().for_each(|mcx| {
        FunctionCx::trans_fn(mcx, &mut ctx);
    });
    std::fs::create_dir_all(".shaders").expect("create dir");
    let file_name =tcx.sess
            .local_crate_source_file
            .as_ref()
            .and_then(|p| p.file_stem())
            .map(Path::new)
            .map(|p| Path::new(".shaders").join(p.with_extension("spv")))
            .expect("file name");
    let module = ctx.build_module();
    graph::export_spirv_cfg(&module);
    context::save_module(&module, file_name);
}

impl<'b, 'a, 'tcx> FunctionCx<'b, 'a, 'tcx> {
    pub fn load_operand<'r>(&mut self, operand: &'r mir::Operand<'tcx>) -> Operand<'tcx> {
        let mir = self.mcx.mir();
        let mcx = self.mcx;
        let local_decls = &mir.local_decls;
        let ty = operand.ty(local_decls, self.mcx.tcx);
        let ty = mcx.monomorphize(&ty);
        let ty = remove_ptr_ty(ty);
        match operand {
            &mir::Operand::Copy(ref place) | &mir::Operand::Move(ref place) => {
                if let Some(ref_place) = self.references.get(place).map(|p| p.clone()) {
                    let spirv_var = Variable::access_chain(self, &ref_place);
                    Operand::new(ty, OperandVariant::Variable(spirv_var))
                } else {
                    let spirv_var = Variable::access_chain(self, place);
                    Operand::new(ty, OperandVariant::Variable(spirv_var))
                }
            }
            &mir::Operand::Constant(ref constant) => match constant.literal {
                mir::Literal::Value { ref value } => {
                    let expr = self.scx.constant(value);
                    Operand::new(ty, OperandVariant::Value(expr))
                }
                ref rest => unimplemented!("{:?}", rest),
            },
        }
    }
    pub fn trans_fn(mcx: &SpirvMir<'a, 'tcx>, scx: &mut CodegenCx<'a, 'tcx>) {
        use mir::visit::Visitor;
        let ret_ty = mcx.monomorphize(&mcx.mir().return_ty());
        let ret_ty_spirv = scx.to_ty_fn(ret_ty);
        let def_id = mcx.def_id;

        let args_ty: Vec<_> = mcx
            .mir()
            .args_iter()
            .map(|l| mcx.monomorphize(&mcx.mir().local_decls[l].ty))
            .collect();
        let fn_sig = scx.tcx.mk_fn_sig(
            args_ty.into_iter(),
            ret_ty,
            false,
            hir::Unsafety::Normal,
            Abi::Rust,
        );
        let fn_ty = scx.tcx.mk_fn_ptr(Binder::bind(fn_sig));
        let fn_ty_spirv = scx.to_ty_fn(fn_ty);

        let forward_fn = scx
            .forward_fns
            .get(&(def_id, mcx.substs))
            .map(|f| f.0)
            .expect("forward");
        let spirv_function = scx
            .builder
            .begin_function(
                ret_ty_spirv.word,
                Some(forward_fn),
                spirv::FunctionControl::empty(),
                fn_ty_spirv.word,
            ).expect("begin fn");

        scx.name_from_def_id(def_id, spirv_function);
        let params: Vec<_> = mcx
            .mir()
            .args_iter()
            .map(|local_arg| {
                let local_decl = &mcx.mir().local_decls[local_arg];
                let local_ty = mcx.monomorphize(&local_decl.ty);
                Param::alloca(scx, local_ty)
            }).collect();
        //println!("{:?} {:#?}", def_id, params);
        scx.builder.begin_basic_block(None).expect("block");
        let mut args_map: HashMap<_, _> = params
            .into_iter()
            .enumerate()
            .map(|(index, param)| {
                let local_arg = mir::Local::new(index + 1);
                let local_decl = &mcx.mir().local_decls[local_arg];
                (
                    local_arg,
                    param.to_variable(scx, spirv::StorageClass::Function),
                )
            }).collect();
        {
            use rustc_data_structures::indexed_vec::Idx;
            let local = mir::Local::new(0);
            let local_decl = &mcx.mir().local_decls[local];
            let ty = mcx.monomorphize(&local_decl.ty);
            let variable = Variable::alloca(scx, ty, spirv::StorageClass::Function);
            // TODO DEBUG
            //scx.name_from_str("retvar", spirv_var);
            args_map.insert(local, variable);
        }
        FunctionCx::new(InstanceType::Fn, mcx, args_map, scx).visit_mir(mcx.mir());
    }
    pub fn trans_entry(
        entry_point: &EntryPoint<'a, 'tcx>,
        entry_input: &Entry<'tcx, Input<'tcx>>,
        entry_output: &Entry<'tcx, Output<'tcx>>,
        entry_descriptor: &Entry<'tcx, Uniform<'tcx>>,
        entry_buffer: &Entry<'tcx, Buffer<'tcx>>,
        scx: &mut CodegenCx<'a, 'tcx>,
    ) {
        use mir::visit::Visitor;
        let def_id = entry_point.mcx.def_id;
        let mir = entry_point.mcx.mir();
        // TODO: Fix properly
        if entry_point.entry_type == IntrinsicEntry::Vertex {
            let first_local = mir::Local::new(1);
            let per_vertex = scx.get_per_vertex(mir.local_decls[first_local].ty);
        }
        if entry_point.entry_type == IntrinsicEntry::Fragment {
            let first_local = mir::Local::new(1);
            let per_fragment = scx.get_per_fragment(mir.local_decls[first_local].ty);
        }
        if entry_point.entry_type == IntrinsicEntry::Compute {
            let first_local = mir::Local::new(1);
            let per_fragment = scx.get_compute(mir.local_decls[first_local].ty);
        }
        let void = scx.tcx.mk_nil();
        let fn_sig = scx.tcx.mk_fn_sig(
            [].into_iter(),
            &void,
            false,
            hir::Unsafety::Normal,
            Abi::Rust,
        );
        let void_spirv = scx.to_ty_fn(void);
        let fn_ty = scx.tcx.mk_fn_ptr(Binder::bind(fn_sig));
        let fn_ty_spirv = scx.to_ty_fn(fn_ty);
        let forward_fn = scx
            .forward_fns
            .get(&(def_id, entry_point.mcx.substs))
            .map(|f| f.0)
            .expect("forward");
        let spirv_function = scx
            .builder
            .begin_function(
                void_spirv.word,
                Some(forward_fn),
                spirv::FunctionControl::empty(),
                fn_ty_spirv.word,
            ).expect("begin fn");
        scx.builder.begin_basic_block(None).expect("block");
        // entry_point.descriptor_iter().for_each(|input| {
        //     if let TypeVariants::TyAdt(adt, substs) = input.ty.sty {
        //         let ty = adt.all_fields()
        //             .nth(0)
        //             .expect("field")
        //             .ty(entry_point.mcx.tcx, substs);
        //         let layout = ::std140_layout(entry_point.mcx.tcx, ty);
        //     }
        // });
        let inputs_iter = entry_input.variable_iter(&entry_point);
        let output_iter = entry_output.variable_iter(&entry_point);
        let descriptor_iter = entry_descriptor.variable_iter(&entry_point);
        let buffer_iter = entry_buffer.variable_iter(&entry_point);
        entry_descriptor
            .global_vars
            .iter()
            .for_each(|(uniform, global)| {
                scx.builder.decorate(
                    global.var,
                    spirv::Decoration::DescriptorSet,
                    &[rspirv::mr::Operand::LiteralInt32(uniform.set)],
                );
                scx.builder.decorate(
                    global.var,
                    spirv::Decoration::Binding,
                    &[rspirv::mr::Operand::LiteralInt32(uniform.binding)],
                );
            });
        entry_buffer
            .global_vars
            .iter()
            .for_each(|(buffer, global)| {
                scx.builder.decorate(
                    global.var,
                    spirv::Decoration::DescriptorSet,
                    &[rspirv::mr::Operand::LiteralInt32(buffer.set)],
                );
                scx.builder.decorate(
                    global.var,
                    spirv::Decoration::Binding,
                    &[rspirv::mr::Operand::LiteralInt32(buffer.binding)],
                );
            });
        let mut variable_map: HashMap<mir::Local, Variable<'tcx>> = inputs_iter
            .chain(output_iter)
            .chain(descriptor_iter)
            .chain(buffer_iter)
            .map(|(local, global)| {
                (
                    local,
                    Variable {
                        word: global.var,
                        ty: global.ty,
                        storage_class: global.storage_class,
                    },
                )
            }).collect();

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
        if entry_point.entry_type == IntrinsicEntry::Compute {
            let first_local = mir::Local::new(1);
            let compute = scx.get_compute(mir.local_decls[first_local].ty);
            variable_map.insert(first_local, compute);
        }
        let outputs = entry_output
            .variable_iter(&entry_point)
            .map(|(_, gv)| gv)
            .collect_vec();
        if let Some(output_var) = outputs.first() {
            // Insert the return variable
            variable_map.insert(
                mir::Local::new(0),
                Variable {
                    word: output_var.var,
                    ty: mir.return_ty(),
                    storage_class: output_var.storage_class,
                },
            );
        }

        let raw_builtin = variable_map
            .get(&mir::Local::new(1))
            .as_ref()
            .expect("")
            .word;
        FunctionCx::new(
            InstanceType::Entry(entry_point.entry_type),
            &entry_point.mcx,
            variable_map,
            scx,
        ).visit_mir(&entry_point.mcx.mir);
        let mut inputs_raw = entry_input
            .variable_iter(&entry_point)
            .map(|(_, gv)| gv.var)
            .collect_vec();
        inputs_raw.extend(outputs.iter().map(|gv| gv.var));
        inputs_raw.push(raw_builtin);
        let name = entry_point.mcx.tcx.item_name(def_id);
        let model = match entry_point.entry_type {
            IntrinsicEntry::Vertex => spirv::ExecutionModel::Vertex,
            IntrinsicEntry::Fragment => spirv::ExecutionModel::Fragment,
            IntrinsicEntry::Compute => spirv::ExecutionModel::GLCompute,
        };
        scx.builder
            .entry_point(model, spirv_function, name, inputs_raw);
        match entry_point.entry_type {
            IntrinsicEntry::Vertex | IntrinsicEntry::Fragment => {
                scx.builder.execution_mode(
                    spirv_function,
                    spirv::ExecutionMode::OriginUpperLeft,
                    &[],
                );
            }
            IntrinsicEntry::Compute => {
                scx.builder.execution_mode(
                    spirv_function,
                    spirv::ExecutionMode::LocalSize,
                    &[1, 1, 1],
                );
            }
        }
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
    pub fn constant(&mut self, val: &ty::Const<'tcx>) -> Value {
        self.scx.constant(val)
    }
    pub fn constant_f32(&mut self, value: f32) -> Value {
        self.scx.constant_f32(value)
    }
    pub fn constant_u32(&mut self, value: u32) -> Value {
        self.scx.constant_u32(value)
    }

    pub fn get_table(&self) -> &'a TypeckTables<'tcx> {
        self.current_table.last().expect("no table yet")
    }
    pub fn get_label(&self, block: mir::BasicBlock) -> Label {
        self.merge_blocks
            .get(&block)
            .map(|r| *r)
            .or_else(|| self.label_blocks.get(&block).map(|r| *r))
            .expect("Get label")
    }
    pub fn new(
        instance_ty: InstanceType,
        mcx: &'b SpirvMir<'a, 'tcx>,
        mut variable_map: HashMap<mir::Local, Variable<'tcx>>,
        scx: &'b mut CodegenCx<'a, 'tcx>,
    ) -> Self {
        let label_blocks: HashMap<_, _> = mcx
            .mir()
            .basic_blocks()
            .iter_enumerated()
            .map(|(block, _)| (block, Label(scx.builder.id())))
            .collect();
        let local_vars: HashMap<_, _> = mcx
            .mir()
            .vars_and_temps_iter()
            .filter_map(|local_var| {
                // Don't generate variables for ptrs
                let local_decl = &mcx.mir().local_decls[local_var];
                let local_ty = mcx.monomorphize(&local_decl.ty);
                if is_ptr(local_ty) {
                    return None;
                }
                let local_ty = remove_ptr_ty(local_ty);
                let variable = Variable::alloca(scx, local_ty, spirv::StorageClass::Function);
                Some((local_var, variable))
            }).collect();
        {
            let spirv_label = label_blocks
                .get(&mir::BasicBlock::new(0))
                .expect("No first label");
            scx.builder.branch(spirv_label.0).expect("branch");
        }
        variable_map.extend(local_vars.into_iter());
        //println!("{:?}", variable_map);
        let visitor = FunctionCx {
            instance_ty,
            scx,
            current_table: Vec::new(),
            mcx,
            constants: HashMap::new(),
            label_blocks,
            vars: variable_map,
            references: HashMap::new(),
            merge_blocks: HashMap::new(),
            resume_at: None,
        };
        visitor
    }
}
fn is_ptr(ty: ty::Ty) -> bool {
    ty.is_unsafe_ptr() || ty.is_mutable_pointer() || ty.is_region_ptr()
}

fn is_unreachable(mir: &mir::Mir, block: mir::BasicBlock) -> bool {
    match mir.basic_blocks()[block].terminator().kind {
        mir::TerminatorKind::Unreachable => true,
        _ => false,
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Enum<'tcx> {
    pub discr_ty: ty::Ty<'tcx>,
    pub index: usize,
}

impl<'tcx> Enum<'tcx> {
    pub fn from_ty<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>, ty: ty::Ty<'tcx>) -> Option<Enum<'tcx>> {
        use rustc::ty::layout::Variants;
        let layout = tcx
            .layout_of(ty::ParamEnvAnd {
                param_env: ty::ParamEnv::reveal_all(),
                value: ty,
            }).ok();
        let e = layout.and_then(|ty_layout| {
            let ty = ty_layout.ty;
            match ty_layout.details.variants {
                Variants::Tagged {
                    ref tag,
                    ref variants,
                } => {
                    // TODO: Find the correct discr type
                    let discr_ty = tcx.types.u32;
                    Some((discr_ty, variants.len()))
                }
                Variants::NicheFilling { .. } => {
                    // TODO: Handle Niechefilling enums
                    None
                }
                _ => None,
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
            //println!("{:?} {:?} {:?}", self.mcx.def_id, block, spirv_label);
            self.scx.builder.name(spirv_label.0, format!("{:?}", block));
            let label = self
                .scx
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
        if let Some(yield_ty) = &mir.yield_ty {
            self.visit_ty(
                yield_ty,
                mir::visit::TyContext::YieldTy(mir::SourceInfo {
                    span: mir.span,
                    scope: mir::OUTERMOST_SOURCE_SCOPE,
                }),
            );
        }
        for (bb, data) in order {
            self.visit_basic_block_data(bb, &data);
        }
        //self.visit_basic_block_data(mir::START_BLOCK, &mir.basic_blocks()[mir::START_BLOCK]);

        for scope in &mir.source_scopes {
            self.visit_source_scope_data(scope);
        }

        let lookup = mir::visit::TyContext::ReturnTy(mir::SourceInfo {
            span: mir.span,
            scope: mir::OUTERMOST_SOURCE_SCOPE,
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
            .ty(&self.mcx.mir().local_decls, self.scx.tcx)
            .to_ty(self.scx.tcx);
        let lvalue_ty = self.mcx.monomorphize(&lvalue_ty);
        if lvalue_ty.is_phantom_data() || lvalue_ty.is_nil() {
            return;
        }
        let ty = rvalue.ty(&self.mcx.mir().local_decls, self.scx.tcx);
        if let TypeVariants::TyTuple(ref slice) = ty.sty {
            if slice.len() == 0 {
                return;
            }
        }
        let ty = self.mcx.monomorphize(&ty);
        let lvalue_ty_spirv = self.to_ty_fn(lvalue_ty);
        // If we find an rvalue ref, this means that we are borrow some lvalue and
        // create a new lvalue variable that is a ptr. In rlsl this lvalue does not
        // exist and we optimize it away by storing the rvalue in a hashmap.
        // Eg _5 = &_6, then we store _5 -> _6. When ever we try to access _5, we need
        // to look up the real place inside this hashmap.
        if let &mir::Rvalue::Ref(_, _, ref place) = rvalue {
            assert!(is_ptr(lvalue_ty), "LValue should be a ptr");
            self.references.insert(lvalue.clone(), place.clone());
            // Don't do anything
            return;
        }
        // Sometimes we try to write to an lvalue that is a ptr. Again this ptr does
        // not exist. The rvalue is a ptr but as a use, which means we need to look up
        // the value where is rvalue ptr points to.
        if let mir::Rvalue::Use(operand) = rvalue {
            if is_ptr(lvalue_ty) && is_ptr(ty) {
                let place = match operand {
                    mir::Operand::Copy(place) | mir::Operand::Move(place) => place,
                    _ => unimplemented!(),
                };
                let deref_place = self
                    .references
                    .get(place)
                    .expect("Reference not found")
                    .clone();
                self.references.insert(lvalue.clone(), deref_place);
                return;
            }
        }
        let expr = match rvalue {
            &mir::Rvalue::BinaryOp(op, ref l, ref r)
            | &mir::Rvalue::CheckedBinaryOp(op, ref l, ref r) => self.binary_op(ty, op, l, r),
            &mir::Rvalue::Use(ref operand) => self.load_operand(operand).load(self.scx),

            &mir::Rvalue::Aggregate(_, ref operands) => {
                // If there are no operands, then it should be 0 sized and we can
                // abort.
                if operands.is_empty() {
                    return;
                }

                let spirv_operands: Vec<_> = operands
                    .iter()
                    .filter_map(|op| {
                        let ty = op.ty(&self.mcx.mir().local_decls, self.mcx.tcx);
                        let ty = self.mcx.monomorphize(&ty);
                        if ty.is_phantom_data() {
                            None
                        } else {
                            Some(self.load_operand(op).load(self.scx).word)
                        }
                    }).collect();
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
                let ty = self.mcx.mir().local_decls[local].ty;
                let ty = self.mcx.monomorphize(&ty);
                // TODO: Cleanup, currently generates 0 value for non enum types
                let load = if let Some(enum_data) = Enum::from_ty(self.mcx.tcx, ty) {
                    let discr_ty = self.mcx.monomorphize(&enum_data.discr_ty);
                    let discr_ty_spirv = self.to_ty_fn(discr_ty);
                    let discr_ty_spirv_ptr = self.to_ty_as_ptr_fn(discr_ty);
                    let index = self.constant_u32(enum_data.index as u32).word;
                    let access = self
                        .scx
                        .builder
                        .access_chain(discr_ty_spirv_ptr.word, None, var.word, &[index])
                        .expect("access");
                    self.scx
                        .builder
                        .load(discr_ty_spirv.word, None, access, None, &[])
                        .expect("load")
                } else {
                    self.constant_u32(0).word
                };
                let lvalue_ty = self.mcx.monomorphize(&lvalue_ty);
                let target_ty_spirv = self.to_ty_fn(lvalue_ty);
                let cast = self
                    .scx
                    .builder
                    .bitcast(target_ty_spirv.word, None, load)
                    .expect("bitcast");

                Value::new(cast)
            }
            &mir::Rvalue::UnaryOp(ref op, ref operand) => match op {
                mir::UnOp::Not => {
                    let load = self.load_operand(operand).load(self.scx);
                    let spirv_ty = self.to_ty_fn(ty);
                    let not = self
                        .scx
                        .builder
                        .not(spirv_ty.word, None, load.word)
                        .expect("op not");
                    Value::new(not)
                }
                _ => unimplemented!("unary op"),
            },
            &mir::Rvalue::Cast(_, ref _op, _ty) => {
                //let op_ty = op.ty(&self.mcx.mir().local_decls, self.mcx.tcx);
                unimplemented!("cast")
            }

            rest => unimplemented!("{:?}", rest),
        };

        let variable = Variable::access_chain(self, lvalue);
        variable.store(self.scx, expr);
    }
    // fn super_terminator_kind(
    //     &mut self,
    //     block: mir::BasicBlock,
    //     kind: &mir::TerminatorKind<'tcx>,
    //     source_location: mir::Location,
    // ) {
    //     use mir::TerminatorKind;
    //     use rustc::mir::visit::PlaceContext;
    //     match *kind {
    //         TerminatorKind::Goto { target } => {
    //             self.visit_branch(block, target);
    //         }

    //         TerminatorKind::SwitchInt {
    //             ref discr,
    //             ref switch_ty,
    //             values: _,
    //             ref targets,
    //         } => {
    //             self.visit_operand(discr, source_location);
    //             self.visit_ty(switch_ty, TyContext::Location(source_location));
    //             for &target in targets {
    //                 self.visit_branch(block, target);
    //             }
    //         }

    //         TerminatorKind::Resume
    //         | TerminatorKind::Abort
    //         | TerminatorKind::Return
    //         | TerminatorKind::GeneratorDrop
    //         | TerminatorKind::Unreachable => {}

    //         TerminatorKind::Drop {
    //             ref location,
    //             target,
    //             ..
    //         } => {
    //             self.visit_place(location, PlaceContext::Drop, source_location);
    //             self.visit_branch(block, target);
    //         }

    //         TerminatorKind::DropAndReplace {
    //             ref location,
    //             ref value,
    //             target,
    //             ..
    //         } => {
    //             self.visit_place(location, PlaceContext::Drop, source_location);
    //             self.visit_operand(value, source_location);
    //             self.visit_branch(block, target);
    //         }

    //         TerminatorKind::Call {
    //             ref func,
    //             ref args,
    //             ref destination,
    //             cleanup
    //         } => {
    //             println!("CALL {:?} {:?}", destination, cleanup);
    //             self.visit_operand(func, source_location);
    //             for arg in args {
    //                 self.visit_operand(arg, source_location);
    //             }
    //             if let Some((ref destination, target)) = *destination {
    //                 self.visit_place(destination, PlaceContext::Call, source_location);
    //                 self.visit_branch(block, target);
    //             }
    //         }

    //         TerminatorKind::Assert {
    //             ref cond,
    //             expected: _,
    //             ref msg,
    //             target,
    //             ..
    //         } => {
    //             self.visit_operand(cond, source_location);
    //             self.visit_assert_message(msg, source_location);
    //             self.visit_branch(block, target);
    //         }

    //         TerminatorKind::Yield {
    //             ref value,
    //             resume,
    //             drop,
    //         } => {
    //             self.visit_operand(value, source_location);
    //             self.visit_branch(block, resume);
    //             drop.map(|t| self.visit_branch(block, t));
    //         }

    //         TerminatorKind::FalseEdges {
    //             real_target,
    //             ref imaginary_targets,
    //         } => {
    //             self.visit_branch(block, real_target);
    //             for target in imaginary_targets {
    //                 self.visit_branch(block, *target);
    //             }
    //         }

    //         TerminatorKind::FalseUnwind {
    //             real_target,
    //             ..
    //         } => {
    //             self.visit_branch(block, real_target);
    //         }
    //     }
    // }

    // fn visit_branch(&mut self, source: mir::BasicBlock, target: mir::BasicBlock) {
    //     self.visit_basic_block_data(target, &self.mcx.mir.basic_blocks()[target]);
    // }
    fn visit_terminator_kind(
        &mut self,
        block: mir::BasicBlock,
        kind: &mir::TerminatorKind<'tcx>,
        location: mir::Location,
    ) {
        let mir = self.mcx.mir();
        match kind {
            &mir::TerminatorKind::Return => {
                // If we are inside an entry, we just return void
                if self.instance_ty != InstanceType::Fn {
                    return self.scx.builder.ret().expect("ret");
                } else {
                    match mir.return_ty().sty {
                        TypeVariants::TyTuple(ref slice) if slice.len() == 0 => {
                            self.scx.builder.ret().expect("ret");
                        }
                        _ => {
                            use rustc_data_structures::indexed_vec::Idx;
                            let ty = self.mcx.monomorphize(&mir.return_ty());
                            let spirv_ty = { self.to_ty_fn(ty) };
                            let var = self.vars.get(&mir::Local::new(0)).unwrap();
                            let load = self
                                .scx
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
                ref values,
                ..
            } => {
                // let _selector = if switch_ty.is_bool() {
                //     // TODO bitcast api
                //     let load = self.load_operand(discr).load(self.scx).word;
                //     let target_ty = self.mcx.tcx.mk_mach_uint(syntax::ast::UintTy::U32);
                //     let target_ty_spirv = self.to_ty_fn(target_ty);
                //     // self.scx
                //     //     .builder
                //     //     .bitcast(target_ty_spirv.word, None, load)
                //     //     .expect("bitcast")
                //     let one = self.scx.constant_u32(1).word;
                //     let zero = self.scx.constant_u32(0).word;
                //     self.scx
                //         .builder
                //         .select(self.scx.bool_ty, None, load, one, zero)
                //         .expect("select")
                // } else {
                //     self.load_operand(discr).load(self.scx).word
                // };
                let default_label = *self
                    .label_blocks
                    .get(targets.last().unwrap())
                    .expect("default label");
                let labels: Vec<_> = targets
                    .iter()
                    .take(targets.len() - 1)
                    .enumerate()
                    .map(|(index, target)| {
                        let label = self.label_blocks.get(&target).expect("label");
                        (values[index] as u32, label.0)
                    }).collect();
                // Sometimes we get duplicated merge block labels. To fix this we
                // always create a new block and branch to it. This will give us a new unique
                // block.
                //let new_block_id = self.scx.builder.id();
                let merge_block = *self.mcx.merge_blocks.get(&block).expect("merge block");
                // {
                //     use rustc_data_structures::control_flow_graph::ControlFlowGraph;
                //     let pred: HashSet<_> = ControlFlowGraph::predecessors(&mir, merge_block)
                //         .into_iter()
                //         .collect();
                //     let suc: HashSet<_> = ControlFlowGraph::successors(&mir, block)
                //         .into_iter()
                //         .collect();
                //     println!("{:?}", pred.intersection(&suc),);
                // }

                let merge_block_label = *self.label_blocks.get(&merge_block).expect("no label");
                let bb = &self.mcx.mir.basic_blocks()[merge_block];
                if switch_ty.is_bool() {
                    let bool_load = self.load_operand(discr).load(self.scx).word;
                    self.scx
                        .builder
                        .selection_merge(merge_block_label.0, spirv::SelectionControl::empty())
                        .expect("selection merge");
                    self.scx
                        .builder
                        .branch_conditional(bool_load, default_label.0, labels[0].1, &[])
                        .expect("if");
                } else {
                    let selector = self.load_operand(discr).load(self.scx).word;
                    //let selector = self.constant_u32(0).word;
                    self.scx
                        .builder
                        .selection_merge(merge_block_label.0, spirv::SelectionControl::empty())
                        .expect("selection merge");
                    self.scx
                        .builder
                        .switch(selector, default_label.0, &labels)
                        .expect("switch");
                    // let bool_ty = self.scx.tcx.types.bool;
                    // let bool_ty_spirv = self.to_ty_fn(bool_ty);
                    // let zero = self.constant_u32(0);
                    // let bool_load = self.load_operand(discr).load(self.scx).word;
                    // // TODO: Check for the discr correctly.
                    // self.scx
                    //     .builder
                    //     .iequal(bool_ty_spirv.word, None, bool_load, zero.word)
                    //     .expect("cast")
                };
                // assert!(labels.len() == 1);
                // self.scx
                //     .builder
                //     .branch_conditional(bool_load, default_label.0, labels[0].1, &[])
                //     .expect("if");
                // self.scx.builder.begin_basic_block(Some(new_block_id));
                // self.scx.builder.branch(merge_block_label.0);
                //self.merge_blocks.insert()
            }
            &mir::TerminatorKind::Call {
                ref func,
                ref args,
                ref destination,
                cleanup,
            } => {
                if let Some(cleanup) = cleanup {
                    self.resume_at = Some(cleanup);
                }
                let local_decls = &self.mcx.mir().local_decls;
                let fn_ty = func.ty(self.mcx.mir(), self.mcx.tcx);
                let (def_id, substs) = match fn_ty.sty {
                    TypeVariants::TyFnDef(def_id, ref substs) => (def_id, substs),
                    _ => panic!("Not a function"),
                };

                let tcx = self.mcx.tcx;
                let fn_sig = fn_ty.fn_sig(tcx);
                let abi = fn_sig.abi();
                let ret_ty_binder = fn_sig.output();
                let ret_ty = ret_ty_binder.skip_binder();
                let ret_ty = self.mcx.monomorphize(ret_ty);
                let spirv_ty = self.to_ty_fn(ret_ty);
                let mono_substs = self.mcx.monomorphize(substs);
                let resolve_instance = Instance::resolve(
                    self.scx.tcx,
                    ty::ParamEnv::reveal_all(),
                    def_id,
                    &mono_substs,
                ).expect("resolve instance call");
                let fn_call = self
                    .scx
                    .get_function_call(resolve_instance.def_id(), resolve_instance.substs)
                    .expect(&format!(
                        "function call for {:?}",
                        resolve_instance.def_id()
                    ));
                // Split the rust-call tupled arguments off.
                let (first_args, untuple) = if abi == Abi::RustCall && !args.is_empty() {
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
                    }).collect();
                if let Some(tup) = untuple {
                    let ty = tup.ty(local_decls, self.mcx.tcx);
                    let ty = self.mcx.monomorphize(&ty);
                    let tuple_var = self
                        .load_operand(tup)
                        .to_variable()
                        .expect("Should be a variable");
                    match ty.sty {
                        TypeVariants::TyTuple(slice) => {
                            let tuple_iter = slice.iter().enumerate().map(|(idx, field_ty)| {
                                let field_ty_spv = self.to_ty_fn(field_ty);
                                let field_ty_ptr_spv = self.to_ty_as_ptr_fn(field_ty);
                                let spirv_idx = self.constant_u32(idx as u32).word;
                                let field = self
                                    .scx
                                    .builder
                                    .access_chain(
                                        field_ty_ptr_spv.word,
                                        None,
                                        tuple_var.word,
                                        &[spirv_idx],
                                    ).expect("access chain");
                                self.scx
                                    .builder
                                    .load(field_ty_spv.word, None, field, None, &[])
                                    .expect("load")
                            });
                            arg_operand_loads.extend(tuple_iter);
                        }
                        _ => panic!("tup"),
                    }
                }
                let spirv_fn_call = match fn_call {
                    FunctionCall::Function(spirv_fn) => {
                        let fn_call = self
                            .scx
                            .builder
                            .function_call(spirv_ty.word, None, spirv_fn.0, &arg_operand_loads)
                            .expect("fn call");
                        Some(fn_call)
                    }
                    FunctionCall::Intrinsic(intrinsic) => match intrinsic {
                        Intrinsic::GlslExt(id) => {
                            let ext_fn = self
                                .scx
                                .builder
                                .ext_inst(
                                    spirv_ty.word,
                                    None,
                                    self.scx.glsl_ext_id,
                                    id,
                                    &arg_operand_loads,
                                ).expect("ext instr");
                            Some(ext_fn)
                        }
                        Intrinsic::Abort => {
                            self.scx.builder.unreachable().expect("unreachable");
                            None
                        }
                        Intrinsic::Discard => {
                            self.scx.builder.kill().expect("unreachable");
                            return;
                        }
                        Intrinsic::RuntimeArray(runtime_array) => match runtime_array {
                            RuntimeArrayIntrinsic::Store => {
                                let ty = args[2].ty(local_decls, self.scx.tcx);
                                let spirv_ptr_ty =
                                    self.to_ty_as_ptr(ty, spirv::StorageClass::StorageBuffer);
                                let access_chain = self
                                    .scx
                                    .builder
                                    .access_chain(
                                        spirv_ptr_ty.word,
                                        None,
                                        arg_operand_loads[0],
                                        &arg_operand_loads[1..2],
                                    ).expect("access chain");
                                self.scx
                                    .builder
                                    .store(access_chain, arg_operand_loads[2], None, &[])
                                    .expect("store");
                                None
                            }
                            RuntimeArrayIntrinsic::Get => {
                                let spirv_ptr_ty =
                                    self.to_ty_as_ptr(ret_ty, spirv::StorageClass::StorageBuffer);
                                let access_chain = self
                                    .scx
                                    .builder
                                    .access_chain(
                                        spirv_ptr_ty.word,
                                        None,
                                        arg_operand_loads[0],
                                        &arg_operand_loads[1..],
                                    ).expect("access chain");
                                let load = self
                                    .scx
                                    .builder
                                    .load(spirv_ty.word, None, access_chain, None, &[])
                                    .expect("Load access_chain");
                                Some(load)
                            }
                        },
                    },
                };
                // only write op store if the result is not nil
                if !ret_ty.is_nil() {
                    if let (&Some(ref dest), Some(spirv_fn_call)) = (destination, spirv_fn_call) {
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
                unwind,
            } => {
                // let target_label = self.label_blocks.get(&target).expect("no label");
                // self.scx.builder.branch(target_label.0).expect("label");
                if let Some(unwind) = unwind {
                    self.resume_at = Some(target);
                    let target_label = self.label_blocks.get(&unwind).expect("no label");
                    self.scx.builder.branch(target_label.0).expect("label");
                } else {
                    let target_label = self.label_blocks.get(&target).expect("no label");
                    self.scx.builder.branch(target_label.0).expect("label");
                }
            }
            &mir::TerminatorKind::Resume => {
                let resume_at = self.resume_at.expect("Resume");
                let target_label = self.label_blocks.get(&resume_at).expect("no label");
                self.scx.builder.branch(target_label.0).expect("label");
                //self.scx.builder.unreachable();
            }
            mir::TerminatorKind::Unreachable => {
                self.scx.builder.unreachable();
            }
            rest => unimplemented!("{:?}", rest),
        };
        self.super_terminator_kind(block, kind, location);
    }
}

#[derive(Debug, Clone)]
pub enum SpirvRvalue {}

impl<'b, 'a, 'tcx> FunctionCx<'b, 'a, 'tcx> {
    pub fn binary_op(
        &mut self,
        return_ty: ty::Ty<'tcx>,
        op: mir::BinOp,
        l: &mir::Operand<'tcx>,
        r: &mir::Operand<'tcx>,
    ) -> Value {
        let spirv_return_ty = self.to_ty_fn(return_ty);
        let ty = l.ty(&self.mcx.mir().local_decls, self.mcx.tcx);
        // TODO: Different types
        let spirv_ty = self.to_ty_fn(ty);
        let left = self.load_operand(l).load(self.scx).word;
        let right = self.load_operand(r).load(self.scx).word;
        // TODO: Impl ops
        match ty.sty {
            ty::TypeVariants::TyUint(_) => {
                match op {
                    mir::BinOp::Add => {
                        let tup = self.scx.tcx.mk_tup([ty, ty].iter());
                        let spirv_tup = self.to_ty_fn(tup);
                        let add = self
                            .scx
                            .builder
                            .iadd_carry(spirv_tup.word, None, left, right)
                            .expect("fmul");
                        let value = self
                            .scx
                            .builder
                            .composite_extract(spirv_ty.word, None, add, &[0])
                            .expect("extract");
                        let carry_u32 = self
                            .scx
                            .builder
                            .composite_extract(spirv_ty.word, None, add, &[1])
                            .expect("extract");
                        let bool_ty = self.scx.tcx.types.bool;
                        let spirv_bool = self.to_ty_fn(bool_ty);
                        let carry_bool = self
                            .scx
                            .builder
                            .bitcast(spirv_bool.word, None, carry_u32)
                            .expect("failed to bitcast");
                        let s = self
                            .scx
                            .builder
                            .composite_construct(spirv_return_ty.word, None, &[value, carry_bool])
                            .expect("c");
                        //let cast = self.scx.builder.bitcast(spirv_return_ty.word, None, add).expect("bitcast");
                        Value::new(s)
                    }
                    mir::BinOp::Mul => {
                        let mul = self
                            .scx
                            .builder
                            .imul(spirv_ty.word, None, left, right)
                            .expect("fmul");
                        Value::new(mul)
                    }
                    _ => unimplemented!("op unsigned"),
                }
            }
            ty::TypeVariants::TyFloat(_) => match op {
                mir::BinOp::Mul => {
                    let mul = self
                        .scx
                        .builder
                        .fmul(spirv_ty.word, None, left, right)
                        .expect("fmul");
                    Value::new(mul)
                }
                mir::BinOp::Add => {
                    let add = self
                        .scx
                        .builder
                        .fadd(spirv_ty.word, None, left, right)
                        .expect("fadd");
                    Value::new(add)
                }
                mir::BinOp::Sub => {
                    let add = self
                        .scx
                        .builder
                        .fsub(spirv_ty.word, None, left, right)
                        .expect("fsub");
                    Value::new(add)
                }
                mir::BinOp::Div => {
                    let add = self
                        .scx
                        .builder
                        .fdiv(spirv_ty.word, None, left, right)
                        .expect("fsub");
                    Value::new(add)
                }
                mir::BinOp::Gt => {
                    let gt = match ty.sty {
                        TypeVariants::TyInt(_) => self
                            .scx
                            .builder
                            .sgreater_than(spirv_return_ty.word, None, left, right)
                            .expect("g"),
                        TypeVariants::TyUint(_) | TypeVariants::TyBool => self
                            .scx
                            .builder
                            .ugreater_than(spirv_return_ty.word, None, left, right)
                            .expect("g"),
                        TypeVariants::TyFloat(_) => self
                            .scx
                            .builder
                            .ford_greater_than(spirv_return_ty.word, None, left, right)
                            .expect("g"),
                        ref rest => unimplemented!("{:?}", rest),
                    };
                    Value::new(gt)
                }
                mir::BinOp::Lt => {
                    let lt = match ty.sty {
                        TypeVariants::TyFloat(_) => self
                            .scx
                            .builder
                            .ford_less_than(spirv_return_ty.word, None, left, right)
                            .expect("g"),
                        ref rest => unimplemented!("{:?}", rest),
                    };
                    Value::new(lt)
                }
                mir::BinOp::Shl => {
                    let shl = self
                        .scx
                        .builder
                        .shift_left_logical(spirv_ty.word, None, left, right)
                        .expect("shl");
                    Value::new(shl)
                }
                mir::BinOp::BitOr => {
                    let bit_or = self
                        .scx
                        .builder
                        .bitwise_or(spirv_ty.word, None, left, right)
                        .expect("bitwise or");
                    Value::new(bit_or)
                }
                mir::BinOp::Ne => {
                    let ne = self
                        .scx
                        .builder
                        .logical_not_equal(spirv_return_ty.word, None, left, right)
                        .expect("not equal");
                    Value::new(ne)
                }
                rest => unimplemented!("{:?}", rest),
            },
            _ => unimplemented!("ops"),
        }
    }
}

pub fn remove_unwind<'tcx>(mir: &mut mir::Mir<'tcx>) {
    for data in mir.basic_blocks_mut() {
        let term = data.terminator_mut();
        match &mut term.kind {
            mir::TerminatorKind::Call {
                cleanup: unwind, ..
            }
            | mir::TerminatorKind::FalseUnwind { unwind, .. }
            | mir::TerminatorKind::Assert { cleanup: unwind, .. }
            | mir::TerminatorKind::DropAndReplace { unwind, .. }
            | mir::TerminatorKind::Drop { unwind, .. } => {
                *unwind = None;
            }
            _ => (),
        }
    }
}
