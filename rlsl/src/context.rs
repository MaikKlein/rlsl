use self::hir::def_id::DefId;
use graph::{Loops, PetMir};
use itertools::Itertools;
use petgraph::visit::VisitMap;
use petgraph::visit::{Bfs, Dfs, DfsPostOrder, Reversed, Walker};
use rspirv;
use rspirv::mr::Builder;
use rustc;
use rustc::hir;
use rustc::middle::lang_items::LangItem;
use rustc::mir;
use rustc::ty;
use rustc::ty::subst::Substs;
use rustc::ty::{subst, TyCtxt};
use rustc_data_structures::control_flow_graph::{iterate::post_order_from_to, ControlFlowGraph};
use rustc_data_structures::indexed_vec::IndexVec;
use spirv;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use syntax;
use syntax_pos::DUMMY_SP;
use ConstructTy;
use {Enum, Variable};
use {Function, FunctionCall, Intrinsic, IntrinsicType, Ty, Value};

trait ToParamEnvAnd<'tcx, T> {
    fn to_param_env_and(self) -> ty::ParamEnvAnd<'tcx, ty::Ty<'tcx>>;
}
impl<'tcx> ToParamEnvAnd<'tcx, ty::Ty<'tcx>> for ty::Ty<'tcx> {
    fn to_param_env_and(self) -> ty::ParamEnvAnd<'tcx, ty::Ty<'tcx>> {
        ty::ParamEnvAnd {
            param_env: ty::ParamEnv::reveal_all(),
            value: self,
        }
    }
}

pub struct CodegenCx<'a, 'tcx: 'a> {
    per_vertex: Option<Variable<'tcx>>,
    per_fragment: Option<Variable<'tcx>>,
    compute: Option<Variable<'tcx>>,
    pub tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    pub builder: Builder,
    pub ty_cache: HashMap<ty::Ty<'tcx>, Ty<'tcx>>,
    pub ty_ptr_cache: HashMap<(ty::Ty<'tcx>, spirv::StorageClass), Ty<'tcx>>,
    pub const_cache: HashMap<ty::Const<'tcx>, Value>,
    pub forward_fns: HashMap<(hir::def_id::DefId, &'a Substs<'tcx>), Function>,
    pub intrinsic_fns: HashMap<hir::def_id::DefId, Intrinsic>,
    pub debug_symbols: bool,
    pub glsl_ext_id: spirv::Word,
    pub bool_ty: spirv::Word,
}

impl<'a, 'tcx> CodegenCx<'a, 'tcx> {
    pub fn bool_to_u32(&mut self, bool_load: spirv::Word) -> Value {
        let zero = self.constant_u32(0);
        let one = self.constant_u32(1);
        let u32_ty = self.tcx.types.u32;
        let u32_ty = self.to_ty_fn(u32_ty);
        let val = self
            .builder
            .select(u32_ty.word, None, bool_load, one.word, zero.word)
            .expect("convert to u32");
        Value::new(val)
    }
    pub fn to_ty_fn(&mut self, ty: ty::Ty<'tcx>) -> Ty<'tcx> {
        self.to_ty(ty, spirv::StorageClass::Function)
    }
    pub fn to_ty_as_ptr_fn(&mut self, ty: ty::Ty<'tcx>) -> Ty<'tcx> {
        self.to_ty_as_ptr(ty, spirv::StorageClass::Function)
    }
    /// Tries to get a function id, if it fails it looks for an intrinsic id
    pub fn get_function_call(
        &self,
        def_id: DefId,
        substs: &'a Substs<'tcx>,
    ) -> Option<FunctionCall> {
        self.forward_fns
            .get(&(def_id, substs))
            .map(|&spirv_fn| FunctionCall::Function(spirv_fn))
            .or(self
                .intrinsic_fns
                .get(&def_id)
                .map(|&id| FunctionCall::Intrinsic(id)))
    }

    pub fn constant_f32(&mut self, value: f32) -> Value {
        let val = ty::Const::from_bits(
            self.tcx,
            value.to_bits() as u128,
            self.tcx.types.f32.to_param_env_and(),
        );
        self.constant(val)
    }

    pub fn constant_u32(&mut self, value: u32) -> Value {
        let val = ty::Const::from_bits(
            self.tcx,
            value as u128,
            self.tcx.types.u32.to_param_env_and(),
        );
        self.constant(val)
    }

    pub fn constant(&mut self, const_val: &ty::Const<'tcx>) -> Value {
        if let Some(val) = self.const_cache.get(const_val) {
            return *val;
        }
        let const_ty = const_val.ty;
        let spirv_val = match const_ty.sty {
            ty::TypeVariants::TyBool | ty::TypeVariants::TyUint(_) | ty::TypeVariants::TyInt(_) => {
                let value = const_val
                    .to_bits(self.tcx, const_ty.to_param_env_and())
                    .expect("bits from const");
                // [FIXME] Storageptr
                let spirv_ty = self.to_ty_fn(const_ty);
                self.builder.constant_u32(spirv_ty.word, value as u32)
            }
            ty::TypeVariants::TyFloat(_) => {
                let value = const_val
                    .to_bits(self.tcx, const_ty.to_param_env_and())
                    .expect("bits from const");
                let spirv_ty = self.to_ty_fn(const_ty);
                self.builder
                    .constant_f32(spirv_ty.word, f32::from_bits(value as u32))
            }
            //[FIXME] Add other constants
            ref rest => unimplemented!("Const {:?}", rest), // ConstValue::Integer(const_int) => {
                                                            //     use rustc::ty::util::IntTypeExt;
                                                            //     let ty = const_int.int_type().to_ty(self.tcx);
                                                            //     let spirv_ty = self.to_ty(ty, spirv::StorageClass::Function);
                                                            //     let value = const_int.to_u128_unchecked() as u32;
                                                            //     self.builder.constant_u32(spirv_ty.word, value)
                                                            // }
                                                            // ConstValue::Bool(b) => self.constant_u32(b as u32).word,
                                                            // ConstValue::Float(const_float) => {
                                                            //     use rustc::infer::unify_key::ToType;
                                                            //     let value: f32 = unsafe { ::std::mem::transmute(const_float.bits as u32) };
                                                            //     let ty = const_float.ty.to_type(self.tcx);
                                                            //     let spirv_ty = self.to_ty(ty, spirv::StorageClass::Function);
                                                            //     self.builder.constant_f32(spirv_ty.word, value)
                                                            // }
        };
        let spirv_expr = Value::new(spirv_val);
        self.const_cache.insert(*const_val, spirv_expr);
        spirv_expr
    }
    pub fn to_ty(
        &mut self,
        ty: rustc::ty::Ty<'tcx>,
        storage_class: spirv::StorageClass,
    ) -> Ty<'tcx> {
        use rustc::ty::TypeVariants;
        use syntax::ast::{IntTy, UintTy};
        let ty = match ty.sty {
            TypeVariants::TyRef(_, ty, _) => {
                let t = ty::TypeAndMut {
                    ty,
                    mutbl: rustc::hir::Mutability::MutMutable,
                };
                self.tcx.mk_ptr(t)
            }
            _ => ty,
        };
        let is_ptr = match ty.sty {
            TypeVariants::TyRawPtr(_) => true,
            _ => false,
        };
        // TODO: Should integer always be 32bit wide?
        let ty = match ty.sty {
            TypeVariants::TyInt(_) => self.tcx.mk_ty(TypeVariants::TyInt(IntTy::I32)),
            TypeVariants::TyUint(_) => self.tcx.mk_ty(TypeVariants::TyUint(UintTy::U32)),
            TypeVariants::TyBool => self.tcx.mk_ty(TypeVariants::TyUint(UintTy::U32)),
            _ => ty,
        };

        if let Some(ty) = self
            .ty_cache
            .get(ty)
            .or_else(|| self.ty_ptr_cache.get(&(ty, storage_class)))
        {
            return *ty;
        }
        let spirv_type: Ty = match ty.sty {
            // TODO: Proper TyNever
            TypeVariants::TyNever => {
                let ty = self.tcx.mk_nil();
                self.to_ty(ty, storage_class)
            }
            //TypeVariants::TyBool => self.builder.type_bool().construct_ty(ty),
            TypeVariants::TyInt(int_ty) => self.builder.type_int(32, 1).construct_ty(ty),
            TypeVariants::TyUint(uint_ty) => self.builder.type_int(32, 0).construct_ty(ty),
            TypeVariants::TyFloat(f_ty) => {
                use syntax::ast::FloatTy;
                match f_ty {
                    FloatTy::F32 => self.builder.type_float(32).construct_ty(ty),
                    FloatTy::F64 => panic!("f64 is not supported"),
                }
            }
            TypeVariants::TyTuple(slice) if slice.len() == 0 => {
                self.builder.type_void().construct_ty(ty)
            }
            TypeVariants::TyTuple(slice) => {
                let field_ty_spirv: Vec<_> = slice
                    .iter()
                    .map(|ty| self.to_ty(ty, storage_class).word)
                    .collect();

                let spirv_struct = self.builder.type_struct(&field_ty_spirv);
                spirv_struct.construct_ty(ty)
            }
            TypeVariants::TyFnPtr(sig) => {
                let ty = self.tcx.normalize_erasing_late_bound_regions(
                    ty::ParamEnv::reveal_all(),
                    &sig.output(),
                );
                let ret_ty = self.to_ty(ty, storage_class);
                let input_ty: Vec<_> = sig
                    .inputs()
                    .skip_binder()
                    .iter()
                    .map(|ty| self.to_ty(ty, storage_class).word)
                    .collect();
                self.builder
                    .type_function(ret_ty.word, &input_ty)
                    .construct_ty(ty)
            }
            TypeVariants::TyRawPtr(type_and_mut) => {
                let inner = self.to_ty(type_and_mut.ty, storage_class);
                self.builder
                    .type_pointer(None, storage_class, inner.word)
                    .construct_ty(ty)
            }
            TypeVariants::TyParam(_) => panic!("TyParam should have been monomorphized"),
            TypeVariants::TyAdt(adt, substs) => {
                //let mono_substs = mtx.monomorphize(&substs);
                let mono_substs = substs;
                match adt.adt_kind() {
                    ty::AdtKind::Enum => {
                        if let Some(e) = Enum::from_ty(self.tcx, ty) {
                            let discr_ty_spirv = self.to_ty(e.discr_ty, storage_class);
                            let mut field_ty_spirv: Vec<_> = adt
                                .variants
                                .iter()
                                .map(|variant| {
                                    let variant_field_ty: Vec<
                                        _,
                                    > = variant
                                        .fields
                                        .iter()
                                        .map(|field| {
                                            let ty = field.ty(self.tcx, mono_substs);
                                            self.to_ty(ty, storage_class).word
                                        }).collect();
                                    let spirv_struct = self.builder.type_struct(&variant_field_ty);
                                    if self.debug_symbols {
                                        for (index, field) in variant.fields.iter().enumerate() {
                                            self.builder.member_name(
                                                spirv_struct,
                                                index as u32,
                                                field.ident.as_str().to_string(),
                                            );
                                        }
                                        self.name_from_def_id(variant.did, spirv_struct);
                                    }
                                    spirv_struct
                                }).collect();
                            field_ty_spirv.push(discr_ty_spirv.word);

                            let spirv_struct = self.builder.type_struct(&field_ty_spirv);
                            if self.debug_symbols {
                                for (index, field) in adt.all_fields().enumerate() {
                                    self.builder.member_name(
                                        spirv_struct,
                                        index as u32,
                                        field.ident.as_str().to_string(),
                                    );
                                }
                                self.name_from_def_id(adt.did, spirv_struct);
                            }
                            spirv_struct.construct_ty(ty)
                        } else {
                            // If we have an enum, but without an layout it should be enum Foo {}
                            // TODO: Empty struct correct?
                            //self.builder.type_struct(&[])
                            let ty = self.tcx.mk_ty(TypeVariants::TyNever);
                            self.to_ty(ty, storage_class)
                            //self.builder.type_opaque(self.tcx.item_name(adt.did).as_ref())
                        }
                    }
                    ty::AdtKind::Struct => {
                        let attrs = self.tcx.get_attrs(adt.did);
                        let intrinsic = IntrinsicType::from_ty(self.tcx, ty)
                            .map(|intrinsic| intrinsic.contruct_ty(storage_class, self));

                        intrinsic.unwrap_or_else(|| {
                            // let fields_ty = adt.all_fields()
                            //     .map(|field| field.ty(self.tcx, substs))
                            //     .filter(|ty| !ty.is_phantom_data())
                            //     .collect_vec();

                            let attrs = self.tcx.get_attrs(adt.did);
                            let needs_block = ::extract_attr(&attrs, "spirv", |s| match s {
                                "Input" => Some(true),
                                "Output" => Some(true),
                                _ => None,
                            }).get(0)
                            .is_some();
                            let field_ty: Vec<_> = adt
                                .all_fields()
                                .map(|f| f.ty(self.tcx, mono_substs))
                                .filter(|ty| !ty.is_phantom_data())
                                .collect();
                            let field_ty_spirv: Vec<_> = field_ty
                                .iter()
                                .map(|ty| self.to_ty(ty, storage_class).word)
                                .collect();
                            let spirv_struct = self.builder.type_struct(&field_ty_spirv);
                            if let Some(layout) = ::std430_layout(self.tcx, ty) {
                                layout
                                    .offsets()
                                    .iter()
                                    .enumerate()
                                    .for_each(|(idx, &offset)| {
                                        self.builder.member_decorate(
                                            spirv_struct,
                                            idx as u32,
                                            spirv::Decoration::Offset,
                                            &[rspirv::mr::Operand::LiteralInt32(offset as u32)],
                                        );
                                    });
                            }
                            // TODO: Proper input
                            if let Some(uniform) = ::Uniform::new(self.tcx, ty) {
                                self.builder
                                    .decorate(spirv_struct, spirv::Decoration::Block, &[]);
                            }
                            if needs_block {
                                self.builder
                                    .decorate(spirv_struct, spirv::Decoration::Block, &[]);
                                let fields: Vec<_> = adt.all_fields().take(2).collect();
                                let location_ty = fields[1].ty(self.tcx, mono_substs);
                                let location_index = ::extract_location(self.tcx, location_ty)
                                    .expect("location index");

                                //println!("atrs {:?}", self.tcx.get_attrs(location_id));
                                //let variable_ty = fields[0].ty(self.tcx, mono_substs);
                                self.builder.member_decorate(
                                    spirv_struct,
                                    0,
                                    spirv::Decoration::Location,
                                    &[rspirv::mr::Operand::LiteralInt32(location_index as u32)],
                                );
                            }

                            if self.debug_symbols {
                                let fields: Vec<_> = adt
                                    .all_fields()
                                    .filter(|field| {
                                        let ty = field.ty(self.tcx, mono_substs);
                                        !ty.is_phantom_data()
                                    }).collect();
                                for (index, field) in fields.iter().enumerate() {
                                    self.builder.member_name(
                                        spirv_struct,
                                        index as u32,
                                        field.ident.as_str().to_string(),
                                    );
                                }
                            }
                            self.name_from_def_id(adt.did, spirv_struct);
                            spirv_struct.construct_ty(ty)
                        })
                    }
                    ref r => unimplemented!("{:?}", r),
                }
            }
            TypeVariants::TyArray(ty, length) => {
                let spirv_lenth = self.constant(length);
                let spirv_ty = self.to_ty(ty, storage_class);
                self.builder
                    .type_array(spirv_ty.word, spirv_lenth.word)
                    .construct_ty(ty)
            }
            TypeVariants::TyClosure(def_id, substs) => {
                let field_ty_spirv: Vec<_> = substs
                    .upvar_tys(def_id, self.tcx)
                    .map(|ty| self.to_ty(ty, storage_class).word)
                    .collect();

                let spirv_struct = self.builder.type_struct(&field_ty_spirv);
                spirv_struct.construct_ty(ty)
            }
            ref r => unimplemented!("{:?}", r),
        };
        self.name_from_str(&format!("{:?}", ty), spirv_type.word);
        if is_ptr {
            self.ty_ptr_cache.insert((ty, storage_class), spirv_type);
        } else {
            self.ty_cache.insert(ty, spirv_type);
        }
        spirv_type
    }

    pub fn to_ty_as_ptr<'gcx>(
        &mut self,
        ty: ty::Ty<'tcx>,
        storage_class: spirv::StorageClass,
    ) -> Ty<'tcx> {
        let t = ty::TypeAndMut {
            ty,
            mutbl: rustc::hir::Mutability::MutMutable,
        };
        let ty_ptr = self.tcx.mk_ptr(t);
        self.to_ty(ty_ptr, storage_class)
    }
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
        if self.debug_symbols {
            //self.builder.name(id, self.tcx.item_name(def_id).as_ref());
            self.builder.name(id, self.tcx.item_path_str(def_id));
        }
    }
    pub fn name_from_str(&mut self, name: &str, id: spirv::Word) {
        if self.debug_symbols {
            self.builder.name(id, name);
        }
    }
    pub fn build_module(self) -> rspirv::mr::Module {
        let mut module = self.builder.module();
        if let Some(header) = module.header.as_mut() {
            header.set_version(1, 0);
        }
        module
    }
    // TODO: Hack to get the correct type for PerVertex
    pub fn get_per_fragment(&mut self, ty: ty::Ty<'tcx>) -> Variable<'tcx> {
        self.per_fragment.unwrap_or_else(|| {
            let is_fragment = ty
                .ty_to_def_id()
                .map(|def_id| {
                    let attrs = self.tcx.get_attrs(def_id);
                    ::extract_attr(&attrs, "spirv", |s| match s {
                        "PerFragment" => Some(true),
                        _ => None,
                    })
                }).is_some();
            let spirv_ty = self.to_ty(ty, spirv::StorageClass::Input);
            self.builder
                .decorate(spirv_ty.word, spirv::Decoration::Block, &[]);
            assert!(is_fragment, "Not a fragment");
            let var = Variable::alloca(self, ty, spirv::StorageClass::Input);
            // TODO
            self.builder.member_decorate(
                spirv_ty.word,
                0,
                spirv::Decoration::BuiltIn,
                &[rspirv::mr::Operand::BuiltIn(spirv::BuiltIn::FragCoord)],
            );
            self.per_fragment = Some(var);
            var
        })
    }
    pub fn get_compute(&mut self, ty: ty::Ty<'tcx>) -> Variable<'tcx> {
        use rustc::ty::TypeVariants;
        let is_compute = ty
            .ty_to_def_id()
            .map(|def_id| {
                let attrs = self.tcx.get_attrs(def_id);
                ::extract_attr(&attrs, "spirv", |s| match s {
                    "Compute" => Some(true),
                    _ => None,
                })
            }).is_some();
        assert!(is_compute, "Not Compute");
        self.compute.unwrap_or_else(|| {
            let spirv_ty = self.to_ty(ty, spirv::StorageClass::Input);
            let var = Variable::alloca(self, ty, spirv::StorageClass::Input);
            // TODO
            self.builder
                .decorate(spirv_ty.word, spirv::Decoration::Block, &[]);
            self.builder.member_decorate(
                spirv_ty.word,
                0,
                spirv::Decoration::BuiltIn,
                &[rspirv::mr::Operand::BuiltIn(
                    spirv::BuiltIn::LocalInvocationIndex,
                )],
            );
            self.builder.member_decorate(
                spirv_ty.word,
                1,
                spirv::Decoration::BuiltIn,
                &[rspirv::mr::Operand::BuiltIn(
                    spirv::BuiltIn::GlobalInvocationId,
                )],
            );
            self.compute = Some(var);
            var
        })
    }
    pub fn get_per_vertex(&mut self, ty: ty::Ty<'tcx>) -> Variable<'tcx> {
        use rustc::ty::TypeVariants;
        assert!(::is_per_vertex(self.tcx, ty), "Not PerVertex");
        self.per_vertex.unwrap_or_else(|| {
            let struct_ty = match ty.sty {
                TypeVariants::TyRef(_, ty_and_mut, _) => ty_and_mut,
                _ => unreachable!(),
            };
            let spirv_ty = self.to_ty(struct_ty, spirv::StorageClass::Output);
            self.builder
                .decorate(spirv_ty.word, spirv::Decoration::Block, &[]);
            let var = Variable::alloca(self, struct_ty, spirv::StorageClass::Output);
            self.builder.member_decorate(
                spirv_ty.word,
                0,
                spirv::Decoration::BuiltIn,
                &[rspirv::mr::Operand::BuiltIn(spirv::BuiltIn::Position)],
            );
            self.builder.member_decorate(
                spirv_ty.word,
                1,
                spirv::Decoration::BuiltIn,
                &[rspirv::mr::Operand::BuiltIn(spirv::BuiltIn::PointSize)],
            );
            self.per_vertex = Some(var);
            var
        })
    }
    pub fn new(tcx: ty::TyCtxt<'a, 'tcx, 'tcx>) -> CodegenCx<'a, 'tcx> {
        let mut builder = Builder::new();
        builder.capability(spirv::Capability::Shader);
        let glsl_ext_id = builder.ext_inst_import("GLSL.std.450");
        builder.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        CodegenCx {
            bool_ty: builder.type_bool(),
            debug_symbols: true,
            builder,
            compute: None,
            per_vertex: None,
            per_fragment: None,
            ty_cache: HashMap::new(),
            ty_ptr_cache: HashMap::new(),
            const_cache: HashMap::new(),
            forward_fns: HashMap::new(),
            intrinsic_fns: HashMap::new(),
            tcx,
            glsl_ext_id,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct MergeBlock {
    pub branch_header: mir::BasicBlock,
    pub merge_block: mir::BasicBlock,
}

pub fn find_merge_blocks(petmir: &PetMir) -> Vec<MergeBlock> {
    let block_iter = DfsPostOrder::new(&petmir.graph, petmir.start_block()).iter(&petmir.graph);
    block_iter
        .filter_map(|bb| {
            let block_data = &petmir.mir.basic_blocks()[bb];
            match block_data.terminator().kind {
                mir::TerminatorKind::SwitchInt {
                    ref discr,
                    switch_ty,
                    ref targets,
                    ..
                } => {
                    let merge_block =
                        find_merge_block(petmir, bb, targets).expect("no merge block");
                    Some(MergeBlock {
                        branch_header: bb,
                        merge_block,
                    })
                }
                _ => None,
            }
        }).collect()
}
pub fn find_merge_block(
    petmir: &PetMir,
    root: mir::BasicBlock,
    targets: &[mir::BasicBlock],
) -> Option<mir::BasicBlock> {
    let dominators = petmir.mir.dominators();
    let ret_block = petmir.return_block();
    let reversed = Reversed(&petmir.graph);
    let mut set: HashSet<_> = Bfs::new(reversed, ret_block).iter(reversed).collect();
    let merge_targets: Vec<_> = targets
        .iter()
        .filter(|&bb| set.contains(bb))
        .cloned()
        .collect();
    // println!("root {:?}", root);
    // println!("targets {:?}", merge_targets);
    // println!("set {:?}", set);
    // println!("ret {:?}", ret_block);
    Bfs::new(&petmir.graph, root)
        .iter(&petmir.graph)
        .find(|&bb| {
            let not_root = bb != root;
            let reachable_by_branches = merge_targets
                .iter()
                .all(|&target| petmir.is_reachable(target, bb));
            // println!("{:?}", bb);
            // println!("not root {}", not_root);
            // println!("reach {}", reachable_by_branches);

            not_root && reachable_by_branches
        })
}

pub fn compute_control_flow(petmir: &PetMir, loops: &Loops) -> BTreeMap<mir::BasicBlock, Cfg> {
    let mut map = BTreeMap::new();
    let block_iter = Dfs::new(&petmir.graph, petmir.start_block()).iter(&petmir.graph);
    // Temporary workaround for a driver bug. Spirv is underspecified and branches that are inside
    // a loop that have an exit node should not be treated as selection merge
    let mut no_selection_merge_needed = Vec::new();
    block_iter.for_each(|bb| {
        let block_data = &petmir.mir.basic_blocks()[bb];
        match block_data.terminator().kind {
            mir::TerminatorKind::Goto { target }
            | mir::TerminatorKind::Call {
                destination: Some((_, target)),
                ..
            } => {
                if let Some(&back_edge) = loops.get(&bb) {
                    let dominators = petmir.mir.dominators();
                    let (block, merge_block) = Bfs::new(&petmir.graph, bb)
                        .iter(&petmir.graph)
                        .filter_map(|bb| {
                            let block_data = &petmir.mir.basic_blocks()[bb];
                            match block_data.terminator().kind {
                                mir::TerminatorKind::SwitchInt { ref targets, .. } => targets
                                    .iter()
                                    .find(|&&target| !dominators.is_dominated_by(back_edge, target))
                                    .map(|&merge_block| (bb, merge_block)),
                                _ => None,
                            }
                        }).nth(0)
                        .expect("Unable to find merge block");

                    no_selection_merge_needed.push(block);
                    let cfg = Cfg {
                        header: bb,
                        merge_block,
                        loop_block: Some(Loop {
                            continue_block: back_edge,
                            back_edge,
                        }),
                    };
                    map.insert(bb, cfg);
                }
            }
            mir::TerminatorKind::SwitchInt {
                ref discr,
                switch_ty,
                ref targets,
                ..
            } => {
                if let Some(&back_edge) = loops.get(&bb) {
                    let continue_block = *targets
                        .iter()
                        .find(|&&target| petmir.is_reachable(target, back_edge))
                        .expect("continue");
                    let merge_block = *targets
                        .iter()
                        .find(|&&target| !petmir.is_reachable(target, back_edge))
                        .expect("merge");
                    let cfg = Cfg {
                        header: bb,
                        merge_block,
                        loop_block: Some(Loop {
                            continue_block: back_edge,
                            back_edge,
                        }),
                    };
                    map.insert(bb, cfg);
                } else {
                    let merge_block = find_merge(petmir, &loops, bb, targets);
                    let cfg = Cfg {
                        header: bb,
                        merge_block,
                        loop_block: None,
                    };
                    map.insert(bb, cfg);
                }
            }
            _ => (),
        }
    });
    for block in no_selection_merge_needed {
        map.remove(&block);
    }
    map
}

#[derive(Debug, Clone)]
pub struct Loop {
    pub back_edge: mir::BasicBlock,
    pub continue_block: mir::BasicBlock,
}
#[derive(Debug, Clone)]
pub struct Cfg {
    pub header: mir::BasicBlock,
    pub merge_block: mir::BasicBlock,
    pub loop_block: Option<Loop>,
}

pub fn find_merge(
    petmir: &PetMir,
    loops: &HashMap<mir::BasicBlock, mir::BasicBlock>,
    block: mir::BasicBlock,
    targets: &[mir::BasicBlock],
) -> mir::BasicBlock {
    debug!("");
    //println!("{:?}", block);
    let mut merge_count: IndexVec<mir::BasicBlock, u32> =
        IndexVec::from_elem_n(0, petmir.mir.basic_blocks().len());
    for &target in targets {
        let mut dfs = Dfs::new(&petmir.graph, target);
        for &bb in loops.keys() {
            dfs.discovered.visit(bb);
        }
        for bb in dfs.iter(&petmir.graph) {
            merge_count[bb] += 1;
        }
    }

    let max = merge_count.iter().cloned().fold(0, ::std::cmp::max);
    let mut merge_block_iter = Bfs::new(&petmir.graph, block)
        .iter(&petmir.graph)
        .filter(|&bb| merge_count[bb] == max);

    let return_block = petmir.return_block();
    let merge_block = merge_block_iter
        .clone()
        .find(|&bb| petmir.is_reachable(bb, return_block))
        .unwrap_or_else(|| merge_block_iter.nth(0).expect("merge block"));

    // println!("{:?}", merge_count);
    // println!("Found merge: {:?}", merge_block);
    merge_block
}
/// Sometimes you will see two branches that share the same block. Or not all blocks
/// that are between the branch header and the merge block are dominated by the
/// header. This can be fixed by duplicating those shared blocks.
pub fn fix_overlapping_control_flow<'a, 'tcx>(
    mut mir: mir::Mir<'tcx>,
    loops: &Loops,
    merge_blocks: impl Iterator<Item = &'a Cfg>,
) -> mir::Mir<'tcx> {
    let overlapping_blocks: Vec<_> = {
        let dominators = mir.dominators();
        merge_blocks
            .map(|cfg| {
                let order = post_order_from_to(&mir, cfg.header, Some(cfg.merge_block));
                //println!("{:?}", order);
                let overlapping_bbs: Vec<_> = order
                    .into_iter()
                    .filter(|&bb| {
                        bb != cfg.merge_block
                            && !dominators.is_dominated_by(bb, cfg.header)
                            && !loops.contains_key(&bb)
                    }).collect();
                (cfg.header, overlapping_bbs)
            }).collect()
    };
    //println!("Overlapping {:#?}", overlapping_blocks);
    for (header, overlapping_bbs) in overlapping_blocks {
        let dominators = mir.dominators();
        for overlapping_block in overlapping_bbs {
            let block = mir.basic_blocks()[overlapping_block].clone();
            let new_block = mir.basic_blocks_mut().push(block);
            let previous_blocks: Vec<_> = {
                let petmir = PetMir::from_mir(&mir);
                ControlFlowGraph::predecessors(&mir, overlapping_block)
                    .filter(|&bb| petmir.is_reachable(header ,bb))
                    //.filter(|&bb| dominators.is_dominated_by(bb, header))
                    .collect()
            };
            assert!(
                previous_blocks.len() > 0,
                "Previous blocks should not be empty for {:?}",
                overlapping_block
            );
            for previous_block in previous_blocks {
                insert_block_inbetween(&mut mir, previous_block, overlapping_block, new_block);
            }
        }
    }
    mir
}

#[derive(Debug, Clone)]
pub struct ControlFlow {
    pub merge_blocks: HashMap<mir::BasicBlock, mir::BasicBlock>,
    pub loop_headers: HashMap<mir::BasicBlock, mir::BasicBlock>,
}

#[derive(Clone)]
pub struct SpirvMir<'a, 'tcx: 'a> {
    pub def_id: hir::def_id::DefId,
    pub mir: mir::Mir<'tcx>,
    pub substs: &'tcx rustc::ty::subst::Substs<'tcx>,
    pub control_flow: BTreeMap<mir::BasicBlock, Cfg>,
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
}
fn insert_block_inbetween<'tcx>(
    mir: &mut mir::Mir<'tcx>,
    previous_block: mir::BasicBlock,
    old_block: mir::BasicBlock,
    target_block: mir::BasicBlock,
) {
    let kind = &mut mir.basic_blocks_mut()[previous_block].terminator_mut().kind;
    match kind {
        mir::TerminatorKind::Call {
            destination: Some((_, ref mut target)),
            ..
        }
        | mir::TerminatorKind::Goto { ref mut target }
        | mir::TerminatorKind::Drop { ref mut target, .. } => {
            *target = target_block;
        }
        mir::TerminatorKind::SwitchInt {
            ref mut targets, ..
        } => {
            let bb = targets
                .iter_mut()
                .find(|bb| **bb == old_block)
                .expect("Should contain merge block");
            *bb = target_block;
        }
        _ => unimplemented!("Fix prev merge block"),
    }
}
impl<'a, 'tcx> SpirvMir<'a, 'tcx> {
    pub fn mir(&self) -> &mir::Mir<'tcx> {
        &self.mir
    }
    pub fn from_mir(mcx: &::MirContext<'a, 'tcx>) -> Self {
        //println!("{:?}", mcx.def_id);
        let mut spirv_mir = mcx.mir.clone();
        ::remove_unwind(&mut spirv_mir);

        let (cfgs, loops) = {
            let petmir = PetMir::from_mir(&spirv_mir);
            let loops = petmir.compute_natural_loops();
            (compute_control_flow(&petmir, &loops), loops)
        };

        //println!("{:#?}", cfgs);
        let mut spirv_mir = fix_overlapping_control_flow(spirv_mir, &loops, cfgs.values());
        let mut cfgs = {
            let petmir = PetMir::from_mir(&spirv_mir);
            let loops = petmir.compute_natural_loops();
            compute_control_flow(&petmir, &loops)
        };
        //println!("{:?}", cfgs.values().collect::<Vec<_>>());
        let order = post_order_from_to(&spirv_mir, mir::START_BLOCK, None);
        // TODO: Clean this up: We need to iterate over the cfgs in post order
        order.iter().for_each(|bb| {
            let cfg = if let Some(cfg) = cfgs.get_mut(bb) {
                cfg
            } else {
                return;
            };
            let terminator = mir::Terminator {
                source_info: mir::SourceInfo {
                    span: DUMMY_SP,
                    scope: mir::OUTERMOST_SOURCE_SCOPE,
                },
                kind: mir::TerminatorKind::Goto {
                    target: cfg.merge_block,
                },
            };
            let dominators = spirv_mir.dominators();
            let pre: Vec<_> = ControlFlowGraph::predecessors(&spirv_mir, cfg.merge_block).collect();
            let previous_blocks: Vec<_> =
                ControlFlowGraph::predecessors(&spirv_mir, cfg.merge_block)
                    .filter(|&bb| {
                        let dominated = dominators.is_dominated_by(bb, cfg.header);
                        //println!("dom {:?} {:?} {}", bb, block, dominated);
                        dominated
                    }).collect();
            assert!(
                previous_blocks.len() > 0,
                "Previous blocks should not be empty for {:?}",
                cfg.merge_block,
            );
            let goto_data = mir::BasicBlockData::new(Some(terminator));
            let goto_block = spirv_mir.basic_blocks_mut().push(goto_data);
            for previous_block in previous_blocks {
                insert_block_inbetween(&mut spirv_mir, previous_block, cfg.merge_block, goto_block);
            }
            cfg.merge_block = goto_block;
        });
        //println!("{:#?}", cfgs);
        {
            //     let mut file =
            //         ::std::fs::File::create("/home/maik/projects/rlsl/issues/mir/mir-before.dot")
            //             .expect("graph");
            //     let graph = PetMir::from_mir(mcx.mir);
            //     graph.export(&mut file);
            // let mut file = ::std::fs::File::create("/home/maik/projects/rlsl/issues/mir/mirfixed.dot")
            //     .expect("graph");
            // let graph = PetMir::from_mir(&spirv_mir);
            // graph.export(&mut file);
        }
        cfgs.values().for_each(|cfg| {
            let dominators = spirv_mir.dominators();
            //println!("cfg {:?}", cfg.header);
            //println!(
            //    "dom {:?}",
            //    dominators.dominators(cfg.header).collect::<Vec<_>>()
            //);
        });
        SpirvMir {
            mir: spirv_mir,
            control_flow: cfgs,
            substs: mcx.substs,
            tcx: mcx.tcx,
            def_id: mcx.def_id,
        }
    }

    pub fn monomorphize<T>(&self, t: &T) -> T
    where
        T: ty::TypeFoldable<'tcx>,
    {
        self.tcx
            .subst_and_normalize_erasing_regions(self.substs, ty::ParamEnv::reveal_all(), t)
    }
}

#[derive(Clone)]
pub struct MirContext<'a, 'tcx: 'a> {
    pub def_id: hir::def_id::DefId,
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub mir: &'a mir::Mir<'tcx>,
    pub substs: &'tcx subst::Substs<'tcx>,
}
impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn mir(&self) -> &'a mir::Mir<'tcx> {
        self.mir
    }
    pub fn monomorphize<T>(&self, t: &T) -> T
    where
        T: ty::TypeFoldable<'tcx>,
    {
        self.tcx
            .subst_and_normalize_erasing_regions(self.substs, ty::ParamEnv::reveal_all(), t)
    }
}

pub fn save_module<P: AsRef<Path>>(module: &rspirv::mr::Module, file_name: P) {
    use rspirv::binary::Assemble;
    use std::fs::File;
    use std::io::Write;
    use std::mem::size_of;
    let bytes: Vec<u8> = module
        .assemble()
        .iter()
        .flat_map(|val| (0..size_of::<u32>()).map(move |i| ((val >> (8 * i)) & 0xff) as u8))
        .collect();
    let mut file = File::create(file_name.as_ref()).expect("file");
    file.write(&bytes);
}
