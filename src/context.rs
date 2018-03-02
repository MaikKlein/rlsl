use rustc_const_math::{ConstFloat, ConstInt};
use rspirv;
use std::collections::HashMap;
use rustc;
use rustc::hir;
use spirv;
use rustc::ty;
use rspirv::mr::Builder;
use syntax;
use rustc::mir;
use rustc::ty::{subst, TyCtxt};
use {Enum, Variable};
use {ConstValue, Function, FunctionCall, Intrinsic, IntrinsicType, Ty, Value};
use rustc::ty::subst::Substs;
use self::hir::def_id::DefId;
use ConstructTy;

pub struct CodegenCx<'a, 'tcx: 'a> {
    per_vertex: Option<Variable<'tcx>>,
    per_fragment: Option<Variable<'tcx>>,
    pub tcx: ty::TyCtxt<'a, 'tcx, 'tcx>,
    pub builder: Builder,
    pub ty_cache: HashMap<ty::Ty<'tcx>, Ty<'tcx>>,
    pub ty_ptr_cache: HashMap<(ty::Ty<'tcx>, spirv::StorageClass), Ty<'tcx>>,
    pub const_cache: HashMap<ConstValue, Value>,
    pub forward_fns: HashMap<(hir::def_id::DefId, &'a Substs<'tcx>), Function>,
    pub intrinsic_fns: HashMap<hir::def_id::DefId, Intrinsic>,
    pub debug_symbols: bool,
    pub glsl_ext_id: spirv::Word,
}

impl<'a, 'tcx> CodegenCx<'a, 'tcx> {
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
            .or(self.intrinsic_fns
                .get(&def_id)
                .map(|&id| FunctionCall::Intrinsic(id)))
    }

    pub fn constant_f32(&mut self, mcx: MirContext<'a, 'tcx>, value: f32) -> Value {
        use std::convert::TryFrom;
        let val = ConstValue::Float(ConstFloat::from_u128(
            TryFrom::try_from(value.to_bits()).expect("Could not convert from f32 to u128"),
            syntax::ast::FloatTy::F32,
        ));
        self.constant(mcx, val)
    }

    pub fn constant_u32(&mut self, mcx: MirContext<'a, 'tcx>, value: u32) -> Value {
        let val = ConstValue::Integer(ConstInt::U32(value));
        self.constant(mcx, val)
    }

    pub fn constant(&mut self, mcx: MirContext<'a, 'tcx>, val: ConstValue) -> Value {
        if let Some(val) = self.const_cache.get(&val) {
            return *val;
        }
        let spirv_val = match val {
            ConstValue::Integer(const_int) => {
                use rustc::ty::util::IntTypeExt;
                let ty = const_int.int_type().to_ty(self.tcx);
                let spirv_ty = self.to_ty(ty, spirv::StorageClass::Function);
                let value = const_int.to_u128_unchecked() as u32;
                self.builder.constant_u32(spirv_ty.word, value)
            }
            ConstValue::Bool(b) => self.constant_u32(mcx, b as u32).word,
            ConstValue::Float(const_float) => {
                use rustc::infer::unify_key::ToType;
                let value: f32 = unsafe { ::std::mem::transmute(const_float.bits as u32) };
                let ty = const_float.ty.to_type(self.tcx);
                let spirv_ty = self.to_ty(ty, spirv::StorageClass::Function);
                self.builder.constant_f32(spirv_ty.word, value)
            }
        };
        let spirv_expr = Value::new(spirv_val);
        self.const_cache.insert(val, spirv_expr);
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
            TypeVariants::TyRef(_, type_and_mut) => {
                let t = ty::TypeAndMut {
                    ty: type_and_mut.ty,
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
            _ => ty,
        };

        if let Some(ty) = self.ty_cache
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
            TypeVariants::TyBool => self.builder.type_bool().construct_ty(ty),
            TypeVariants::TyInt(int_ty) => self.builder.type_int(32, 1).construct_ty(ty),
            TypeVariants::TyUint(uint_ty) => self.builder.type_int(32, 0).construct_ty(ty),
            TypeVariants::TyFloat(f_ty) => {
                use syntax::ast::FloatTy;
                match f_ty {
                    FloatTy::F32 => self.builder.type_float(32).construct_ty(ty),
                    FloatTy::F64 => panic!("f64 is not supported"),
                }
            }
            TypeVariants::TyTuple(slice, _) if slice.len() == 0 => {
                self.builder.type_void().construct_ty(ty)
            }
            TypeVariants::TyTuple(slice, _) => {
                let field_ty_spirv: Vec<_> = slice
                    .iter()
                    .map(|ty| self.to_ty(ty, storage_class).word)
                    .collect();

                let spirv_struct = self.builder.type_struct(&field_ty_spirv);
                spirv_struct.construct_ty(ty)
            }
            TypeVariants::TyFnPtr(sig) => {
                let ty = self.tcx
                    .erase_late_bound_regions_and_normalize(&sig.output());
                let ret_ty = self.to_ty(ty, storage_class);
                let input_ty: Vec<_> = sig.inputs()
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
                            let mut field_ty_spirv: Vec<_> = adt.variants
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
                                        })
                                        .collect();
                                    let spirv_struct = self.builder.type_struct(&variant_field_ty);
                                    if self.debug_symbols {
                                        for (index, field) in variant.fields.iter().enumerate() {
                                            self.builder.member_name(
                                                spirv_struct,
                                                index as u32,
                                                field.name.as_str().to_string(),
                                            );
                                        }
                                        self.name_from_def_id(variant.did, spirv_struct);
                                    }
                                    spirv_struct
                                })
                                .collect();
                            field_ty_spirv.push(discr_ty_spirv.word);

                            let spirv_struct = self.builder.type_struct(&field_ty_spirv);
                            if self.debug_symbols {
                                for (index, field) in adt.all_fields().enumerate() {
                                    self.builder.member_name(
                                        spirv_struct,
                                        index as u32,
                                        field.name.as_str().to_string(),
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
                        let intrinsic = IntrinsicType::from_ty(self.tcx, ty);

                        if let Some(intrinsic) = intrinsic {
                            let intrinsic_spirv = match intrinsic {
                                IntrinsicType::TyVec(ty_vec) => {
                                    let spirv_ty = self.to_ty(ty_vec.ty, storage_class);
                                    self.builder.type_vector(spirv_ty.word, ty_vec.dim as u32)
                                }
                            };
                            intrinsic_spirv.construct_ty(ty)
                        } else {
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
                            let field_ty: Vec<_> = adt.all_fields()
                                .map(|f| f.ty(self.tcx, mono_substs))
                                .filter(|ty| !ty.is_phantom_data())
                                .collect();
                            let field_ty_spirv: Vec<_> = field_ty
                                .iter()
                                .map(|ty| self.to_ty(ty, storage_class).word)
                                .collect();
                            let spirv_struct = self.builder.type_struct(&field_ty_spirv);
                            if let Some(layout) = ::std140_layout(self.tcx, ty) {
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
                            if let Some(descriptor) = ::Descriptor::new(self.tcx, ty) {
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
                                let fields: Vec<_> = adt.all_fields()
                                    .filter(|field| {
                                        let ty = field.ty(self.tcx, mono_substs);
                                        !ty.is_phantom_data()
                                    })
                                    .collect();
                                for (index, field) in fields.iter().enumerate() {
                                    self.builder.member_name(
                                        spirv_struct,
                                        index as u32,
                                        field.name.as_str().to_string(),
                                    );
                                }
                            }
                            self.name_from_def_id(adt.did, spirv_struct);
                            spirv_struct.construct_ty(ty)
                        }
                    }
                    ref r => unimplemented!("{:?}", r),
                }
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
            self.builder
                .name(id, self.tcx.def_symbol_name(def_id).name.as_ref());
        }
    }
    pub fn name_from_str(&mut self, name: &str, id: spirv::Word) {
        if self.debug_symbols {
            self.builder.name(id, name);
        }
    }
    pub fn build_module(self) {
        use rspirv::binary::Assemble;
        use std::mem::size_of;
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create("shader.spv").unwrap();
        let spirv_module = self.builder.module();
        let bytes: Vec<u8> = spirv_module
            .assemble()
            .iter()
            .flat_map(|val| (0..size_of::<u32>()).map(move |i| ((val >> (8 * i)) & 0xff) as u8))
            .collect();

        let mut loader = rspirv::mr::Loader::new();
        //let bytes = b.module().assemble_bytes();
        rspirv::binary::parse_bytes(&bytes, &mut loader);
        f.write_all(&bytes).expect("write bytes");
    }
    // TODO: Hack to get the correct type for PerVertex
    pub fn get_per_fragment(&mut self, ty: ty::Ty<'tcx>) -> Variable<'tcx> {
        self.per_fragment.unwrap_or_else(|| {
            let is_fragment = ty.ty_to_def_id()
                .map(|def_id| {
                    let attrs = self.tcx.get_attrs(def_id);
                    ::extract_attr(&attrs, "spirv", |s| match s {
                        "PerFragment" => Some(true),
                        _ => None,
                    })
                })
                .is_some();
            let spirv_ty = self.to_ty(ty, spirv::StorageClass::Input);
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
    pub fn get_per_vertex(&mut self, ty: ty::Ty<'tcx>) -> Variable<'tcx> {
        use rustc::ty::TypeVariants;
        assert!(::is_per_vertex(self.tcx, ty), "Not PerVertex");
        self.per_vertex.unwrap_or_else(|| {
            let struct_ty = match ty.sty {
                TypeVariants::TyRef(_, ty_and_mut) => ty_and_mut.ty,
                _ => unreachable!(),
            };
            let spirv_ty = self.to_ty(struct_ty, spirv::StorageClass::Output);
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
            debug_symbols: true,
            builder,
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

#[derive(Copy, Clone)]
pub struct MirContext<'a, 'tcx: 'a> {
    pub def_id: hir::def_id::DefId,
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub mir: &'a mir::Mir<'tcx>,
    pub substs: &'tcx subst::Substs<'tcx>,
}
impl<'a, 'tcx> MirContext<'a, 'tcx> {
    pub fn monomorphize<T>(&self, value: &T) -> T
    where
        T: rustc::infer::TransNormalize<'tcx>,
    {
        self.tcx.trans_apply_param_substs(self.substs, value)
    }
}
