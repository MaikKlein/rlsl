use std::collections::HashMap;
use rustc;
use super::*;
pub struct SpirvTyCache<'a> {
    pub ty_cache: HashMap<rustc::ty::Ty<'a>, SpirvTy>,
}
use rustc::ty;
impl<'tcx> SpirvTyCache<'tcx> {
    pub fn new() -> Self {
        SpirvTyCache {
            ty_cache: HashMap::new(),
        }
    }
    pub fn from_ty(&mut self, builder: &mut Builder, ty: rustc::ty::Ty<'tcx>) -> SpirvTy {
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
                    .map(|ty| self.from_ty(builder, ty).word)
                    .collect();
                builder.type_function(ret_ty.word, &input_ty).into()
            }
            TypeVariants::TyRawPtr(type_and_mut) => {
                let inner = self.from_ty(builder, type_and_mut.ty);
                builder
                    .type_pointer(None, spirv::StorageClass::Function, inner.word)
                    .into()
            }
            ref r => unimplemented!("{:?}", r),
        };
        self.ty_cache.insert(ty, spirv_type);
        spirv_type
    }

    pub fn from_ty_as_ptr<'a, 'gcx>(
        &mut self,
        builder: &mut Builder,
        tcx: ty::TyCtxt<'a, 'gcx, 'tcx>,
        ty: ty::Ty<'tcx>,
    ) -> SpirvTy {
        let t = ty::TypeAndMut {
            ty,
            mutbl: rustc::hir::Mutability::MutMutable,
        };
        let ty_ptr = tcx.mk_ptr(t);
        self.from_ty(builder, ty_ptr)
    }
}
