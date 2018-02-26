use rustc_const_math::{ConstFloat, ConstInt};
use spirv;
use rustc::ty;
use rustc::ty::Ty;
use context::SpirvCtx;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum SpirvConstVal {
    Float(ConstFloat),
    Integer(ConstInt),
    Bool(bool),
}
#[derive(Debug, Clone)]
pub struct SpirvOperand<'tcx> {
    pub ty: Ty<'tcx>,
    pub variant: SpirvOperandVariant<'tcx>,
}

#[derive(Debug, Clone)]
pub enum SpirvOperandVariant<'tcx> {
    Variable(SpirvVar<'tcx>),
    Value(SpirvValue),
}
impl<'tcx> SpirvOperand<'tcx> {
    pub fn new(ty: Ty<'tcx>, variant: SpirvOperandVariant<'tcx>) -> SpirvOperand<'tcx> {
        SpirvOperand { ty, variant }
    }

    pub fn to_variable(self) -> Option<SpirvVar<'tcx>> {
        match self.variant {
            SpirvOperandVariant::Variable(var) => Some(var),
            _ => None,
        }
    }

    pub fn load<'a, 'b>(self, scx: &'b mut SpirvCtx<'a, 'tcx>) -> spirv::Word {
        match self.variant {
            SpirvOperandVariant::Variable(var) => {
                let spirv_ty = scx.to_ty_fn(self.ty);
                scx.builder
                    .load(spirv_ty.word, None, var.word, None, &[])
                    .expect("load")
            }
            SpirvOperandVariant::Value(expr) => expr.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SpirvLabel(pub spirv::Word);
#[derive(Copy, Clone, Debug)]
pub struct SpirvFn(pub spirv::Word);
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SpirvVar<'tcx> {
    pub word: spirv::Word,
    pub is_param: bool,
    pub ty: ty::Ty<'tcx>,
    pub storage_class: spirv::StorageClass,
}

impl<'tcx> SpirvVar<'tcx> {
    pub fn new(
        word: spirv::Word,
        is_param: bool,
        ty: ty::Ty<'tcx>,
        storage_class: spirv::StorageClass,
    ) -> Self {
        SpirvVar {
            word,
            is_param,
            ty,
            storage_class,
        }
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
#[derive(Copy, Clone, Debug, Hash)]
pub struct SpirvValue(pub spirv::Word);
#[derive(Copy, Clone, Debug, Hash, Eq, PartialOrd, PartialEq)]
pub struct SpirvTy {
    pub word: spirv::Word,
}
impl From<spirv::Word> for SpirvTy {
    fn from(word: spirv::Word) -> SpirvTy {
        SpirvTy { word }
    }
}
