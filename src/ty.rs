use rustc_const_math::{ConstFloat, ConstInt};
use spirv;
use rustc::ty;
use context::SpirvCtx;

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum SpirvConstVal {
    Float(ConstFloat),
    Integer(ConstInt),
}
#[derive(Debug)]
pub enum SpirvOperand<'tcx> {
    Variable(SpirvVar<'tcx>),
    Value(SpirvValue),
}
impl<'tcx> SpirvOperand<'tcx> {
    pub fn is_param(&self) -> bool {
        match self {
            &SpirvOperand::Variable(ref var) => var.is_param(),
            _ => false,
        }
    }
    pub fn expect_var(self) -> SpirvVar<'tcx> {
        match self {
            SpirvOperand::Variable(var) => var,
            _ => panic!("Expected var"),
        }
    }
    pub fn load_raw<'a, 'b>(self, ctx: &'b mut SpirvCtx<'a, 'tcx>, ty: SpirvTy) -> spirv::Word {
        match self {
            SpirvOperand::Variable(var) => {
                if var.is_ptr() {
                    // If the variable is a ptr, then we need to load the value
                    ctx.builder
                        .load(ty.word, None, var.word, None, &[])
                        .expect("load")
                } else {
                    // Otherwise we can just use the value
                    var.word
                }
            }
            SpirvOperand::Value(expr) => expr.0,
        }
    }
    pub fn into_raw_word(self) -> spirv::Word {
        match self {
            SpirvOperand::Variable(var) => var.word,
            SpirvOperand::Value(expr) => expr.0,
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
