use context::CodegenCx;
use rustc::mir;
use rustc::ty;
use spirv;
use FunctionCx;

// #[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
// pub enum ConstValue {
//     Float(ConstFloat),
//     Integer(ConstInt),
//     Bool(bool),
// }
#[derive(Debug, Clone)]
pub struct Operand<'tcx> {
    pub ty: ty::Ty<'tcx>,
    pub variant: OperandVariant<'tcx>,
}

#[derive(Debug, Clone)]
pub enum OperandVariant<'tcx> {
    Variable(Variable<'tcx>),
    Value(Value),
}
impl<'tcx> Operand<'tcx> {
    pub fn new(ty: ty::Ty<'tcx>, variant: OperandVariant<'tcx>) -> Operand<'tcx> {
        Operand { ty, variant }
    }

    pub fn to_variable(self) -> Option<Variable<'tcx>> {
        match self.variant {
            OperandVariant::Variable(var) => Some(var),
            _ => None,
        }
    }

    pub fn load<'a, 'b>(self, cx: &'b mut CodegenCx<'a, 'tcx>) -> Value {
        match self.variant {
            OperandVariant::Variable(var) => var.load(cx),
            OperandVariant::Value(value) => value,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Label(pub spirv::Word);
#[derive(Copy, Clone, Debug)]
pub struct Function(pub spirv::Word);
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Variable<'tcx> {
    pub word: spirv::Word,
    pub ty: ty::Ty<'tcx>,
    pub storage_class: spirv::StorageClass,
}

pub struct Param<'tcx> {
    pub word: spirv::Word,
    pub ty: ty::Ty<'tcx>,
}

impl<'tcx> Param<'tcx> {
    pub fn to_variable<'a>(
        &self,
        cx: &mut CodegenCx<'a, 'tcx>,
        storage_class: spirv::StorageClass,
    ) -> Variable<'tcx> {
        if ::is_ptr(self.ty) {
            let ty = ::remove_ptr_ty(self.ty);
            Variable {
                word: self.word,
                ty,
                storage_class,
            }
        } else {
            let variable = Variable::alloca(cx, self.ty, storage_class);
            let load = self.load(cx);
            variable.store(cx, load);
            variable
        }
    }
    pub fn load<'a>(&self, cx: &mut CodegenCx<'a, 'tcx>) -> Value {
        if ::is_ptr(self.ty) {
            let ty = ::remove_ptr_ty(self.ty);
            let spirv_ty = cx.to_ty(ty, spirv::StorageClass::Function);
            let load = cx
                .builder
                .load(spirv_ty.word, None, self.word, None, &[])
                .expect("Load variable");
            Value::new(load)
        } else {
            Value::new(self.word)
        }
    }

    pub fn alloca<'a>(cx: &mut CodegenCx<'a, 'tcx>, ty: ty::Ty<'tcx>) -> Param<'tcx> {
        let spirv_ty_ptr = cx.to_ty(ty, spirv::StorageClass::Function);
        let spirv_var = cx
            .builder
            .function_parameter(spirv_ty_ptr.word)
            .expect("Function param");
        Param {
            word: spirv_var,
            ty,
        }
    }
}

impl<'tcx> Variable<'tcx> {
    pub fn access_chain<'fx, 'a>(
        fx: &mut FunctionCx<'fx, 'a, 'tcx>,
        lvalue: &mir::Place<'tcx>,
    ) -> Variable<'tcx> {
        use rustc_data_structures::indexed_vec::Idx;
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
        let lvalue_ty = lvalue
            .ty(&fx.mcx.mir().local_decls, fx.scx.tcx)
            .to_ty(fx.scx.tcx);
        let variable = fx.vars.get(&local).cloned().unwrap_or_else(|| {
            let place = fx
                .references
                .get(&mir::Place::Local(local))
                .cloned()
                .expect("ref");
            Variable::access_chain(fx, &place)
        });
        indices.reverse();
        if indices.is_empty() {
            variable
        } else {
            let lvalue_ty = lvalue
                .ty(&fx.mcx.mir().local_decls, fx.scx.tcx)
                .to_ty(fx.scx.tcx);
            let lvalue_ty = fx.mcx.monomorphize(&lvalue_ty);
            let lvalue_ty = ::remove_ptr_ty(lvalue_ty);
            let spirv_ty_ptr = fx.scx.to_ty_as_ptr(lvalue_ty, variable.storage_class);
            let indices: Vec<_> = indices
                .iter()
                .map(|&i| fx.constant_u32(i as u32).word)
                .collect();
            let access = fx
                .scx
                .builder
                .access_chain(spirv_ty_ptr.word, None, variable.word, &indices)
                .expect("access_chain");
            Variable {
                word: access,
                ty: lvalue_ty,
                storage_class: variable.storage_class,
            }
        }
    }
    pub fn load<'a>(&self, cx: &mut CodegenCx<'a, 'tcx>) -> Value {
        let spirv_ty = cx.to_ty(self.ty, self.storage_class);
        let load = cx
            .builder
            .load(spirv_ty.word, None, self.word, None, &[])
            .expect("Load variable");
        Value::new(load)
    }

    pub fn store<'a>(&self, cx: &mut CodegenCx<'a, 'tcx>, value: Value) {
        cx.builder
            .store(self.word, value.word, None, &[])
            .expect("Store variable");
    }

    pub fn alloca<'a>(
        cx: &mut CodegenCx<'a, 'tcx>,
        ty: ty::Ty<'tcx>,
        storage_class: spirv::StorageClass,
    ) -> Variable<'tcx> {
        let spirv_ty_ptr = cx.to_ty_as_ptr(ty, storage_class);
        let spirv_var = cx
            .builder
            .variable(spirv_ty_ptr.word, None, storage_class, None);
        Variable {
            word: spirv_var,
            ty,
            storage_class,
        }
    }
}
#[derive(Copy, Clone, Debug, Hash)]
pub struct Value {
    pub word: spirv::Word,
}
impl Value {
    pub fn new(word: spirv::Word) -> Value {
        Value { word }
    }
}

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Ty<'tcx> {
    pub word: spirv::Word,
    pub ty: ty::Ty<'tcx>,
}

pub trait ConstructTy<'tcx> {
    fn construct_ty(self, ty: ty::Ty<'tcx>) -> Ty<'tcx>;
}
impl<'tcx> ConstructTy<'tcx> for spirv::Word {
    fn construct_ty(self, ty: ty::Ty<'tcx>) -> Ty<'tcx> {
        Ty::new(self, ty)
    }
}
impl<'tcx> Ty<'tcx> {
    pub fn new(word: spirv::Word, ty: ty::Ty<'tcx>) -> Ty<'tcx> {
        Ty { word, ty }
    }
}
