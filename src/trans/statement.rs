use RlslVisitor;
use rustc::mir;
use rustc::ty::layout::{Integer, Layout};
use SpirvConstVal;
use rustc_const_math::ConstInt;
use rustc::ty::ParamEnv;
use rustc::traits::Reveal;
pub fn trans_statement<'scope, 'a: 'scope, 'tcx: 'a>(
    visitor: &mut RlslVisitor<'scope, 'a, 'tcx>,
    _: mir::BasicBlock,
    statement: &mir::Statement<'tcx>,
    _: mir::Location,
) {
    if let mir::StatementKind::SetDiscriminant {
        ref lvalue,
        variant_index,
    } = statement.kind
    {
        let ty = lvalue
            .ty(&visitor.mcx.mir.local_decls, visitor.mcx.tcx)
            .to_ty(visitor.mcx.tcx);
        let adt = ty.ty_adt_def().expect("Should be an enum");
        let layout = ty.layout(visitor.scx.tcx, ParamEnv::empty(Reveal::All))
            .expect("layout");
        let discr_ty_int = if let &Layout::General { discr, .. } = layout {
            discr
        } else {
            panic!("No enum layout")
        };
        let discr_ty = discr_ty_int.to_ty(&visitor.mcx.tcx, false);
        let discr_ty_spirv_ptr = visitor.to_ty_as_ptr_fn(discr_ty);

        let spirv_var = match lvalue {
            &mir::Lvalue::Local(local) => *visitor.vars.get(&local).expect("Local"),
            _ => panic!("Should be local"),
        };

        let discr_index = adt.variants.len();
        let index = visitor.constant_u32(discr_index as u32).0;

        let variant_const_val = match discr_ty_int {
            Integer::I32 => SpirvConstVal::Integer(ConstInt::U32(variant_index as u32)),
            _ => panic!(""),
        };
        let variant = visitor.constant(variant_const_val).0;
        let access = visitor
            .scx
            .builder
            .access_chain(discr_ty_spirv_ptr.word, None, spirv_var.word, &[index])
            .expect("access_chain");
        visitor
            .scx
            .builder
            .store(access, variant, None, &[])
            .expect("store");
    }
}
