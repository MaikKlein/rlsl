use FunctionCx;
use rustc::mir;
use rustc::ty::layout::Integer;
use ConstValue;
use rustc_const_math::ConstInt;
use rustc::ty::ParamEnv;
use rustc::traits::Reveal;
use Enum;
pub fn trans_statement<'scope, 'a: 'scope, 'tcx: 'a>(
    visitor: &mut FunctionCx<'scope, 'a, 'tcx>,
    _: mir::BasicBlock,
    statement: &mir::Statement<'tcx>,
    _: mir::Location,
) {
    if let mir::StatementKind::SetDiscriminant {
        ref place,
        variant_index,
    } = statement.kind
    {
        let ty = place
            .ty(&visitor.mcx.mir.local_decls, visitor.mcx.tcx)
            .to_ty(visitor.mcx.tcx);
        let e = Enum::from_ty(visitor.mcx.tcx, ty).expect("Enum");
        let discr_ty_spirv_ptr = visitor.to_ty_as_ptr_fn(e.discr_ty);

        let spirv_var = match place {
            &mir::Place::Local(local) => *visitor.vars.get(&local).expect("Local"),
            _ => panic!("Should be local"),
        };

        let discr_index = e.index;
        let index = visitor.constant_u32(discr_index as u32).0;

        // FIXME check for the real type
        let variant_const_val = ConstValue::Integer(ConstInt::U32(variant_index as u32));
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
