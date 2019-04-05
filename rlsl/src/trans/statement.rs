use rustc::mir;
use Enum;
use FunctionCx;
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
            .ty(&visitor.mcx.mir().local_decls, visitor.mcx.tcx)
            .to_ty(visitor.mcx.tcx);
        let ty = visitor.mcx.monomorphize(&ty);
        let e = Enum::from_ty(visitor.mcx.tcx, ty).expect("Enum");
        let discr_ty_spirv_ptr = visitor.to_ty_as_ptr_fn(e.discr_ty);

        let spirv_var = *visitor.vars.get(&place.local().expect("should be local")).expect("Local");

        let discr_index = e.index;
        let index = visitor.constant_u32(discr_index as u32).word;

        // FIXME check for the real type
        let variant = visitor.scx.constant_u32(variant_index.as_u32()).word;
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
