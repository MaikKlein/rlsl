use rustc::mir;
use {SpirvTy, SpirvValue, SpirvVar};
use context::{MirContext, SpirvCtx};
use std::collections::HashMap;

impl<'a, 'tcx> SpirvCtx<'a, 'tcx> {
    pub fn binary_op(
        &mut self,
        mcx: MirContext<'a, 'tcx>,
        vars: &HashMap<mir::Local, SpirvVar<'tcx>>,
        spirv_ty: SpirvTy,
        op: mir::BinOp,
        l: &mir::Operand<'tcx>,
        r: &mir::Operand<'tcx>,
    ) -> SpirvValue {
        // TODO: Different types
        let left = self.load_operand(mcx, vars, l).load_raw(self, spirv_ty);
        let right = self.load_operand(mcx, vars, r).load_raw(self, spirv_ty);
        // TODO: Impl ops
        match op {
            mir::BinOp::Mul => {
                let add = self.builder
                    .fmul(spirv_ty.word, None, left, right)
                    .expect("fmul");
                SpirvValue(add)
            }
            mir::BinOp::Add => {
                let add = self.builder
                    .fadd(spirv_ty.word, None, left, right)
                    .expect("fadd");
                SpirvValue(add)
            }
            mir::BinOp::Gt => {
                let gt = self.builder
                    .ugreater_than(spirv_ty.word, None, left, right)
                    .expect("g");
                SpirvValue(gt)
            }
            mir::BinOp::Shl => {
                let shl = self.builder
                    .shift_left_logical(spirv_ty.word, None, left, right)
                    .expect("shl");
                SpirvValue(shl)
            }
            mir::BinOp::BitOr => {
                let bit_or = self.builder
                    .bitwise_or(spirv_ty.word, None, left, right)
                    .expect("bitwise or");
                SpirvValue(bit_or)
            }
            rest => unimplemented!("{:?}", rest),
        }
    }
}
