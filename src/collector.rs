use rustc::mir::visit::Visitor;
use rustc;
use rustc::mir;
use rustc_data_structures::fx::FxHashSet;
use rustc::mir::mono::MonoItem;
use rustc::ty::{Instance, ParamEnv, TyCtxt};
use context::MirContext;
pub struct CollectCrateItems<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    items: Vec<MonoItem<'tcx>>,
}
pub fn collect_crate_items<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &mir::Mir<'tcx>,
) -> Vec<MonoItem<'tcx>> {
    let mut collector = CollectCrateItems {
        tcx,
        items: Vec::new(),
    };
    collector.visit_mir(mir);
    collector.items
}
impl<'a, 'tcx> rustc::mir::visit::Visitor<'tcx> for CollectCrateItems<'a, 'tcx> {
    fn visit_terminator_kind(
        &mut self,
        block: mir::BasicBlock,
        kind: &mir::TerminatorKind<'tcx>,
        location: mir::Location,
    ) {
        self.super_terminator_kind(block, kind, location);
        if let &mir::TerminatorKind::Call { ref func, .. } = kind {
            if let &mir::Operand::Constant(ref constant) = func {
                if let mir::Literal::Value { ref value } = constant.literal {
                    use rustc::middle::const_val::ConstVal;
                    if let ConstVal::Function(def_id, ref substs) = value.val {
                        //let mono_substs = self.mtx.monomorphize(substs);
                        let instance = Instance::resolve(
                            self.tcx,
                            ParamEnv::empty(rustc::traits::Reveal::All),
                            def_id,
                            substs,
                        ).unwrap();
                        self.items.push(MonoItem::Fn(instance));
                    }
                }
            }
        }
    }
}

/// The collector only collects items for the current crate, but we need to access
/// items in all crates (rlibs) so we need to manually find them.
pub fn trans_all_items<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    start_items: &'a FxHashSet<MonoItem<'tcx>>,
) -> FxHashSet<MonoItem<'tcx>> {
    let mut hash_set = FxHashSet();
    let mut uncollected_items: Vec<Vec<MonoItem<'tcx>>> = Vec::new();
    uncollected_items.push(start_items.iter().cloned().collect());
    while let Some(items) = uncollected_items.pop() {
        for item in &items {
            if let &MonoItem::Fn(ref instance) = item {
                let mir = tcx.maybe_optimized_mir(instance.def_id());
                if let Some(mir) = mir {
                    let new_items = collect_crate_items(tcx, &mir);
                    if !new_items.is_empty() {
                        uncollected_items.push(new_items)
                    }
                }
                hash_set.insert(*item);
            }
        }
    }
    hash_set
}
