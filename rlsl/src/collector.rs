use context::MirContext;
use rustc;
use rustc::mir;
use rustc::mir::mono::MonoItem;
use rustc::mir::visit::Visitor;
use rustc::ty::subst::SubstsRef;
use rustc::ty::{self, Instance, ParamEnv, TyCtxt};
use rustc_data_structures::fx::FxHashSet;
pub struct CollectCrateItems<'a, 'tcx: 'a> {
    mir: &'a mir::Mir<'tcx>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    items: Vec<MonoItem<'tcx>>,
    substs: SubstsRef<'tcx>,
}
pub fn collect_crate_items<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &mir::Mir<'tcx>,
    substs: SubstsRef<'tcx>,
) -> Vec<MonoItem<'tcx>> {
    let mut collector = CollectCrateItems {
        mir,
        tcx,
        items: Vec::new(),
        substs,
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
            let callee_ty = func.ty(self.mir, self.tcx);
            let callee_ty = self.tcx.subst_and_normalize_erasing_regions(
                self.substs,
                ty::ParamEnv::reveal_all(),
                &callee_ty,
            );
            let (def_id, substs) = match callee_ty.sty {
                ty::TyKind::FnDef(def_id, substs) => (def_id, substs),
                _ => panic!("Not a function"),
            };
            let instance =
                Instance::resolve(self.tcx, ty::ParamEnv::reveal_all(), def_id, substs).unwrap();
            self.items.push(MonoItem::Fn(instance));
        }
    }
}

/// The collector only collects items for the current crate, but we need to access
/// items in all crates (rlibs) so we need to manually find them.
pub fn trans_all_items<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    start_items: &'a FxHashSet<MonoItem<'tcx>>,
) -> FxHashSet<MonoItem<'tcx>> {
    let mut hash_set = FxHashSet::default();
    let mut uncollected_items: Vec<Vec<MonoItem<'tcx>>> = Vec::new();
    uncollected_items.push(start_items.iter().cloned().collect());
    while let Some(items) = uncollected_items.pop() {
        for item in &items {
            if let &MonoItem::Fn(ref instance) = item {
                if tcx.is_mir_available(instance.def_id()) {
                    let mir = tcx.optimized_mir(instance.def_id());
                    let new_items = collect_crate_items(tcx, &mir, instance.substs);
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
