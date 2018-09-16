use std::io::Write;
use petgraph;
use petgraph::graphmap::GraphMap;
use petgraph::Directed;
use rustc::mir;
use rustc_data_structures::control_flow_graph::{iterate::post_order_from_to, ControlFlowGraph};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct PetMir<'a, 'tcx: 'a> {
    pub mir: &'a mir::Mir<'tcx>,
    pub graph: GraphMap<mir::BasicBlock, (), Directed>,
}
impl<'a, 'tcx> PetMir<'a, 'tcx> {
    /// Checks if two basic blocks are connected
    /// `from` should appear before `to`.
    pub fn is_reachable(&self, from: mir::BasicBlock, to: mir::BasicBlock) -> bool {
        use petgraph::visit::{Dfs, Reversed, Walker};
        let reversed = Reversed(&self.graph);
        Dfs::new(reversed, to)
            .iter(reversed)
            .find(|&bb| bb == from)
            .is_some()
    }
    pub fn start_block(&self) -> mir::BasicBlock {
        self.mir.start_node()
    }
    pub fn resume_block(&self, start: mir::BasicBlock) -> Option<mir::BasicBlock> {
        post_order_from_to(self.mir, start, None)
            .into_iter()
            .filter_map(|bb| {
                let data = &self.mir.basic_blocks()[bb];
                match data.terminator().kind {
                    mir::TerminatorKind::Resume => Some(bb),
                    _ => None,
                }
            })
            .nth(0)
    }
    pub fn return_block(&self, start: mir::BasicBlock) -> Option<mir::BasicBlock> {
        post_order_from_to(self.mir, start, None)
            .into_iter()
            .filter_map(|bb| {
                let data = &self.mir.basic_blocks()[bb];
                match data.terminator().kind {
                    mir::TerminatorKind::Return => Some(bb),
                    _ => None,
                }
            })
            .nth(0)
    }
    pub fn from_mir(mir: &'a mir::Mir<'tcx>) -> Self {
        let mut graph = GraphMap::new();
        for bb in mir.basic_blocks().indices() {
            for successor in mir.successors(bb) {
                graph.add_edge(bb, successor, ());
            }
        }
        PetMir { mir, graph }
    }

    pub fn export(&self, w: &mut impl Write) {
        let dot = petgraph::dot::Dot::with_config(&self.graph, &[]);
        write!(w, "{:?}", dot);
    }
}
