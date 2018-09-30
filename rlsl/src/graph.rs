use petgraph;
use petgraph::graphmap::GraphMap;
use petgraph::visit::{Dfs, Reversed, Walker};
use petgraph::Directed;
use petgraph::Direction;
use rspirv::binary::Disassemble;
use rspirv::mr::{BasicBlock, Function, Instruction, Module, Operand};
use rustc::mir;
use rustc_data_structures::control_flow_graph::{iterate::post_order_from_to, ControlFlowGraph};
use spirv;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
pub struct PetMir<'a, 'tcx: 'a> {
    pub mir: &'a mir::Mir<'tcx>,
    pub graph: GraphMap<mir::BasicBlock, (), Directed>,
}
impl<'a, 'tcx> PetMir<'a, 'tcx> {
    pub fn return_block(&self) -> mir::BasicBlock {
        self.mir
            .basic_blocks()
            .iter_enumerated()
            .filter_map(|(bb, data)| match data.terminator().kind {
                mir::TerminatorKind::Return => Some(bb),
                _ => None,
            }).nth(0)
            .expect("return")
    }
    pub fn compute_natural_loops(&self) -> HashMap<mir::BasicBlock, mir::BasicBlock> {
        let dominators = self.mir.dominators();
        let mut map = HashMap::new();
        Dfs::new(&self.graph, self.start_block())
            .iter(&self.graph)
            .for_each(|bb| {
                for suc in self.graph.neighbors_directed(bb, Direction::Outgoing) {
                    if dominators.is_dominated_by(bb, suc) {
                        map.insert(suc, bb);
                    }
                }
            });
        map
    }
    /// Checks if two basic blocks are connected
    /// `from` should appear before `to`.
    pub fn is_reachable(&self, from: mir::BasicBlock, to: mir::BasicBlock) -> bool {
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
            }).nth(0)
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
