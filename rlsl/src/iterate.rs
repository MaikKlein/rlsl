use rustc_data_structures::{graph::ControlFlowGraph, indexed_vec::IndexVec};
use std::collections::VecDeque;
pub fn breadth_first_order_from<G: ControlFlowGraph>(
    graph: &G,
    start_node: G::Node,
) -> Vec<G::Node> {
    breadth_first_order_from_to(graph, start_node, None)
}

pub fn breadth_first_order_from_to<G: ControlFlowGraph>(
    graph: &G,
    start_node: G::Node,
    end_node: Option<G::Node>,
) -> Vec<G::Node> {
    let mut visited: IndexVec<G::Node, bool> = IndexVec::from_elem_n(false, graph.num_nodes());
    let mut result: Vec<G::Node> = Vec::with_capacity(graph.num_nodes());
    let mut stack: VecDeque<G::Node> = VecDeque::with_capacity(graph.num_nodes());
    if let Some(end_node) = end_node {
        visited[end_node] = true;
    }
    stack.push_front(start_node);
    visited[start_node] = true;

    while let Some(node) = stack.pop_front() {
        println!("{:?}", node);
        let v: Vec<_> = graph.successors(node).collect();
        println!("{:?}", v);
        for successor in graph.successors(node) {
            if !visited[successor] {
                visited[successor] = true;
                stack.push_back(successor);
            }
        }
        result.push(node);
    }
    result
}
