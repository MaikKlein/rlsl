extern crate petgraph;
extern crate rspirv;
extern crate spirv_headers as spirv;
use petgraph::graphmap::GraphMap;
use petgraph::{Directed, Direction};
use rspirv::binary::{Assemble, Disassemble};
use rspirv::mr::{BasicBlock, Function, Instruction, Module, Operand};
use std::collections::BTreeMap;
use std::fs::{read, File};
use std::io::Write;
use std::path::Path;

macro_rules! extract {
    ($val:expr, $name:path) => {
        match $val {
            $name(inner) => inner,
            _ => panic!("Extract failed for {:?}", $val),
        }
    };
}
pub struct SpirvModule {
    pub module: Module,
    pub names: BTreeMap<u32, String>,
}
impl SpirvModule {
    pub fn get_name_fn<'module>(&'module self, f: &Function) -> Option<&'module str> {
        let id = f.def.as_ref()?.result_id?;
        self.names.get(&id).map(String::as_str)
    }
    pub fn get_name_bb<'module>(&'module self, bb: &BasicBlock) -> Option<&'module str> {
        let label = bb.label.as_ref()?;
        let return_id = label.result_id?;
        self.names.get(&return_id).map(String::as_str)
    }
    pub fn load<P: AsRef<Path>>(p: &P) -> Self {
        fn inner(p: &Path) -> SpirvModule {
            use rspirv::binary::Parser;
            use rspirv::mr::Loader;
            let module = {
                let spv_file = read(p).expect("file");
                let mut loader = Loader::new();
                {
                    let p = Parser::new(&spv_file, &mut loader);
                    p.parse().expect("parse")
                };
                loader.module()
            };
            let names = module
                .debugs
                .iter()
                .filter_map(|inst| match inst.class.opcode {
                    spirv::Op::Name => {
                        let id = extract!(inst.operands[0], Operand::IdRef);
                        let name = extract!(&inst.operands[1], Operand::LiteralString);
                        Some((id, name.clone()))
                    }
                    _ => None,
                }).collect();
            SpirvModule { names, module }
        }
        inner(p.as_ref())
    }
}

pub struct PetSpirv<'spir> {
    pub module: &'spir SpirvModule,
    pub function: &'spir Function,
    pub block_map: BTreeMap<u32, &'spir BasicBlock>,
}

pub fn export_spirv_cfg(module: &SpirvModule) {
    let mut file = File::create("test.dot").expect("file");
    writeln!(&mut file, "digraph {{");
    writeln!(&mut file, "graph [fontname=\"monospace\"];");
    writeln!(&mut file, "node [fontname=\"monospace\"];");
    writeln!(&mut file, "edge [fontname=\"monospace\"];");
    for f in &module.module.functions {
        let s = PetSpirv::new(module, f);
        s.add_fn_to_dot(&mut file);
    }
    writeln!(&mut file, "}}");
}
pub enum Terminator {
    Branch {
        target: spirv::Word,
    },
    BranchConditional {
        merge_block: Option<spirv::Word>,
        true_block: spirv::Word,
        false_block: spirv::Word,
    },
    Switch {
        merge_block: Option<spirv::Word>,
        values: Vec<spirv::Word>,
        targets: Vec<spirv::Word>,
    },
    /// Indicates that the control flow has stopped
    End,
}

impl Terminator {
    pub fn from_basic_block(bb: &BasicBlock) -> Terminator {
        let get_merge_block = || -> Option<spirv::Word> {
            let before_last = bb.instructions.get(bb.instructions.len() - 2)?;
            match before_last.class.opcode {
                spirv::Op::SelectionMerge => {
                    Some(extract!(before_last.operands[0], Operand::IdRef))
                }
                _ => None,
            }
        };
        let inst = if let Some(inst) = bb.instructions.last() {
            inst
        } else {
            return Terminator::End;
        };
        match inst.class.opcode {
            spirv::Op::Switch => {
                let default = extract!(inst.operands[1], Operand::IdRef);
                let merge_block = get_merge_block();
                let mut values: Vec<u32> = inst
                    .operands
                    .iter()
                    .skip(2)
                    .step_by(2)
                    .map(|operand| *extract!(operand, Operand::LiteralInt32))
                    .collect();
                let mut targets: Vec<spirv::Word> = inst
                    .operands
                    .iter()
                    .skip(3)
                    .step_by(2)
                    .map(|operand| *extract!(operand, Operand::IdRef))
                    .collect();
                targets.push(default);
                Terminator::Switch {
                    merge_block,
                    values,
                    targets,
                }
            }
            spirv::Op::BranchConditional => {
                let merge_block = get_merge_block();
                let true_block = extract!(inst.operands[1], Operand::IdRef);
                let false_block = extract!(inst.operands[2], Operand::IdRef);
                Terminator::BranchConditional {
                    merge_block,
                    true_block,
                    false_block,
                }
            }
            spirv::Op::Branch => {
                let target = extract!(inst.operands[0], Operand::IdRef);
                Terminator::Branch { target }
            }
            _ => Terminator::End,
        }
    }
    pub fn successors(&self) -> impl Iterator<Item = spirv::Word> {
        match self {
            Terminator::Switch { ref targets, .. } => targets.clone(),
            Terminator::Branch { target } => vec![*target],
            Terminator::BranchConditional {
                true_block,
                false_block,
                ..
            } => vec![*true_block, *false_block],
            _ => Vec::new(),
        }.into_iter()
    }
}
impl<'spir> PetSpirv<'spir> {
    pub fn get_block(&self, id: u32) -> &'spir BasicBlock {
        self.block_map
            .get(&id)
            .expect(&format!("BasicBlock {}", id))
    }
    pub fn add_fn_to_dot(&self, write: &mut impl Write) {
        let fn_name = self.module.get_name_fn(&self.function).unwrap_or("Unknown");
        let fn_id = self.function.def.as_ref().unwrap().result_id.unwrap();
        let entry = self.block_map.keys().nth(0).expect("entry key");
        writeln!(
            write,
            "{id} [shape=\"box\", label={name:?}];",
            id = fn_id,
            name = fn_name
        );
        writeln!(write, "{} -> {}", fn_id, entry);

        for (id, block) in &self.block_map {
            let default = format!("{}", id);
            let name = self.module.get_name_bb(block).unwrap_or(&default);
            writeln!(write, "  {id} [shape=none, label=<", id = id,);
            writeln!(write, "\t<table>");
            writeln!(
                write,
                "\t\t<tr><td align=\"center\" bgcolor=\"gray\" colspan=\"1\">{name} {id}</td></tr>",
                id = id,
                name = name
            );
            writeln!(write, "\t\t<tr><td align=\"left\" balign=\"left\">");
            for inst in &block.instructions {
                writeln!(write, "\t\t\t{}<br/>", inst.disassemble());
            }
            writeln!(write, "\t</td></tr></table>>];");
        }
        self.traverse(write);
        //writeln!(write, "}}");
    }
    pub fn get_label(&self, id: u32) -> String {
        self.module
            .names
            .get(&id)
            .cloned()
            .unwrap_or(format!("{}", id))
    }
    pub fn traverse(&self, write: &mut impl Write) {
        use petgraph::visit::Dfs;
        let mut map = GraphMap::new();
        for &node in self.block_map.keys() {
            map.add_node(node);
        }
        if let Some(start_block) = self.function.basic_blocks.first() {
            let label = start_block.label.as_ref().unwrap();
            let id = label.result_id.unwrap();
            self.traverse_from(&mut map, id);
            let mut dfs = Dfs::new(&map, id);
            while let Some(node) = dfs.next(&map) {
                let node_label = self.get_label(node);
                for bb in map.neighbors_directed(node, Direction::Outgoing) {
                    let bb_label = self.get_label(bb);
                    writeln!(write, "  {node} -> {target}", node = node, target = bb);
                }
            }
        }
    }
    fn traverse_from(&self, map: &mut GraphMap<u32, (), Directed>, root_id: u32) {
        let root = self.get_block(root_id);
        let terminator = Terminator::from_basic_block(root);
        for bb in terminator.successors() {
            map.add_edge(root_id, bb, ());
            self.traverse_from(map, bb);
        }
    }
    pub fn new(module: &'spir SpirvModule, function: &'spir Function) -> Self {
        let block_map = function
            .basic_blocks
            .iter()
            .filter_map(|bb| {
                let label = bb.label.as_ref()?;
                label.result_id.map(|id| (id, bb))
            }).collect();
        PetSpirv {
            module,
            function,
            block_map,
        }
    }
}
