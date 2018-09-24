extern crate petgraph;
extern crate rspirv;
extern crate spirv_headers as spirv;
use petgraph::graphmap::GraphMap;
use petgraph::{Directed, Direction};
use rspirv::binary::Disassemble;
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
    for f in &module.module.functions {
        let s = PetSpirv::new(module, f);
        s.add_fn_to_dot(&mut file);
    }
    writeln!(&mut file, "}}");
}
impl<'spir> PetSpirv<'spir> {
    pub fn get_block(&self, id: u32) -> &'spir BasicBlock {
        self.block_map.get(&id).expect(&format!("BasicBlock {}", id))
    }
    pub fn successors(&self, bb: &BasicBlock) -> Vec<u32> {
        if let Some(inst) = bb.instructions.last() {
            match inst.class.opcode {
                spirv::Op::FunctionCall => {
                    vec![]
                }
                spirv::Op::Switch => {
                    let default = extract!(inst.operands[1], Operand::IdRef);
                    let mut targets: Vec<u32> = inst.operands.iter().skip(3).step_by(2).map(|operand|{
                        *extract!(operand, Operand::IdRef)

                    }).collect();
                    targets.push(default);
                    targets
                }
                spirv::Op::BranchConditional => {
                    let id_true = extract!(inst.operands[1], Operand::IdRef);
                    let id_false = extract!(inst.operands[2], Operand::IdRef);
                    // let target_true = self.get_block(id_true);
                    // let target_false = self.get_block(id_false);
                    vec![id_true, id_false]
                }
                spirv::Op::Branch => {
                    let target_id = extract!(inst.operands[0], Operand::IdRef);
                    //let bb = self.get_block(target_id);
                    vec![target_id]
                }
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        }
    }
    pub fn add_fn_to_dot(&self, write: &mut impl Write) {
        for (id, block) in &self.block_map {
            let default = format!("{}", id);
            let name = self.module.get_name_bb(block).unwrap_or(&default);
            writeln!(write, "  {id}[label={label:?}]", id = id, label = name);
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
                    writeln!(
                        write,
                        "  {node} -> {target}",
                        node = node,
                        target = bb
                    );
                }
            }
        }
    }
    fn traverse_from(&self, map: &mut GraphMap<u32, (), Directed>, root_id: u32) {
        let root = self.get_block(root_id);
        for bb in self.successors(root) {
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
