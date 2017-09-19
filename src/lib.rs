#![feature(rustc_private)]
#![feature(box_syntax)]

extern crate arena;
extern crate env_logger;
extern crate getopts;
#[macro_use]
extern crate log;
extern crate rspirv;
extern crate rustc;
extern crate rustc_borrowck;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_incremental;
extern crate rustc_mir;
extern crate rustc_passes;
extern crate rustc_plugin;
extern crate rustc_resolve;
extern crate rustc_trans;
extern crate spirv_headers as spirv;
#[macro_use]
extern crate syntax;
extern crate syntax_pos;
pub mod trans;
