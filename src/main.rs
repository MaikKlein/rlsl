#![feature(rustc_private)]
extern crate rustc;
fn main() {
    let item = rustc::hir::ItemLocalId(4);
    println!("{:?}", item);
}
