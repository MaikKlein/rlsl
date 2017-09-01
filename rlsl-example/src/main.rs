#![feature(custom_attribute, attr_literals)]
#![feature(no_core)]
#![feature(no_core)]
#![feature(lang_items)]
#![no_std]
#![no_core]
#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
pub trait Copy {}

#[spirv(Vec2)]
#[repr(C)]
struct Vec2 {
    x: f32,
    y: f32,
}
fn test(f: f32) -> bool{
    true
}

fn vert() {
    let f: f32 = 1.0;
    //let vec = Vec2 { x: 1.0, y: 2.0 };
    //let vec1: Vec2 = Vec2 { x: 1.0, y: 2.0 };
}
 fn main(){}
