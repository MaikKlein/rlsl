#![feature(no_core)]
#![feature(lang_items)]
#![no_std]
#![no_core]
#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
pub trait Copy { }

struct Vec2{
    x: f32,
    y: f32
}

fn test(){

}

fn main() {
    let v = Vec2{x: 1.0, y: 2.0};
}
