#![feature(custom_attribute, attr_literals)]

use std::ops::Add;
#[spirv(Vec2)]
#[repr(C)]
#[derive(Copy, Clone)]
struct Vec2 {
    x: f32,
    y: f32,
}
//
trait Foo{
    fn foo() -> f32;
}
impl Foo for Vec2{
    fn foo() -> f32{
       4.0
    }
}
impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, other: Vec2) -> Vec2 {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
fn test(f: f32, f1: f32) -> f32 {
    f + f1
}
fn vert() {
    let f = 4.0;
    let f1 = 5.0;
    let f3 = Vec2::foo(); // Not visible
    let f2 = test(f, f1); // Visible
    let f3 = test(f2, f1); // Visible
    let v = Vec2 { x: 1.0, y: 2.0 };
    let v1 = v.clone();
   let v1 = v + v; // Add impl NOT visible
}
fn main() {}
