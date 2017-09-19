#![feature(custom_attribute, attr_literals)]

use std::ops::Add;
#[spirv(Vec2)]
#[repr(C)]
//#[derive(Copy, Clone)]
struct Vec2<T> {
    x: T,
    y: T,
}
//pub struct Position<T>{
//    pos: T,
//}
//impl<T> Position<T>{
//    pub fn new(pos: T) -> Self{
//        Position{
//            pos
//        }
//    }
//}
//
//pub struct Ref<'a, T: 'a>{
//    f: &'a T
//}
//
//trait Foo{
//    fn foo() -> f32;
//}
//impl Foo for Vec2{
//    fn foo() -> f32{
//       4.0
//    }
//}
impl<T> Add for Vec2<T>
where T: Add<Output = T>{
    type Output = Vec2<T>;
    fn add(self, other: Vec2<T>) -> Vec2<T> {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

fn add(f: f32, f1: f32) -> f32{
    f + f1
}
fn test<T: Add<Output = T>>(v: Vec2<T>,v1: Vec2<T>) -> Vec2<T> {
    v + v1
}

#[spirv(vertex)]
fn vert() {
    let f = 1.0f32;
    let f2 = add(f, f);
    let v = Vec2 { x: f, y: f };
    let v1 = Vec2 { x: f, y: f };
    let v2 = test(v, v1);
}
fn main() {}
