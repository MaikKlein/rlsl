#![feature(custom_attribute, attr_literals)]

use std::ops::Add;
#[spirv(Vec2)]
#[repr(C)]
//#[derive(Copy, Clone)]
pub struct Vec2<T: Copy> {
    x: T,
    y: T,
}
//impl<T: Copy> Add for Vec2<T>
//where
//    T: Add<Output = T>,
//{
//    type Output = Vec2<T>;
//    fn add(self, other: Vec2<T>) -> Vec2<T> {
//        Vec2 {
//            x: self.x + other.x,
//            y: self.y + other.y,
//        }
//    }
//}
impl Add for Vec2<f32>
{
    type Output = Vec2<f32>;
    fn add(self, other: Vec2<f32>) -> Vec2<f32> {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

#[spirv(vertex)]
fn vert() {
    let f: f32 = 2.0;
    let v = Vec2 { x: f, y: f };
    let v1 = Vec2 { x: f, y: f };
    let v2 = v + v1;
}
fn main() {}
