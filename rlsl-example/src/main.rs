#![feature(custom_attribute, attr_literals)]

//use std::ops::Add;
////#[spirv(Vec2)]
////#[repr(C)]
////#[derive(Copy, Clone)]
//pub struct Vec2<T: Copy> {
//    x: T,
//    y: T,
//}
////impl<T: Copy> Add for Vec2<T>
////where
////    T: Add<Output = T>,
////{
////    type Output = Vec2<T>;
////    fn add(self, other: Vec2<T>) -> Vec2<T> {
////        Vec2 {
////            x: self.x + other.x,
////            y: self.y + other.y,
////        }
////    }
////}
//impl Add for Vec2<f32>
//{
//    type Output = Vec2<f32>;
//    fn add(self, other: Vec2<f32>) -> Vec2<f32> {
//        Vec2 {
//            x: self.x + other.x,
//            y: self.y + other.y,
//        }
//    }
//}
//
//impl Vec2<f32>{
//    #[spirv(dot)]
//    pub fn dot(self, other: Vec2<f32>) -> f32{
//        self.x * other.x + self.y * other.y
//    }
//}

struct Bar{
    t: Test
}
struct Test{
    v: Vec2<f32>
}
#[spirv(vertex)]
fn vert() {
//    let f: f32 = 2.0;
//    let v = Vec2 { x: f, y: f };
    let v = Vec2 { x: 1.0f32, y: 2.0 };
    let f1 = v.x;
    //let f1 = v.dot(v1);
    let b = Bar{t: Test{v: v}};
    let f2 = b.t.v.x;
//    let f3 = b.t.v.y;
}
fn main() {
}
