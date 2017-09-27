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



enum Test {
    A(f32),
    B(u32, u32),
    C,
}
#[spirv(vertex)]
fn vert2(v: Vec2<f32>, u: u32) {}
#[spirv(vertex)]
fn vert(v: Vec2<f32>, f: f32) {
    //let t = Test::A(1.0f32);
    let t1 = Test::A(1.0f32);
    //if let Test::A(f) = t1 {}
    if 1.0f32 > 1.0{
        let i = 4.0f32;
    }
    //let i = Some(1.0f32);
    //    let v = Vec2 { x: 1.0f32, y: 2.0 };
    //    let v1 = v;
    //    let t = Test{x: 1.0, y: 2};
    //    let f1 = v.x;
    //    //let f1 = v.dot(v1);
    //    let mut b = Bar{t: Test{v: v}};
    //    let f2 = b.t.v.y;
    //    b.t.v.x = 1.0;
    //let f4: f32 = if f1 > 1.0 { 1.0 } else { 2.0 };
    //    let f3 = b.t.v.y;
    //for i in (0u32 .. 100){}
}
fn main() {}
