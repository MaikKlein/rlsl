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

#[lang = "freeze"]
pub trait Freeze {}

#[lang = "add"]
pub trait Add<RHS = Self> {
    /// The resulting type after applying the `+` operator
    type Output;

    /// The method for the `+` operator
    fn add(self, rhs: RHS) -> Self::Output;
}

macro_rules! add_impl {
    ($($t:ty)*) => ($(
        impl Add for $t {
            type Output = $t;

            #[inline]
            fn add(self, other: $t) -> $t { self + other }
        }
    )*)
}

add_impl! { f32 }
#[spirv(Vec2)]
#[repr(C)]
struct Vec2 {
    x: f32,
    y: f32,
}
//fn test(f: f32) -> bool{
//    true
//}

fn test() -> f32{
    4.0 + 4.0
}
fn vert() {
    let f = test();
    let v = Vec2{x:1.0, y: 2.0};
}
 fn main(){}
