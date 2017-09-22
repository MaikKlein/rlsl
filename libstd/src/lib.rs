
#![no_std]
#![feature(macro_reexport)]
#![feature(lang_items)]
#![feature(core_intrinsics)]
#![feature(unwind_attributes)]
#![feature(core_panic)]
#![feature(prelude_import)]
#![feature(custom_attribute, attr_literals)]

#[macro_reexport(assert, assert_eq, assert_ne, debug_assert, debug_assert_eq,
debug_assert_ne, unreachable, unimplemented, write, writeln, try)]
extern crate core as __core;

pub use core::iter;
pub use core::clone;
pub use core::marker;
pub use core::ops;
pub use core::fmt;
pub use core::panicking;
#[lang = "panic_fmt"]
pub use core::panicking::panic_fmt;
pub use core::result;
pub use core::option;
pub use core::cmp;
pub use core::convert;
pub use core::slice;
pub use core::borrow;
pub use core::mem;
pub mod prelude;
#[prelude_import]
pub use prelude::v1::*;

#[lang = "eh_personality"] pub extern fn eh_personality() {}

pub mod vec{
    use ops::Add;
    #[repr(C)]
    #[derive(Copy, Clone)]
    #[spirv(Vec2)]
    pub struct Vec2<T: Copy> {
        pub x: T,
        pub y: T,
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

    impl Vec2<f32>{
        #[spirv(dot)]
        #[inline(never)]
        pub fn dot(self, other: Vec2<f32>) -> f32{
            self.x * other.x + self.y * other.y
        }
    }
}
