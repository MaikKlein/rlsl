#![feature(custom_attribute, attr_literals)]
#![feature(core_intrinsics)]
#[macro_use]
pub mod macros;
#[macro_use]
pub mod vector;
pub mod constants;
pub mod entry;
pub mod intrinsics;
pub mod num;
pub mod random;
pub mod range;

pub mod unit {
    use std::ops::Deref;
    use vector::Vector;
    #[derive(Copy, Clone)]
    pub struct Unit<T> {
        inner: T,
    }

    impl<T> Unit<T>
    where
        T: Vector,
    {
        pub fn new(inner: T) -> Self {
            Unit {
                inner: inner.normalize(),
            }
        }
    }
    impl<T> Deref for Unit<T> {
        type Target = T;
        #[inline(always)]
        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }
}
pub mod polynomial {
    use vector::Vec2;
    pub fn quadratic(a: f32, b: f32, c: f32) -> Option<Vec2<f32>> {
        let discr = b * b - 4.0 * a * c;
        if discr < 0.0 {
            return None;
        }
        let two_a = 2.0 * a;
        let x1 = (-1.0 * b + discr.sqrt()) / two_a;
        let x2 = (-1.0 * b - discr.sqrt()) / two_a;
        Some(Vec2::new(x1, x2))
    }
}
pub mod prelude {
    pub use constants::*;
    pub use entry::*;
    pub use num::*;
    pub use random::*;
    pub use range::*;
    pub use unit::*;
    pub use vector::*;
}

pub trait Array<T> {
    fn get(&self, index: usize) -> T;
    fn length(&self) -> usize;
}

impl<T: Copy> Array<T> for [T; 2] {
    fn length(&self) -> usize {
        2
    }
    fn get(&self, index: usize) -> T {
        self[index]
    }
}

impl<T: Copy> Array<T> for [T; 3] {
    fn length(&self) -> usize {
        3
    }
    fn get(&self, index: usize) -> T {
        self[index]
    }
}
