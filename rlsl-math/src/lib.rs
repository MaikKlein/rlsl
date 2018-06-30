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
    pub use vector::*;
}
