#![feature(custom_attribute, attr_literals)]
#![feature(core_intrinsics)]
pub mod vector;
pub mod intrinsics;
pub mod num;
pub mod constants;
pub mod entry;
pub mod random;

pub mod prelude {
    pub use vector::*;
    pub use num::*;
    pub use constants::*;
    pub use entry::*;
    pub use random::*;
}
