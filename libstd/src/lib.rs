#![no_std]
#![feature(macro_reexport)]
#![feature(lang_items)]
#![feature(core_intrinsics)]
#![feature(unwind_attributes)]
#![feature(core_panic)]
#![feature(prelude_import)]

#[macro_reexport(assert, assert_eq, assert_ne, debug_assert, debug_assert_eq,
debug_assert_ne, unreachable, unimplemented, write, writeln, try)]
extern crate core as __core;

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


