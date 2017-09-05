#![no_std]
#![feature(macro_reexport)]
#![feature(lang_items)]
#![feature(core_intrinsics)]
#![feature(unwind_attributes)]
#![feature(core_panic)]

#[macro_reexport(assert, assert_eq, assert_ne, debug_assert, debug_assert_eq,
debug_assert_ne, unreachable, unimplemented, write, writeln, try)]
extern crate core as __core;

pub use core::clone;
pub use core::marker;
pub use core::ops;
pub use core::fmt;
pub use core::panicking;
pub use core::panicking::panic_fmt;

#[lang = "eh_personality"] pub extern fn eh_personality() {}

