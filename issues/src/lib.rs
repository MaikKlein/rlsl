extern crate rlsl_math;
use rlsl_math::prelude::*;
pub fn square(_: u32, val: f32) -> f32 {
    val * val
}

pub fn single_branch(_: u32, val: f32) -> f32 {
    if val > 1.0 {
        1.0
    }
    else{
        0.0
    }
}

pub fn simple_loop(_: u32, val: f32) -> f32 {
    val
}

pub fn u32_add(_: u32, val: f32) -> f32 {
    let i = 1u32;
    let i2 = i + i;
    val
}
