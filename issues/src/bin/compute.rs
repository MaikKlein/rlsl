#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;

#[spirv(compute)]
fn compute(
    compute: Compute,
    ssao: Buffer<N0, N0, u32>
) {
}

fn main() {}
