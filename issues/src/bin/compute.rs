#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::{Compute, Descriptor, Fragment, Input, N0, N1, N2, Output, Vec2, Vec3, Vec4, Vertex};

#[spirv(compute)]
fn compute(
    compute: Compute,
) {
}

fn main() {}
