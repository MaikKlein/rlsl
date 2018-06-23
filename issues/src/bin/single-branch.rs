#![feature(custom_attribute)]
extern crate rlsl_math;
extern crate issues;
use rlsl_math::prelude::*;
use issues::single_branch;

#[spirv(compute)]
fn compute(compute: Compute, buffer: Buffer<N0, N0, RuntimeArray<f32>>) {
    let index = compute.global_invocation_index.x;
    let value = buffer.data.get(index);

    let result = single_branch(index, value);
    buffer.data.store(index, result);
}

fn main() {}
