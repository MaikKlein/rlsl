#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;

pub struct Data {
    i1: f32,
    arr: RuntimeArray<f32>,
    i2: f32
}
#[spirv(compute)]
fn compute(compute: Compute, ssao: Buffer<N0, N0, Data>) {
    ssao.data.arr.get(1000);
}

fn main() {}
