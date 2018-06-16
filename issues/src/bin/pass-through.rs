#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;
#[spirv(fragment)]
fn fragment(frag: Fragment, uv: Input<N0, Vec2<f32>>) -> Output<N0, Vec4<f32>> {
    let uv = uv.data;
    Output::new(uv.extend2(0.0, 1.0))
}

fn main() {}
