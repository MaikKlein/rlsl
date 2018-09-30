#![feature(custom_attribute)]
extern crate rlsl_math;
extern crate issues;
use rlsl_math::prelude::*;
use issues::ray::render;

#[spirv(fragment)]
fn fragment(
    frag: Fragment,
    uv: Input<N0, Vec2<f32>>,
    time: Uniform<N0, N0, f32>,
) -> Output<N0, Vec4<f32>> {
    Output::new(render(*uv, *time))
}

fn main() {}
