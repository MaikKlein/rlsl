#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;
#[spirv(fragment)]
fn fragment(frag: Fragment, uv: Input<N0, Vec2<f32>>) -> Output<N0, Vec4<f32>> {
    let uv = uv.data;
    let mut color = Vec3::new(1.0, 0.0, 0.0);
    if uv.x > 0.5 {
        color.x = 0.5;
    }
    Output::new(color.extend(1.0))
}

fn main() {}
