#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;

#[spirv(fragment)]
fn fragment(frag: Fragment, uv: Input<N0, Vec2<f32>>, time: Uniform<N2, N0, f32>) -> Output<N0, Vec4<f32>> {
    let uv = *uv;
    let time = *time;
    let offset = Vec3::new(0.0, 2.0, 4.0);
    let f = uv.dot(uv);
    let coord = uv
        .extend(uv.y)
        .add(offset)
        .map(move |f| f32::cos(time + f) * 0.5)
        .add(Vec3::single(0.5))
        .extend(1.0);
    Output::new(coord)
}

fn main() {}
