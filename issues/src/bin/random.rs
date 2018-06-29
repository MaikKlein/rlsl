#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;

#[spirv(fragment)]
fn fragment(
    frag: Fragment,
    uv: Input<N0, Vec2<f32>>,
    time: Uniform<N0, N0, f32>,
) -> Output<N0, Vec4<f32>> {
    let time = *time;
    let scaled_uv = *uv * 40.0 * time.cos().abs();
    let new_uv = Vec2::new(scaled_uv.x.floor(), scaled_uv.y.floor());
    let mut rng = Rng::from_seed(new_uv.x + new_uv.y);
    let color = Vec4::single(rng.random());
    Output::new(color)
}

fn main() {}
