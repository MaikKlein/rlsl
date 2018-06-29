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
    let frag_coord = Vec2::new(frag.frag_coord.x, frag.frag_coord.y);
    let r = Vec2::new(1000.0, 1000.0);
    let color = Vec3::single(1.0).map(move |f|{
        let mut p = frag_coord / r - 0.5;
        p.x *= r.x / r.y;
        1.0
    });
    Output::new(color.extend(1.0))
}

fn main() {}
