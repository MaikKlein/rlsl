#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;
// #[inline(always)]
// fn id(i: &u32) -> &u32 {
//     i
// }
#[spirv(vertex)]
fn vertex(
    vertex: &mut Vertex,
    pos: Input<N0, Vec2<f32>>,
    uv: Input<N1, Vec2<f32>>,
) -> Output<N0, Vec2<f32>> {
    vertex.position = pos.extend2(0.0, 1.0);
    Output::new(uv.data)
}

fn main() {}
