#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::{Descriptor, Fragment, Input, N0, N1, N2, Output, Vec2, Vec3, Vec4, Vertex};

#[spirv(vertex)]
fn vertex(
    vertex: &mut Vertex,
    pos: Input<N0, Vec2<f32>>,
    uv: Input<N1, Vec2<f32>>,
) -> Output<N0, Vec2<f32>> {
    vertex.position = pos.data.extend2(0.0, 1.0);
    Output::new(uv.data)
}

fn main() {}
