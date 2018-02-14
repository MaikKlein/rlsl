#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::{Input, N0, N1, Output, Vec2, Vec4, Vertex};

#[spirv(fragment)]
fn color_frag(color: Input<N0, Vec2<f32>>) -> Output<N0, Vec4<f32>> {
    let color = color.data.extend2(0.0, 1.0);
    Output::new(color)
}

#[spirv(fragment)]
fn red_frag(color: Input<N0, Vec2<f32>>) -> Output<N0, Vec4<f32>> {
    let color = Vec4::new(1.0, 0.0, 0.0, 1.0);
    Output::new(color)
}

#[spirv(fragment)]
fn blue_frag(color: Input<N0, Vec2<f32>>) -> Output<N0, Vec4<f32>> {
    let color = Vec4::new(0.0, 0.0, 1.0, 1.0);
    Output::new(color)
}

#[spirv(vertex)]
fn vertex(
    vertex: &mut Vertex,
    pos: Input<N0, Vec2<f32>>,
    color: Input<N1, Vec2<f32>>,
) -> Output<N0, Vec2<f32>> {
    vertex.position = pos.data.extend2(0.0, 1.0);
    Output::new(color.data)
}

fn main() {}
