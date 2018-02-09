#![feature(custom_attribute, attr_literals)]
extern crate rlsl_math;
use rlsl_math::{Vec4, Vertex};
#[spirv(fragment)]
fn frag(color: Vec4<f32>) -> Vec4<f32> {
    color
}

#[spirv(vertex)]
fn vertex(vertex: &mut Vertex, pos: Vec4<f32>, color: Vec4<f32>) -> Vec4<f32> {
    vertex.position = pos;
    color
}

fn main(){}
