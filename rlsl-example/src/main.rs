// vertex!(
//     input(location=0) pos_in: Vec4<f32>,
//     input(location=1) color_in: Vec4<f32>,
//     uniform(set = 0, binding = 0) foo: Foo,
//     output(location=0, flat) mut color: Vec4<f32>,
//     output(location=1) mut color: Vec4<f32>, // smooth default
//     output(location=2, noperspective) mut color2: Vec4<f32> {
//         color = color_in;
//     }
// )

// type VertexOut =(Output<Flat, 0, Vec4<f32>>,
//                  Output<1, Vec4<f32>>,
//                  Output<NoPersp, 2, Vec4<f32>>);

// fn vertex(pos_in: Input<0, Vec4<f32>>,
//           color_in: Input<1, Vec4<f32>>,
//           foo: Descriptor<0, 0, Foo>) -> VertexOut {

// }

#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::{N0, N1, Input, Output, Vec2, Vec4, Vertex};

#[spirv(fragment)]
fn frag(color: Input<N0, Vec4<f32>>) -> Output<Vec4<f32>> {
    color.into()
}

#[spirv(vertex)]
fn vertex(
    vertex: &mut Vertex,
    pos: Input<N0, Vec4<f32>>,
    color: Input<N1, Vec4<f32>>,
) -> Output<Vec4<f32>> {
    vertex.position = pos.data;
    color.into()
}

fn main() {}
