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
use rlsl_math::{Descriptor, Input, N0, N1, N2, Output, Vec2, Vec4, Vec3, Vertex};
#[spirv(fragment)]
// fn frag(
//     color: Input<N0, Vec4<f32>>,
//     uniform_color: Descriptor<N0, N0, Vec4<f32>>,
// ) -> Output<N0, Vec4<f32>> {
//     let color = color.data;
//     Output::new(uniform_color.data)
// }
// pub struct Test {
//     v1: Vec2<f32>,
//     v: Vec3<f32>,
//     f: f32
// }
// pub struct Test1 {
//     v: Vec3<f32>,
//     v1: Vec2<f32>,
//     f: f32
// }
// pub struct Test2 {
//     v1: Vec4<f32>,
//     f: f32
// }
// pub struct Test3 {
//     t: Test,
//     t1: Test1,
//     t2: Test2,
// }

#[spirv(fragment)]
fn color_frag(
    color: Input<N0, Vec2<f32>>,
    uniform_color: Descriptor<N0, N0, Vec4<f32>>,
) -> Output<N0, Vec4<f32>> {
    let color = color.data.extend2(0.0, 1.0);
    Output::new(uniform_color.data)
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
// #[spirv(vertex)]
// fn vertex(
//     vertex: &mut Vertex,
//     pos: Input<N0, Vec4<f32>>,
//     color: Input<N1, Vec4<f32>>,
// ) -> Output<N0, Vec4<f32>> {
//     vertex.position = pos.data;
//     Output::new(color.data)
// }

//fn main(){}
