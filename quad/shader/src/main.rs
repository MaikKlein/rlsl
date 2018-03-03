#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::{Descriptor, Fragment, Input, N0, N1, N2, Output, Vec2, Vec3, Vec4, Vertex};

#[spirv(fragment)]
fn color_frag(
    frag: Fragment,
    uv: Input<N0, Vec2<f32>>,
    time: Descriptor<N2, N0, f32>,
) -> Output<N0, Vec4<f32>> {
    let uv = uv.data;
    // let mut time = time.data;
    let offset = Vec3::new(0.0f32, 2.0, 4.0);
    let right = uv.x > 0.5;
    let top = uv.y > 0.5;

    let color: Vec3<f32> = match (right, top) {
        (true, true) => Vec3::new(0.0, 0.0, 1.0),
        (true, false) => Vec3::new(0.0, 1.0, 0.0),
        (false, true) => Vec3::new(1.0, 0.0, 0.0),
        (false, false) => Vec3::new(1.0, 1.0, 1.0),
    };
    // let color: Vec3<f32> = match right {
    //     true => Vec3::new(0.0, 0.0, 1.0),
    //     false => Vec3::new(0.0, 1.0, 0.0),
    // };

    // let coord = uv.extend(uv.x)
    //     .add(offset)
    //     .map(move |f| f32::cos(time + f) * 0.5)
    //     .add(Vec3::single(0.5))
    //     .extend(1.0);
    Output::new(color.extend(1.0))
}

// #[spirv(vertex)]
// fn vertex(
//     vertex: &mut Vertex,
//     pos: Input<N0, Vec4<f32>>,
//     uv: Input<N1, Vec2<f32>>,
// ) -> Output<N0, Vec2<f32>> {
//     vertex.position = pos.data;
//     Output::new(uv.data)
// }

//#[spirv(fragment)]
//fn test_frag(
//    frag: Fragment,
//) -> Output<N0, Vec4<f32>> {
//    // let mut f = 1.0f32;
//    // test_mut(&mut f);
//    //let coord = Vec3::new(1.0, 2.0, 3.0);
//    let coord = Vec3::new(0.0, 0.0, 0.0).map(move |f| f + 1.0).extend(1.0);
//    // let coord = Vec3::new(1.0, 2.0, 3.0);
//    // let coord1 = Vec3::new(coord.x, 2.0, 3.0);
//    Output::new(coord)
//}

// fn test(f: f32) -> f32 {
//     f + 1.0
// }

// fn test_mut(f: &mut f32) {
//     *f += 1.0;
// }
//fn test_add(f: &mut f32){
//    *f += 0.5f32;
//}
// #[spirv(fragment)]
// fn red_frag(color: Input<N0, Vec2<f32>>) -> Output<N0, Vec4<f32>> {
//     let color = Vec4::new(1.0, 0.0, 0.0, 1.0);
//     Output::new(color)
// }

// #[spirv(fragment)]
// fn blue_frag(color: Input<N0, Vec2<f32>>) -> Output<N0, Vec4<f32>> {
//     let color = Vec4::new(0.0, 0.0, 1.0, 1.0);
//     Output::new(color)
// }

// fn main() {

//     let v = Vec4::new(1.0f32, 0.0, 0.0, 1.0);
//     let v1 = Vec4::new(0.0f32, 0.0, 1.0, 1.0);
//     println!("{:?}", v.interpolate(v1, 0.5));
// }
//

// fn main() {
//     let uv = Vec2 { x: 0.5, y: 0.5 };
//     let time = 1.0f32;
//     let offset = Vec3::new(0.0, 2.0, 4.0);
//     let coord = uv.extend(uv.y)
//         .add(offset)
//         .map(move |f| f32::cos(time + f) * 0.5)
//         .add(Vec3::single(0.5))
//         .extend(1.0);
//     println!("coord {:?}", coord);
// }
fn main() {}
