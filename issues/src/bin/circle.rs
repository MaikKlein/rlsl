#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;

pub struct Circle {
    pub origin: Vec2<f32>,
    pub radius: f32,
}

impl Circle {
    pub fn new(origin: Vec2<f32>, radius: f32) -> Circle {
        Circle { origin, radius }
    }

    pub fn is_inside(&self, point: Vec2<f32>) -> bool {
        let p = point - self.origin;
        self.radius > p.length()
    }
}
#[spirv(fragment)]
fn fragment(
    frag: Fragment,
    uv: Input<N0, Vec2<f32>>,
    time: Uniform<N2, N0, f32>,
) -> Output<N0, Vec4<f32>> {
    let uv = *uv;
    let time = *time;
    let c = time.cos().abs();
    let s = time.sin().abs();
    let origin = Vec2::new(0.5, 0.5) + Vec2::from_polar(time, 0.2 * c);
    let circle = Circle::new(origin, c * 0.1 + 0.05);
    let inner_color = Vec4::new(uv.x * c, uv.y * s, s , 1.0);
    let inner_color_inv = inner_color.map(move |f| 1.0 - f);
    let color = if circle.is_inside(uv) {
        inner_color
    } else {
        inner_color_inv
    };
    Output::new(color)
}

fn main() {}
