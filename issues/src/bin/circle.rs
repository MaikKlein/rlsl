#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;

//#[derive(Copy, Clone)]
pub struct Circle {
    pub origin: Vec2<f32>,
    pub radius: f32,
}

impl Circle {
    pub fn new(origin: Vec2<f32>, radius: f32) -> Circle {
        Circle { origin, radius }
    }

    pub fn is_inside(&self, point: Vec2<f32>) -> bool {
        let p = Vec2 {
            x: point.x - self.origin.x,
            y: point.y - self.origin.y,
        };
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
    let c = time.cos().abs();
    let s = time.sin().abs();
    let circle = Circle::new(Vec2::new(0.5, 0.5), c);

    let color = if circle.is_inside(uv) {
        Vec4::new(c, s, c, 1.0)
    } else {
        Vec4::new(0.0, 0.0, 0.0, 1.0)
    };
    Output::new(color)
}

fn main() {}
