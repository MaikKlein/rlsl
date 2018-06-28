#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;

pub struct Sphere {
    pub origin: Vec3<f32>,
    pub radius: f32,
}

pub struct Ray {
    pub origin: Vec3<f32>,
    pub dir: Vec3<f32>,
}

impl Ray {
    pub fn new(origin: Vec3<f32>, dir: Vec3<f32>) -> Ray {
        Ray { origin, dir }
    }
    pub fn position(&self, t: f32) -> Vec3<f32> {
        self.origin + self.dir * t
    }
}

pub struct RayHit {
    pub dist: f32,
    pub position: Vec3<f32>,
    pub normal: Vec3<f32>,
}

impl Sphere {
    pub fn new(origin: Vec3<f32>, radius: f32) -> Sphere {
        Sphere { origin, radius }
    }
    pub fn intersect(&self, ray: Ray) -> Option<RayHit> {
        let oc = ray.origin - self.origin;
        //a is always one if the direction is a unit vector
        let a = 1.0f32;
        let b = 2.0 * ray.dir.dot(oc);
        let c = Vec3::dot(oc, oc) - self.radius * self.radius;
        let discr = b * b - 4.0 * a * c;
        let t = if discr < 0.0 {
            -1.0
        } else {
            let two_a = 2.0 * a;
            let t1 = (b* -1.0 - discr.sqrt()) / two_a;
            let t2 = (b* -1.0 + discr.sqrt()) / two_a;
            if t1 < t2 {
                t1
            }
            else {
                t2
            }
        };
        if t > 0.0001 {
            let position = ray.position(t);
            let normal = position - self.origin;
            let normal = normal.normalize();
            Some(RayHit {
                position,
                normal,
                dist: t,
            })
        } else {
            None
        }
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
    let coord = (uv * 2.0 - 1.0).extend(0.0);
    let look = Vec3::new(0.0f32, 0.0, 1.0);
    let origin = Vec3::new(0.0, 0.0, 0.0);
    let dir = origin + coord;
    let ray = Ray::new(origin, dir);
    let sphere = Sphere::new(Vec3::new(0.0, 0.0, 100.0), 0.00001);
    let hit = sphere.intersect(ray);
    let color = if let Some(hit) = hit {
        Vec4::new(1.0, 0.0, 0.0, 1.0)
    }
    else{
        Vec4::new(0.0, 0.0, 0.0, 1.0)
    };

    Output::new(color)
}

fn main() {
}
