#![feature(custom_attribute)]
extern crate rlsl_math;
use rlsl_math::prelude::*;

pub struct Sphere {
    pub origin: Vec3<f32>,
    pub radius: f32,
}

// #[allow(unused_qualifications)]
// impl ::std::clone::Clone for Ray {
//     #[inline]
//     fn clone(&self) -> Ray {
//         match *self {
//             Ray { origin: ref __self_0_0, dir: ref __self_0_1 } =>
//             Ray{origin: ::std::clone::Clone::clone(&(*__self_0_0)),
//                 dir: ::std::clone::Clone::clone(&(*__self_0_1)),},
//         }
//     }
// }

//#[derive(Clone)]
pub struct Ray {
    pub origin: Vec3<f32>,
    pub dir: Unit<Vec3<f32>>,
}

impl Ray {
    pub fn new(origin: Vec3<f32>, dir: Unit<Vec3<f32>>) -> Ray {
        Ray { origin, dir }
    }
    pub fn position(&self, t: f32) -> Vec3<f32> {
        self.origin + *self.dir * t
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
    pub fn intersect(self, ray: Ray) -> Option<RayHit> {
        use rlsl_math::polynomial::quadratic;
        let oc = ray.origin - self.origin;
        //a is always one if the direction is a unit vector
        let a = 1.0f32;
        let b = 2.0 * ray.dir.dot(oc);
        let c = Vec3::dot(oc, oc) - self.radius * self.radius;
        let t = quadratic(a, b, c)?.min();
        if t < 0.0 {
            return None;
        }
        let position = ray.position(t);
        let normal = position - self.origin;
        let normal = normal.normalize();
        let hit = RayHit {
            position,
            normal,
            dist: t,
        };
        Some(hit)
    }
}

#[spirv(fragment)]
fn fragment(
    frag: Fragment,
    uv: Input<N0, Vec2<f32>>,
    time: Uniform<N0, N0, f32>,
) -> Output<N0, Vec4<f32>> {
    let uv = *uv;
    let time = *time;
    let coord = (uv * 2.0 - 1.0).extend(0.0);
    // The vector where the camera points to
    let look = Vec3::new(0.0f32, 0.0, 1.0);
    // The location of the camera
    let origin = Vec3::new(0.0, 0.0, 0.0);
    // The direction of the ray that we will shoot
    let dir = look + coord;
    let ray = Ray::new(origin, dir.to_unit());
    let position = Vec2::from_polar(1.0 * time, 2.0).extend(10.0);
    let sphere = Sphere::new(position, 1.0);
    let hit = sphere.intersect(ray);
    let light_pos = Vec3::new(2.0, -4.0, 0.0);
    let color = if let Some(hit) = hit {
        // Calculate the vector from the hit location to the light source
        let light_vec = Vec3::normalize(light_pos - hit.position);
        // The simple phong cosine term
        let cos = light_vec.dot(hit.normal);
        // Multiply the color with the cos term
        let color_sphere = Vec3::new(0.9, 0.1, 0.1) * cos;
        color_sphere.extend(1.0)
    } else {
        // Pseudo sky color. We lerp between white an blue. The uv.y value is
        // inbetween 0 and 1 and starts from the top.
        let t = uv.y;
        let white = Vec3::single(1.0);
        let blue = Vec3::new(0.2, 0.5, 1.0);
        blue.lerp(white, t).extend(1.0)
    };
    Output::new(color)
}

fn main() {}
