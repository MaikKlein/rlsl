#![feature(custom_attribute, attr_literals)]
use std::ops::{Add, Mul};
use std::convert::From;
pub trait Float: Copy + Add<Output = Self> + Mul<Output = Self> {
    fn sqrt(self) -> Self;
    fn one() -> Self;
}

impl Float for f32 {
    fn sqrt(self) -> f32 {
        self.sqrt()
    }
    fn one() -> f32 {
        1.0
    }
}

#[spirv(PerFragment)]
pub struct Fragment {
    pub frag_coord: Vec4<f32>,
}

#[spirv(PerVertex)]
pub struct Vertex {
    pub position: Vec4<f32>,
    pub point_size: f32,
}

#[spirv(Vec4)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Vec4<T> {
    #[inline]
    pub fn new(x: T, y: T, z: T, w: T) -> Vec4<T> {
        Vec4 { x, y, z, w }
    }
    // pub fn shrink(self) -> Vec3<T> {
    //     Vec3 {
    //         x: self.x,
    //         y: self.y,
    //         z: self.z,
    //     }
    // }
}

impl Vec4<f32> {
    #[inline]
    pub fn interpolate(self, other: Self, t: f32) -> Self {
        let i_t = 1.0 - t;
        let x = i_t * self.x + t * other.x;
        let y = i_t * self.y + t * other.y;
        let z = i_t * self.z + t * other.z;
        let w = i_t * self.w + t * other.w;
        Vec4 { x, y, z, w }
    }
}

#[spirv(Vec3)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3<T: Copy> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Float> Vec3<T> {
    #[inline]
    pub fn add(self, other: Self) -> Self {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}
impl<T: Copy> Vec3<T> {
    pub fn single(t: T) -> Vec3<T> {
        Vec3 { x: t, y: t, z: t }
    }
    pub fn new(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { x, y, z }
    }

    #[inline]
    pub fn map<R, F>(self, mut f: F) -> Vec3<R>
    where
        F: FnMut(T) -> R,
        R: Copy,
    {
        Vec3 {
            x: f(self.x),
            y: f(self.y),
            z: f(self.z),
        }
    }
    #[inline]
    pub fn map2<R, F>(self, mut f: F) -> Vec3<R>
    where
        F: FnMut(T, T) -> R,
        R: Copy,
    {
        Vec3 {
            x: f(self.x, self.x),
            y: f(self.y, self.x),
            z: f(self.z, self.x),
        }
    }

    pub fn extend(self, t: T) -> Vec4<T> {
        Vec4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w: t,
        }
    }
}

#[spirv(Vec2)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}
impl<T: Float> Vec2<T> {
    pub fn dot(self, other: Self) -> T {
        <Self as Vector>::dot(self, other)
    }

    pub fn length(self) -> T {
        <Self as Vector>::length(self)
    }

    pub fn extend(self, z: T) -> Vec3<T> {
        Vec3 {
            x: self.x,
            y: self.y,
            z,
        }
    }
    pub fn extend2(self, z: T, w: T) -> Vec4<T> {
        Vec4 {
            x: self.x,
            y: self.y,
            z,
            w,
        }
    }
}

impl<T: Float> Vector for Vec2<T> {
    type T = T;
    const DIM: usize = 2;
    fn dot(self, other: Self) -> T {
        self.x * other.y + self.y + other.y
    }
}

pub trait Vector
where
    Self: Copy + Sized,
{
    type T: Float;
    const DIM: usize;
    fn dot(self, Self) -> Self::T;
    fn length(self) -> Self::T {
        self.dot(self).sqrt()
    }
}

#[spirv(Input)]
pub struct Input<Location: Sized, T> {
    pub data: T,
    pub _location: std::marker::PhantomData<Location>,
}

// impl<LInput, LOutput, T> From<Input<LInput, T>> for Output<LOutput, T> {
//     fn from(input: Input<LInput, T>) -> Output<LOutput, T> {
//         Output::new(data: input.data)
//     }
// }

#[spirv(Output)]
pub struct Output<Location: Sized, T> {
    pub data: T,
    pub _location: std::marker::PhantomData<Location>,
}

impl<Location, T> Output<Location, T> {
    pub fn new(data: T) -> Output<Location, T> {
        Output {
            _location: std::marker::PhantomData,
            data,
        }
    }
}
#[spirv(Descriptor)]
pub struct Descriptor<Binding: Sized, Set: Sized, T> {
    pub data: T,
    pub _location: std::marker::PhantomData<Binding>,
    pub _binding: std::marker::PhantomData<Set>,
}

#[spirv(Const0)]
pub enum N0 {}

#[spirv(Const1)]
pub enum N1 {}

#[spirv(Const2)]
pub enum N2 {}
