use num::Float;

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
    pub fn shrink(self) -> Vec3<T> {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z,
        }
    }
}

#[spirv(Vec3)]
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy> Vec3<T> {
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

    pub fn shrink(self, t: T) -> Vec2<T> {
        Vec2 {
            x: self.x,
            y: self.y,
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
    #[inline]
    pub fn from_polar(angle: T, dist: T) -> Self {
        let x = dist * angle.cos();
        let y = dist * angle.sin();
        Vec2::new(x, y)
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

pub trait Vector
where
    Self: Copy + Sized,
{
    type T: Float;
    fn dot(self, Self) -> Self::T;

    fn length(self) -> Self::T {
        self.dot(self).sqrt()
    }
}

impl<T: Float> Vector for Vec2<T> {
    type T = T;
    fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y
    }
}
impl<T: Float> Vector for Vec3<T> {
    type T = T;
    fn dot(self, other: Self) -> T {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z
    }
}
impl<T: Float> Vector for Vec4<T> {
    type T = T;
    fn dot(self, other: Self) -> T {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z +
        self.w * other.w
    }
}

vec_ops_vec!(Vec2 { x, y });
vec_ops_vec!(Vec3 { x, y, z });
vec_ops_vec!(Vec4 { x, y, z, w });

vec_ops_scalar!(Vec2 { x, y });
vec_ops_scalar!(Vec3 { x, y, z });
vec_ops_scalar!(Vec4 { x, y, z, w });

vec_common!(Vec2 { x, y });
vec_common!(Vec3 { x, y, z });
vec_common!(Vec4 { x, y, z, w });
