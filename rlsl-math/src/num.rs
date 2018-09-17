use std::ops::{Add, Div, Mul, Sub};
pub trait Float:
    PartialOrd + Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn one() -> Self;
    fn zero() -> Self;
}

impl Float for f32 {
    fn sqrt(self) -> f32 {
        self.sqrt()
    }

    fn sin(self) -> f32 {
        self.sin()
    }

    fn cos(self) -> f32 {
        self.cos()
    }

    fn one() -> f32 {
        1.0
    }

    fn zero() -> f32 {
        0.0
    }
}

pub trait Array<T: Copy> {
    fn get(&self, index: u32) -> T;
    //fn len(&self) -> u32;
}
