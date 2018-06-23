use std::ops::{Add, Mul};
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

pub trait Array<T: Copy> {
    fn get(&self, index: u32) -> T;
    //fn len(&self) -> u32;
}
