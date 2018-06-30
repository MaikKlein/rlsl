use constants::*;
use intrinsics::spirv_discard;
use std::marker::PhantomData;
use std::ops::Deref;
use vector::*;
#[spirv(PerFragment)]
pub struct Fragment {
    pub frag_coord: Vec4<f32>,
}

impl Fragment {
    #[spirv(discard)]
    pub fn discard() {
        unsafe {
            spirv_discard();
        }
    }
}

#[spirv(PerVertex)]
pub struct Vertex {
    pub position: Vec4<f32>,
    pub point_size: f32,
}

#[spirv(Compute)]
pub struct Compute {
    pub local_invocation_index: u32,
    pub global_invocation_index: Vec3<u32>,
}

#[spirv(Input)]
pub struct Input<Location: Sized, T> {
    pub data: T,
    pub _location: PhantomData<Location>,
}

// impl<LInput, LOutput, T> From<Input<LInput, T>> for Output<LOutput, T> {
//     fn from(input: Input<LInput, T>) -> Output<LOutput, T> {
//         Output::new(data: input.data)
//     }
// }
impl<Location, T> Deref for Input<Location, T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.data
    }
}

#[spirv(Output)]
pub struct Output<Location: Sized, T> {
    pub data: T,
    pub _location: PhantomData<Location>,
}

impl<Location, T> Deref for Output<Location, T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.data
    }
}

impl<Location, T> Output<Location, T> {
    pub fn new(data: T) -> Output<Location, T> {
        Output {
            _location: PhantomData,
            data,
        }
    }
}
#[spirv(RuntimeArray)]
pub struct RuntimeArray<T> {
    pub _m: PhantomData<T>,
}

#[inline(never)]
#[spirv(runtime_array_get)]
fn runtime_array_get<T, R>(data: T, index: u32) -> R {
    unsafe { ::std::intrinsics::abort() }
}
#[inline(never)]
#[spirv(runtime_array_store)]
fn runtime_array_store<T, T1>(data: T, index: u32, value: T1) {
    unsafe {
        ::std::intrinsics::abort();
    }
}

impl<T> RuntimeArray<T> {
    pub fn get(&self, index: u32) -> T {
        runtime_array_get(self, index)
    }
    pub fn store(&self, index: u32, value: T) {
        runtime_array_store(self, index, value);
    }
}

#[spirv(Uniform)]
pub struct Uniform<Binding, Set, T>
where
    Binding: Constant,
    Set: Constant,
{
    pub data: T,
    pub _location: PhantomData<Binding>,
    pub _binding: PhantomData<Set>,
}

impl<Binding, Set, T> Deref for Uniform<Binding, Set, T>
where
    Binding: Constant,
    Set: Constant,
{
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.data
    }
}

#[spirv(Buffer)]
pub struct Buffer<Binding, Set, T>
where
    Binding: Constant,
    Set: Constant,
{
    pub data: T,
    pub _location: PhantomData<Binding>,
    pub _binding: PhantomData<Set>,
}
