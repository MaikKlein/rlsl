pub trait Constant: Sized {}

#[spirv(Const0)]
pub enum N0 {}

#[spirv(Const1)]
pub enum N1 {}

#[spirv(Const2)]
pub enum N2 {}

#[spirv(Const2)]
pub enum N3 {}

impl Constant for N0 {}
impl Constant for N1 {}
impl Constant for N2 {}
impl Constant for N3 {}
