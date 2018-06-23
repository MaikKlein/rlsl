pub fn square(_: u32, val: f32) -> f32 {
    val * val
}

pub fn single_branch(_: u32, val: f32) -> f32 {
    if val > 1.0 {
        1.0
    }
    else{
        0.0
    }
}
