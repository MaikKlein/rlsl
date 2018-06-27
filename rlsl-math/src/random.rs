pub struct Rng {
    seed: f32,
}
impl Rng {
    pub fn random(&mut self) -> f32 {
        self.seed += 1.0;
        f32::fract(self.seed.sin() * 100000.0)
    }

    pub fn from_seed(seed: f32) -> Rng {
        Rng {
            seed
        }
    }
}
