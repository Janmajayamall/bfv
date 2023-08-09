use crate::{BfvParameters, Poly, PolyType};
use itertools::Itertools;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, PartialEq)]
pub struct Ciphertext {
    pub(crate) c: Vec<Poly>,
    pub(crate) poly_type: PolyType,
    pub(crate) seed: Option<<ChaCha8Rng as SeedableRng>::Seed>,
    pub(crate) level: usize,
}

impl Ciphertext {
    pub fn new(c: Vec<Poly>, poly_type: PolyType, level: usize) -> Ciphertext {
        Ciphertext {
            c,
            poly_type,
            level,
            seed: None,
        }
    }

    pub fn placeholder() -> Ciphertext {
        Ciphertext {
            c: vec![],
            poly_type: PolyType::Q,
            level: 0,
            seed: None,
        }
    }

    pub fn c_ref_mut(&mut self) -> &mut [Poly] {
        &mut self.c
    }

    pub fn c_ref(&self) -> &[Poly] {
        &self.c
    }

    pub fn poly_type(&self) -> PolyType {
        self.poly_type.clone()
    }

    pub fn level(&self) -> usize {
        self.level
    }
}

mod tests {
    use super::*;
    use crate::{Encoding, Evaluator, SecretKey};
}
