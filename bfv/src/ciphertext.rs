use crate::parameters::{BfvParameters, PolyType};
use crate::poly::{Poly, Representation};
use crate::warn;
use std::sync::Arc;
use traits::Ntt;

#[derive(Debug, Clone)]
pub struct Ciphertext {
    pub(crate) c: Vec<Poly>,
    pub poly_type: PolyType,
    pub level: usize,
}
