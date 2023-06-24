use crate::modulus::Modulus;
use crypto_bigint::U192;
use fhe_util::sample_vec_cbd;
use itertools::{izip, Itertools};
use ndarray::{azip, s, Array2, ArrayView2, Axis, IntoNdProducer};
use num_bigint::BigUint;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{identities::One, ToPrimitive, Zero};
use rand::{CryptoRng, RngCore};
use seq_macro::seq;
use std::{
    mem::{self, MaybeUninit},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    sync::Arc,
};

pub mod poly_context;

pub use poly_context::PolyContext;

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum Representation {
    Evaluation,
    Coefficient,
    Unknown,
}

#[derive(Debug)]
pub struct Substitution {
    exponent: usize,
    power_bitrev: Box<[usize]>,
    bit_rev: Box<[usize]>,
    degree: usize,
}

impl Substitution {
    /// Computes substitution map for polynomial degree
    ///
    /// exponent must be an odd integer not a multiple of 2 * degree.
    pub fn new(exponent: usize, degree: usize) -> Substitution {
        assert!(exponent & 1 == 1);
        let exponent = exponent % (2 * degree);
        let mask = degree - 1;
        let mut power = (exponent - 1) / 2;
        let power_bitrev = (0..degree)
            .map(|_| {
                let r = (power & mask).reverse_bits() >> (degree.leading_zeros() + 1);
                power += exponent;
                r
            })
            .collect_vec()
            .into_boxed_slice();

        Substitution {
            exponent,
            power_bitrev,
            degree,
            bit_rev: todo!(),
        }
    }
}

/// Should only be concerned with polynomial operations.
/// This mean don't store any BFV related pre-computation here
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    pub coefficients: Array2<u64>,
    pub representation: Representation,
}

impl Poly {
    pub fn placeholder() -> Poly {
        Poly {
            coefficients: Array2::zeros((0, 0)),
            representation: Representation::Unknown,
        }
    }
}
