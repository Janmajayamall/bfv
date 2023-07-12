use crate::proto;
use itertools::Itertools;
use ndarray::Array2;
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

        let bit_rev = (0..degree)
            .into_iter()
            .map(|i| i.reverse_bits() >> (degree.leading_zeros() + 1))
            .collect_vec()
            .into_boxed_slice();

        Substitution {
            exponent,
            power_bitrev,
            degree,
            bit_rev,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    pub(crate) coefficients: Array2<u64>,
    pub(crate) representation: Representation,
}

impl Poly {
    pub fn new(coefficients: Array2<u64>, representation: Representation) -> Poly {
        Poly {
            coefficients,
            representation,
        }
    }

    pub fn placeholder() -> Poly {
        Poly {
            coefficients: Array2::default((0, 0)),
            representation: Representation::Unknown,
        }
    }
}

impl Poly {
    // fn to_proto(&self, poly_ctx: crate::PolyContext<'_>) -> proto::Poly {}
}
