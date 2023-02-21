use fhe_math::zq::{ntt::NttOperator, Modulus};
use itertools::{izip, Itertools};
use ndarray::{Array2, Axis};
use num_bigint::{BigInt, BigUint};
use num_bigint_dig::BigUint as BigUintDig;
use num_traits::{identities::One, ToPrimitive};
use rand::{CryptoRng, RngCore};
use std::{
    ops::{Add, AddAssign, Sub, SubAssign},
    sync::Arc,
};

#[derive(Clone, PartialEq, Debug)]
pub enum Representation {
    Evaluation,
    Coefficient,
    Unknown,
}

/// 1. moduli: all the modulus in poly
/// 2. modulus structs for each moduli to perform operations
/// 3. ntt structs for each moduli to perform ntt operations
/// 4.
#[derive(Clone, Debug)]
pub struct PolyContext {
    pub moduli: Box<[u64]>,
    moduli_ops: Box<[Modulus]>,
    ntt_ops: Box<[NttOperator]>,
    degree: usize,
}

impl PolyContext {
    // Creates a new polynomial context
    pub fn new(moduli: &[u64], degree: usize) -> PolyContext {
        let moduli_ops = moduli
            .iter()
            .map(|modulus| Modulus::new(*modulus).unwrap())
            .collect_vec();
        //TODO: change this to use moduli_ops instead of moduli
        let ntt_ops = moduli
            .iter()
            .enumerate()
            .map(|(index, modulus)| NttOperator::new(&moduli_ops[index], degree).unwrap())
            .collect_vec();
        PolyContext {
            moduli: moduli.to_vec().into_boxed_slice(),
            moduli_ops: moduli_ops.into_boxed_slice(),
            ntt_ops: ntt_ops.into_boxed_slice(),
            degree,
        }
    }

    pub fn modulus(&self) -> BigUint {
        let mut product = BigUint::one();
        self.moduli.iter().for_each(|m| {
            product *= *m;
        });
        product
    }

    pub fn modulus_dig(&self) -> BigUintDig {
        let mut product = BigUintDig::one();
        self.moduli.iter().for_each(|m| {
            product *= *m;
        });
        product
    }
}

/// Should only be concerned with polynomial operations.
/// This mean don't store any BFV related pre-computation here
#[derive(Clone, Debug)]
pub struct Poly {
    coefficients: Array2<u64>,
    representation: Representation,
    context: Arc<PolyContext>,
}

impl Poly {
    /// Creates zero polynomial with a given context and representation
    fn zero(poly_context: &Arc<PolyContext>, representation: &Representation) -> Poly {
        Poly {
            coefficients: Array2::zeros((poly_context.moduli.len(), poly_context.degree)),
            representation: representation.clone(),
            context: poly_context.clone(),
        }
    }

    /// Creates a polynomial with random values for given context and representation
    fn random<R: RngCore + CryptoRng>(
        poly_context: &Arc<PolyContext>,
        representation: &Representation,
        rng: &mut R,
    ) -> Poly {
        let mut poly = Poly::zero(poly_context, representation);
        izip!(
            poly.coefficients.outer_iter_mut(),
            poly_context.moduli_ops.iter()
        )
        .for_each(|(mut coefficients, q)| {
            coefficients
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(q.random_vec(poly_context.degree, rng).as_slice());
        });
        poly
    }

    /// Creates a polynomial with random values sampled from gaussian distribution with given variance
    fn random_gaussian(
        poly_context: &Arc<PolyContext>,
        representation: Representation,
        variance: usize,
    ) {
        todo!()
    }

    /// Changes representation of the polynomial to `to` representation
    pub fn change_representation(&mut self, to: Representation) {
        if self.representation == Representation::Evaluation {
            if to == Representation::Coefficient {
                izip!(
                    self.coefficients.outer_iter_mut(),
                    self.context.ntt_ops.iter()
                )
                .for_each(|(mut coefficients, ntt)| {
                    ntt.backward(coefficients.as_slice_mut().unwrap())
                });
            } else {
            }
        } else if self.representation == Representation::Coefficient {
            if to == Representation::Evaluation {
                izip!(
                    self.coefficients.outer_iter_mut(),
                    self.context.ntt_ops.iter()
                )
                .for_each(|(mut coefficients, ntt)| {
                    ntt.forward(coefficients.as_slice_mut().unwrap())
                });
            } else {
            }
        } else {
        }
    }

    //TODO: add rest of the operations needed to scale, switch context, and other necessary ops required for bfv.
}

impl AddAssign<&Poly> for Poly {
    fn add_assign(&mut self, rhs: &Poly) {
        izip!(
            self.coefficients.outer_iter_mut(),
            rhs.coefficients.outer_iter(),
            self.context.moduli_ops.iter()
        )
        .for_each(|(mut p1, p2, q)| q.add_vec(p1.as_slice_mut().unwrap(), p2.as_slice().unwrap()));
    }
}

impl Add<&Poly> for &Poly {
    type Output = Poly;
    fn add(self, rhs: &Poly) -> Self::Output {
        let mut lhs = self.clone();
        lhs += rhs;
        lhs
    }
}

impl SubAssign<&Poly> for Poly {
    fn sub_assign(&mut self, rhs: &Poly) {
        izip!(
            self.coefficients.outer_iter_mut(),
            rhs.coefficients.outer_iter(),
            self.context.moduli_ops.iter()
        )
        .for_each(|(mut p1, p2, q)| q.sub_vec(p1.as_slice_mut().unwrap(), p2.as_slice().unwrap()));
    }
}

impl Sub<&Poly> for &Poly {
    type Output = Poly;
    fn sub(self, rhs: &Poly) -> Self::Output {
        let mut lhs = self.clone();
        lhs -= rhs;
        lhs
    }
}

//TODO: Implement conversion using trait. Below method is ugly.
impl Poly {
    /// Constructs a polynomial with given BigUint values. It simply reduces any BigUint coefficient by each modulus in poly_context and assumes the specified representation.
    ///
    /// values length should be smaller than or equal to poly_context degree. Values after index > polynomial degree are ignored.
    pub fn try_convert_from_biguint(
        values: &[BigUint],
        poly_context: &Arc<PolyContext>,
        representation: &Representation,
    ) -> Poly {
        debug_assert!(poly_context.degree >= values.len());
        let mut poly = Poly::zero(poly_context, representation);

        izip!(values.iter(), poly.coefficients.axis_iter_mut(Axis(1))).for_each(
            |(v, mut rests)| {
                izip!(rests.iter_mut(), poly_context.moduli.iter()).for_each(|(xi, qi)| {
                    *xi = (v % qi).to_u64().unwrap();
                })
            },
        );

        poly
    }
}
