use fhe_math::zq::{ntt::NttOperator, Modulus};
use itertools::{izip, Itertools};
use ndarray::{Array2, Axis};
use num_bigint::{BigInt, BigUint};
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{identities::One, ToPrimitive, Zero};
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
/// 4. q_hats and q_hat_invs
///
#[derive(Clone, Debug)]
pub struct PolyContext {
    pub moduli: Box<[u64]>,
    moduli_ops: Box<[Modulus]>,
    ntt_ops: Box<[NttOperator]>,
    q_hat: Box<[BigUint]>,
    q_hat_inv: Box<[BigUint]>,
    q: BigUint,
    q_dig: BigUintDig,
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

        let mut q = BigUint::one();
        let mut q_dig = BigUintDig::one();
        moduli.iter().for_each(|qi| {
            q *= *qi;
            q_dig *= *qi;
        });

        let mut q_hat = vec![];
        let mut q_hat_inv = vec![];
        moduli.iter().for_each(|qi| {
            q_hat.push(&q / qi);

            let q_hat = &q_dig / qi;
            q_hat_inv.push(BigUint::from_bytes_le(
                &q_hat
                    .mod_inverse(BigUintDig::from(*qi))
                    .unwrap()
                    .to_biguint()
                    .unwrap()
                    .to_bytes_le(),
            ));
        });

        PolyContext {
            moduli: moduli.to_vec().into_boxed_slice(),
            moduli_ops: moduli_ops.into_boxed_slice(),
            ntt_ops: ntt_ops.into_boxed_slice(),
            degree,
            q_hat: q_hat.into_boxed_slice(),
            q_hat_inv: q_hat_inv.into_boxed_slice(),
            q,
            q_dig,
        }
    }

    pub fn modulus(&self) -> BigUint {
        self.q.clone()
    }

    pub fn modulus_dig(&self) -> BigUintDig {
        self.q_dig.clone()
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
    pub fn scale_and_round_decryption(
        &mut self,
        t: &Modulus,
        b: usize,
        t_qhat_inv_modq_divq_modt: &[u64],
        t_bqhat_inv_modq_divq_modt: &[u64],
        t_qhat_inv_modq_divq_frac: &[f64],
        t_bqhat_inv_modq_divq_frac: &[f64],
    ) -> Vec<u64> {
        let t_f64 = t.p.to_f64().unwrap();
        let t_inv = 1.0 / t_f64;

        let mut t_values = vec![];

        izip!(self.coefficients.axis_iter(Axis(1))).for_each(|rests| {
            let mut rational_sum = 0u64;
            let mut fractional_sum = 0f64;

            izip!(
                rests.iter(),
                t_qhat_inv_modq_divq_modt.iter(),
                t_bqhat_inv_modq_divq_modt.iter(),
                t_qhat_inv_modq_divq_frac.iter(),
                t_bqhat_inv_modq_divq_frac.iter()
            )
            .for_each(|(xi, rational, brational, fractional, bfractional)| {
                let xi_hi = xi >> b;
                let xi_lo = xi - (xi_hi << b);

                rational_sum = t.add(rational_sum, t.mul(xi_lo, *rational));
                rational_sum = t.add(rational_sum, t.mul(xi_hi, *brational));

                fractional_sum += xi_lo.to_f64().unwrap() * fractional;
                fractional_sum += xi_hi.to_f64().unwrap() * bfractional;
            });

            fractional_sum += rational_sum.to_f64().unwrap();

            // round
            fractional_sum += 0.5;

            let quotient = (fractional_sum * t_inv).floor();
            t_values.push((fractional_sum - (quotient * t_f64)).to_u64().unwrap());
        });

        t_values
    }
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

impl From<&Poly> for Vec<BigUint> {
    fn from(p: &Poly) -> Vec<BigUint> {
        let mut values = vec![];
        p.coefficients.axis_iter(Axis(1)).for_each(|rests| {
            let mut v = BigUint::zero();
            izip!(
                rests.iter(),
                p.context.q_hat.iter(),
                p.context.q_hat_inv.iter(),
                p.context.moduli.iter()
            )
            .for_each(|(xi, qi_hat, qi_hat_inv, qi)| {
                v += ((xi * qi_hat_inv) % qi) * qi_hat;
            });
            values.push(v % p.context.modulus());
        });
        values
    }
}

//TODO: write tests for poly
mod test {
    use num_bigint_dig::UniformBigUint;
    use num_traits::Zero;
    use rand::{
        distributions::{uniform::UniformSampler, Uniform},
        thread_rng, Rng,
    };

    use super::*;
    use crate::{nb_theory::generate_prime, BfvParameters};

    #[test]
    fn test_scale_and_round_decryption() {
        let mut rng = thread_rng();
        let bfv_params = BfvParameters::new(&[60, 60, 60, 60], 65537, 8);

        let top_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&top_context, &Representation::Coefficient, &mut rng);

        // let's scale q_poly by t/Q and switch its context from Q to t.
        let t_coeffs = q_poly.scale_and_round_decryption(
            &Modulus::new(bfv_params.plaintext_modulus).unwrap(),
            bfv_params.max_bit_size_by2,
            &bfv_params.t_qlhat_inv_modql_divql_modt[0],
            &bfv_params.t_bqlhat_inv_modql_divql_modt[0],
            &bfv_params.t_qlhat_inv_modql_divql_frac[0],
            &bfv_params.t_bqlhat_inv_modql_divql_frac[0],
        );

        let q = q_poly.context.modulus();
        let t = bfv_params.plaintext_modulus;
        let t_expected = izip!(Vec::<BigUint>::from(&q_poly))
            .map(|qi| -> BigUint {
                if (&qi >= &(&q >> 1)) {
                    if &q & BigUint::one() == BigUint::zero() {
                        t - ((((t * (&q - &qi)) + ((&q >> 1) - 1u64)) / &q) % t)
                    } else {
                        t - ((((t * (&q - &qi)) + (&q >> 1)) / &q) % t)
                    }
                } else {
                    (((&qi * t) + (&q >> 1)) / &q) % t
                }
            })
            .map(|value| value.to_u64().unwrap())
            .collect_vec();

        assert_eq!(t_coeffs, t_expected);
    }

    #[test]
    pub fn test_poly_to_biguint() {
        let rng = thread_rng();
        let values = rng
            .sample_iter(Uniform::new(0u128, 1 << 127))
            .take(8)
            .map(BigUint::from)
            .collect_vec();

        let bfv_params = BfvParameters::new(&[60, 60, 60, 60], 65537, 8);
        let top_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let q_poly =
            Poly::try_convert_from_biguint(&values, &top_context, &Representation::Coefficient);

        assert_eq!(values, Vec::<BigUint>::from(&q_poly));
    }
}
