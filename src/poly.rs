use crypto_bigint::U192;
use fhe_math::{
    rq::Context,
    zq::{ntt::NttOperator, Modulus},
};
use fhe_util::sample_vec_cbd;
use itertools::{izip, DedupBy, Itertools};
use ndarray::{s, Array2, Axis};
use num_bigint::{BigInt, BigUint};
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{identities::One, ToPrimitive, Zero};
use rand::{CryptoRng, RngCore};
use std::{
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
    sync::Arc,
};

#[derive(Clone, PartialEq, Debug, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyContext {
    pub moduli: Box<[u64]>,
    pub moduli_ops: Box<[Modulus]>,
    ntt_ops: Box<[NttOperator]>,
    q_hat: Box<[BigUint]>,
    q_hat_inv: Box<[BigUint]>,
    pub g: Box<[BigUint]>,
    q: BigUint,
    q_dig: BigUintDig,
    pub degree: usize,
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
        // g = q_hat * q_hat_inv
        let mut g = vec![];
        moduli.iter().for_each(|qi| {
            let qh = &q / qi;
            q_hat.push(qh.clone());

            let qhi = BigUint::from_bytes_le(
                &(&q_dig / qi)
                    .mod_inverse(BigUintDig::from(*qi))
                    .unwrap()
                    .to_biguint()
                    .unwrap()
                    .to_bytes_le(),
            );
            q_hat_inv.push(qhi.clone());
            g.push(qh * qhi);
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
            g: g.into_boxed_slice(),
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    pub coefficients: Array2<u64>,
    pub representation: Representation,
    pub context: Arc<PolyContext>,
}

impl Poly {
    /// Creates zero polynomial with a given context and representation
    pub fn zero(poly_context: &Arc<PolyContext>, representation: &Representation) -> Poly {
        Poly {
            coefficients: Array2::zeros((poly_context.moduli.len(), poly_context.degree)),
            representation: representation.clone(),
            context: poly_context.clone(),
        }
    }

    /// Creates a polynomial with random values for given context and representation
    pub fn random<R: RngCore + CryptoRng>(
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
    pub fn random_gaussian<R: CryptoRng + RngCore>(
        poly_context: &Arc<PolyContext>,
        representation: &Representation,
        variance: usize,
        rng: &mut R,
    ) -> Poly {
        // TODO: replace this
        let v = sample_vec_cbd(poly_context.degree, variance, rng).unwrap();
        Poly::try_convert_from_i64(&v, poly_context, representation)
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
                self.representation = Representation::Coefficient;
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
                self.representation = Representation::Evaluation;
            } else {
            }
        } else {
            panic!("Unknown representation");
        }
    }

    //TODO: add rest of the operations needed to scale, switch context, and other necessary ops required for bfv.
    pub fn scale_and_round_decryption(
        &self,
        t: &Modulus,
        b: usize,
        t_qhat_inv_modq_divq_modt: &[u64],
        t_bqhat_inv_modq_divq_modt: &[u64],
        t_qhat_inv_modq_divq_frac: &[f64],
        t_bqhat_inv_modq_divq_frac: &[f64],
    ) -> Vec<u64> {
        assert!(self.representation == Representation::Coefficient);

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

    /// Assumes the polynomial is in context with modulus PQ, and the out_context is either P or Q
    ///
    /// If out_context is P, scales the polynomial by t/Q otherwise scales the polynomial by t/P
    pub fn scale_and_round(
        &self,
        out_context: &Arc<PolyContext>,
        p_context: &Arc<PolyContext>,
        q_context: &Arc<PolyContext>,
        to_s_hat_inv_mods_divs_modo: &Array2<u64>,
        to_s_hat_inv_mods_divs_frachi: &[u64],
        to_s_hat_inv_mods_divs_fraclo: &[u64],
    ) -> Poly {
        let mut o = Poly::zero(out_context, &Representation::Coefficient);

        let mut input_offset = 0;
        let mut output_offset = 0;
        let mut input_size = 0;
        if out_context == p_context {
            input_offset = p_context.moduli.len();
            input_size = q_context.moduli.len();
        } else {
            output_offset = p_context.moduli.len();
            input_size = p_context.moduli.len();
        }

        izip!(
            self.coefficients.axis_iter(Axis(1)),
            o.coefficients.axis_iter_mut(Axis(1))
        )
        .for_each(|(pq_rests, mut o_rests)| {
            let mut frac = U192::ZERO;
            izip!(
                pq_rests
                    .slice(s![input_offset..input_offset + input_size])
                    .iter(),
                to_s_hat_inv_mods_divs_frachi.iter(),
                to_s_hat_inv_mods_divs_fraclo.iter()
            )
            .for_each(|(xi, frac_hi, frac_lo)| {
                let lo = *xi as u128 * *frac_lo as u128;
                let hi = (*xi as u128 * *frac_hi as u128) + (lo >> 64);
                frac =
                    frac.wrapping_add(&U192::from_words([lo as u64, hi as u64, (hi >> 64) as u64]));
            });

            let frac = frac.shr_vartime(127).as_words()[0] as u128;

            // let mut now = std::time::Instant::now();
            unsafe {
                let input = pq_rests.slice(s![input_offset..input_offset + input_size]);
                for i in 0..o_rests.len() {
                    let modo = out_context.moduli_ops.get_unchecked(i);

                    let mut s = frac;
                    for j in 0..input.len() {
                        s += modo.mul(
                            *input.get(j).unwrap(),
                            *to_s_hat_inv_mods_divs_modo.get((i, j)).unwrap(),
                        ) as u128;
                    }

                    s += modo.mul(
                        *pq_rests.get(output_offset + i).unwrap(),
                        *to_s_hat_inv_mods_divs_modo.get((i, input_size)).unwrap(),
                    ) as u128;

                    let oxi = o_rests.get_mut(i).unwrap();
                    *oxi = modo.reduce_u128(s);
                }
            }
            // println!("inner time1: {:?}", now.elapsed());
        });

        o
    }

    /// Given a polynomial in context with moduli Q returns a polynomial in context with moduli P by calculating [round(P/Q([poly]_Q))]_P
    pub fn fast_conv_p_over_q(
        &self,
        p_context: &Arc<PolyContext>,
        neg_pq_hat_inv_modq: &[u64],
        q_inv_modp: &Array2<u64>,
    ) -> Poly {
        debug_assert!(self.representation == Representation::Coefficient);

        let mut p = Poly::zero(p_context, &Representation::Coefficient);

        // TODO: make parallel
        izip!(
            p.coefficients.axis_iter_mut(Axis(1)),
            self.coefficients.axis_iter(Axis(1))
        )
        .for_each(|(mut p_rests, q_rests)| {
            izip!(
                q_rests.iter(),
                q_inv_modp.outer_iter(),
                neg_pq_hat_inv_modq.iter(),
                self.context.moduli_ops.iter()
            )
            .for_each(|(xi, qi_inv_modp, neg_pqi_hat_inv, modqi)| {
                let xi_v = modqi.mul(*xi, *neg_pqi_hat_inv);
                izip!(
                    p_rests.iter_mut(),
                    qi_inv_modp.iter(),
                    p_context.moduli_ops.iter()
                )
                .for_each(|(pxi, qi_inv, modpi)| {
                    let tmp = modpi.mul(xi_v, *qi_inv);
                    *pxi = modpi.add(*pxi, tmp);
                })
            });
        });
        p
    }

    pub fn switch_crt_basis(
        &self,
        p_context: &Arc<PolyContext>,
        q_hat_modp: &Array2<u64>,
        q_hat_inv_modq: &[u64],
        q_inv: &[f64],
        alpha_modp: &Array2<u64>,
    ) -> Poly {
        debug_assert!(self.representation == Representation::Coefficient);

        let mut p = Poly::zero(p_context, &Representation::Coefficient);
        izip!(
            p.coefficients.axis_iter_mut(Axis(1)),
            self.coefficients.axis_iter(Axis(1))
        )
        .for_each(|(mut p_rests, q_rests)| {
            let mut xi_q_hat_inv_modq = vec![];
            let mut nu = 0.5f64;
            izip!(
                q_rests.iter(),
                q_hat_inv_modq.iter(),
                q_inv.iter(),
                self.context.moduli_ops.iter()
            )
            .for_each(|(xi, qi_hat_inv, q_inv, modq)| {
                let tmp = modq.mul(*xi, *qi_hat_inv);
                xi_q_hat_inv_modq.push(tmp);

                nu += tmp as f64 * q_inv;
            });

            let alpha = alpha_modp.slice(s![nu as usize, ..]);

            izip!(
                p_rests.iter_mut(),
                q_hat_modp.outer_iter(),
                p_context.moduli_ops.iter(),
                alpha.iter(),
            )
            .for_each(|(pxi, q_hat_modpi, modpi, alpha_modpi)| {
                izip!(xi_q_hat_inv_modq.iter(), q_hat_modpi.iter()).for_each(
                    |(xi_q_hat_inv, qi_hat)| {
                        *pxi = modpi.add(*pxi, modpi.mul(*xi_q_hat_inv, *qi_hat));
                    },
                );

                *pxi = modpi.sub(*pxi, *alpha_modpi);
            });
        });

        p
    }

    pub fn fast_expand_crt_basis_p_over_q(
        &self,
        p_context: &Arc<PolyContext>,
        pq_context: &Arc<PolyContext>,
        neg_pq_hat_inv_modq: &[u64],
        q_inv_modp: &Array2<u64>,
        p_hat_modq: &Array2<u64>,
        p_hat_inv_modp: &[u64],
        p_inv: &[f64],
        alpha_modq: &Array2<u64>,
    ) -> Poly {
        let p = self.fast_conv_p_over_q(p_context, neg_pq_hat_inv_modq, q_inv_modp);

        // switch p to q
        let q = p.switch_crt_basis(&self.context, p_hat_modq, p_hat_inv_modp, p_inv, alpha_modq);

        let mut pq = Poly::zero(pq_context, &Representation::Coefficient);
        izip!(
            pq.coefficients.outer_iter_mut(),
            p.coefficients.outer_iter()
        )
        .for_each(|(mut pq_row, p_row)| {
            pq_row
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(p_row.as_slice().unwrap());
        });
        izip!(
            pq.coefficients
                .outer_iter_mut()
                .skip(p_context.moduli.len()),
            q.coefficients.outer_iter()
        )
        .for_each(|(mut pq_row, q_row)| {
            pq_row
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(q_row.as_slice().unwrap());
        });

        pq
    }

    pub fn expand_crt_basis(
        &mut self,
        pq_context: &Arc<PolyContext>,
        p_context: &Arc<PolyContext>,
        q_hat_modp: &Array2<u64>,
        q_hat_inv_modq: &[u64],
        q_inv: &[f64],
        alpha_modp: &Array2<u64>,
    ) -> Poly {
        let representation_cache = self.representation.clone();
        let mut ntt_cache: Option<Array2<u64>> = None;

        if self.representation == Representation::Evaluation {
            // Save coefficients to avoid ntt operations later
            ntt_cache = Some(self.coefficients.clone());
            self.change_representation(Representation::Coefficient);
        }

        let mut p = self.switch_crt_basis(p_context, q_hat_modp, q_hat_inv_modq, q_inv, alpha_modp);
        p.change_representation(representation_cache.clone());

        let mut pq = Poly::zero(pq_context, &representation_cache);
        izip!(
            pq.coefficients.outer_iter_mut(),
            p.coefficients.outer_iter()
        )
        .for_each(|(mut pq_row, p_row)| {
            pq_row
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(p_row.as_slice().unwrap());
        });

        if let Some(ntt_cache) = ntt_cache {
            izip!(
                pq.coefficients
                    .outer_iter_mut()
                    .skip(p_context.moduli.len()),
                ntt_cache.outer_iter()
            )
            .for_each(|(mut pq_row, q_row)| {
                pq_row
                    .as_slice_mut()
                    .unwrap()
                    .copy_from_slice(q_row.as_slice().unwrap());
            });
        } else {
            self.change_representation(representation_cache);
            izip!(
                pq.coefficients
                    .outer_iter_mut()
                    .skip(p_context.moduli.len()),
                self.coefficients.outer_iter()
            )
            .for_each(|(mut pq_row, q_row)| {
                pq_row
                    .as_slice_mut()
                    .unwrap()
                    .copy_from_slice(q_row.as_slice().unwrap());
            });
        }

        pq
    }
}

impl AddAssign<&Poly> for Poly {
    fn add_assign(&mut self, rhs: &Poly) {
        // Note: Use debug_assert instead of assert since it takes significantly longer trick of just checking arc pointers in stack fails.
        debug_assert!(self.context == rhs.context);
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
        debug_assert!(self.context == rhs.context);
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

impl MulAssign<&Poly> for Poly {
    fn mul_assign(&mut self, rhs: &Poly) {
        debug_assert!(self.context == rhs.context);

        assert!(self.representation == rhs.representation);
        assert!(self.representation == Representation::Evaluation);

        izip!(
            self.coefficients.outer_iter_mut(),
            rhs.coefficients.outer_iter(),
            self.context.moduli_ops.iter()
        )
        .for_each(|(mut p, p2, qi)| {
            qi.mul_vec(p.as_slice_mut().unwrap(), p2.as_slice().unwrap());
        });
    }
}

impl Mul<&Poly> for &Poly {
    type Output = Poly;
    fn mul(self, rhs: &Poly) -> Self::Output {
        let mut lhs = self.clone();
        lhs *= rhs;
        lhs
    }
}

//TODO: Implement conversion using trait. Below method is ugly.
impl Poly {
    pub fn try_convert_from_u64(
        values: &[u64],
        poly_context: &Arc<PolyContext>,
        representation: &Representation,
    ) -> Poly {
        assert!(values.len() == poly_context.degree);
        let mut p = Poly::zero(poly_context, representation);
        izip!(
            p.coefficients.outer_iter_mut(),
            poly_context.moduli_ops.iter()
        )
        .for_each(|(mut qi_values, qi)| {
            let mut xi = values.to_vec();
            qi.reduce_vec(&mut xi);
            qi_values.as_slice_mut().unwrap().copy_from_slice(&xi);
        });
        p
    }

    /// Constructs a polynomial with given BigUint values. It simply reduces each BigUint coefficient by every modulus in poly_context and assumes the specified representation.
    ///
    /// values length should be smaller than or equal to poly_context degree. Values at index beyond polynomial degree are ignored.
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

    /// Constructs a polynomial with given i32 values and assumes the given representation.  
    ///
    /// Panics if length of values is not equal to polynomial degree
    pub fn try_convert_from_i64(
        values: &[i64],
        poly_context: &Arc<PolyContext>,
        representation: &Representation,
    ) -> Poly {
        assert!(values.len() == poly_context.degree);
        let mut p = Poly::zero(poly_context, representation);
        izip!(
            p.coefficients.outer_iter_mut(),
            poly_context.moduli_ops.iter()
        )
        .for_each(|(mut qi_values, qi)| {
            qi_values
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(qi.reduce_vec_i64(values).as_slice());
        });
        p
    }
}

impl From<&Poly> for Vec<BigUint> {
    fn from(p: &Poly) -> Vec<BigUint> {
        assert!(p.representation == Representation::Coefficient);
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
mod tests {
    use num_bigint::ToBigInt;
    use num_bigint_dig::UniformBigUint;
    use num_traits::Zero;
    use rand::{
        distributions::{uniform::UniformSampler, Uniform},
        thread_rng, Rng,
    };

    use super::*;
    use crate::{nb_theory::generate_prime, BfvParameters};

    #[test]
    fn test_scale_and_roun1d_decryption() {
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

    #[test]
    pub fn test_fast_conv_p_over_q() {
        let mut rng = thread_rng();
        let bfv_params = BfvParameters::new(
            &[
                60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
            ],
            1153,
            8,
        );

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);
        let p_poly = q_poly.fast_conv_p_over_q(
            &p_context,
            &bfv_params.neg_pql_hat_inv_modql[0],
            &bfv_params.ql_inv_modp[0],
        );

        let q = q_context.modulus();
        let p = p_context.modulus();
        let p_expected: Vec<BigUint> = Vec::<BigUint>::from(&q_poly)
            .iter()
            .map(|xi| {
                if xi >= &(&q >> 1usize) {
                    if &q & BigUint::one() == BigUint::zero() {
                        &p - &(((((&q - xi) * &p) + ((&q >> 1) - 1usize)) / &q) % &p)
                    } else {
                        &p - &(((((&q - xi) * &p) + (&q >> 1)) / &q) % &p)
                    }
                } else {
                    (((xi * &p) + (&q >> 1)) / &q) % &p
                }
            })
            .collect();

        izip!(Vec::<BigUint>::from(&p_poly).iter(), p_expected.iter()).for_each(
            |(res, expected)| {
                let diff: BigInt = res.to_bigint().unwrap() - expected.to_bigint().unwrap();
                dbg!(diff.bits());
            },
        );
    }

    #[test]
    pub fn test_switch_crt_basis() {
        let mut rng = thread_rng();
        let bfv_params = BfvParameters::new(&[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60], 1153, 8);

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);
        let p_poly = q_poly.switch_crt_basis(
            &p_context,
            &bfv_params.ql_hat_modp[0],
            &bfv_params.ql_hat_inv_modql[0],
            &bfv_params.ql_inv[0],
            &bfv_params.alphal_modp[0],
        );

        let q = q_context.modulus();
        let p = p_context.modulus();
        let p_expected: Vec<BigUint> = Vec::<BigUint>::from(&q_poly)
            .iter()
            .map(|xi| {
                if xi >= &(&q >> 1) {
                    &p - ((&q - xi) % &p)
                } else {
                    xi % &p
                }
            })
            .collect();

        assert_eq!(p_expected, Vec::<BigUint>::from(&p_poly));
    }

    #[test]
    pub fn test_fast_expand_crt_basis_p_over_q() {
        let mut rng = thread_rng();
        let bfv_params = BfvParameters::new(
            &[
                60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                60, 60, 60, 60, 60, 60, 60,
            ],
            1153,
            8,
        );

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let pq_context = bfv_params.pq_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

        let pq_poly = q_poly.fast_expand_crt_basis_p_over_q(
            &p_context,
            &pq_context,
            &bfv_params.neg_pql_hat_inv_modql[0],
            &bfv_params.ql_inv_modp[0],
            &bfv_params.pl_hat_modq[0],
            &bfv_params.pl_hat_inv_modpl[0],
            &bfv_params.pl_inv[0],
            &bfv_params.alphal_modq[0],
        );

        let q = q_context.modulus();
        let p = p_context.modulus();
        let pq = pq_context.modulus();
        let p_expected: Vec<BigUint> = Vec::<BigUint>::from(&q_poly)
            .iter()
            .map(|xi| {
                if xi >= &(&q >> 1usize) {
                    if &q & BigUint::one() == BigUint::zero() {
                        &pq - &(((((&q - xi) * &p) + ((&q >> 1) - 1usize)) / &q) % &pq)
                    } else {
                        &pq - &(((((&q - xi) * &p) + (&q >> 1)) / &q) % &pq)
                    }
                } else {
                    (((xi * &p) + (&q >> 1)) / &q) % &pq
                }
            })
            .collect();

        izip!(Vec::<BigUint>::from(&pq_poly).iter(), p_expected.iter()).for_each(
            |(res, expected)| {
                let diff: BigInt = res.to_bigint().unwrap() - expected.to_bigint().unwrap();
                dbg!(diff.bits());
            },
        );
    }

    #[test]
    pub fn test_scale_and_round() {
        let mut rng = thread_rng();
        let bfv_params = BfvParameters::new(
            &[
                60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
                60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
            ],
            1553,
            1 << 3,
        );

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let pq_context = bfv_params.pq_poly_contexts[0].clone();

        let pq_poly = Poly::random(&pq_context, &Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let q_poly = pq_poly.scale_and_round(
            &q_context,
            &p_context,
            &q_context,
            &bfv_params.tql_p_hat_inv_modp_divp_modql[0],
            &bfv_params.tql_p_hat_inv_modp_divp_frac_hi[0],
            &bfv_params.tql_p_hat_inv_modp_divp_frac_lo[0],
        );
        println!("time1: {:?}", now.elapsed());

        let t = bfv_params.plaintext_modulus;
        let p = p_context.modulus();
        let q = q_context.modulus();
        let pq = pq_context.modulus();
        let q_expected: Vec<BigUint> = Vec::<BigUint>::from(&pq_poly)
            .iter()
            .map(|xi| {
                if xi >= &(&pq >> 1usize) {
                    if &pq & BigUint::one() == BigUint::zero() {
                        &q - &(((((&pq - xi) * &t) + ((&p >> 1) - 1usize)) / &p) % &q)
                    } else {
                        &q - &(((((&pq - xi) * &t) + (&p >> 1)) / &p) % &q)
                    }
                } else {
                    (((xi * &t) + (&p >> 1)) / &p) % &q
                }
            })
            .collect();

        izip!(Vec::<BigUint>::from(&q_poly).iter(), q_expected.iter()).for_each(
            |(res, expected)| {
                let diff: BigInt = res.to_bigint().unwrap() - expected.to_bigint().unwrap();
                dbg!(diff.bits());
                assert!(diff.bits() <= 1);
            },
        );
    }
}
