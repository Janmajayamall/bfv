use crate::modulus::Modulus;
use crypto_bigint::U192;
use fhe_math::{zq::ntt::NttOperator, zq::Modulus as ModulusOld};
use fhe_util::sample_vec_cbd;
use itertools::{izip, Itertools};
use ndarray::{azip, s, Array2, ArrayView2, Axis, IntoNdProducer};
use num_bigint::{BigInt, BigUint};
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{identities::One, ToPrimitive, Zero};
use rand::{CryptoRng, RngCore};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
    sync::Arc,
};

mod poly_hexl;

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum Representation {
    Evaluation,
    Coefficient,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolyContext {
    pub moduli: Box<[u64]>,
    pub moduli_ops: Box<[Modulus]>,

    #[cfg(not(feature = "hexl"))]
    pub ntt_ops: Box<[NttOperator]>,

    #[cfg(feature = "hexl")]
    pub ntt_ops: Box<[NttOperator]>,

    q_hat: Box<[BigUint]>,
    q_hat_inv: Box<[BigUint]>,
    pub g: Box<[BigUint]>,
    q: BigUint,
    q_dig: BigUintDig,
    pub degree: usize,
    pub bit_reverse: Box<[usize]>,
}

impl PolyContext {
    // Creates a new polynomial context
    pub fn new(moduli: &[u64], degree: usize) -> PolyContext {
        let moduli_ops = moduli
            .iter()
            .map(|modulus| Modulus::new(*modulus))
            .collect_vec();

        #[cfg(not(feature = "hexl"))]
        let ntt_ops = moduli_ops
            .iter()
            .map(|m| {
                let m = ModulusOld::new(m.modulus()).unwrap();
                NttOperator::new(&m, degree).unwrap()
            })
            .collect_vec();

        // TODO: crate ntt operatoes for HEXL

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

        let mut bit_reverse = (0..degree)
            .map(|v| v.reverse_bits() >> (degree.leading_zeros() + 1))
            .collect_vec()
            .into_boxed_slice();

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
            bit_reverse,
        }
    }

    pub fn modulus(&self) -> BigUint {
        self.q.clone()
    }

    pub fn modulus_dig(&self) -> BigUintDig {
        self.q_dig.clone()
    }
}

#[derive(Debug)]
pub struct Substitution {
    exponent: usize,
    power_bitrev: Box<[usize]>,
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
        }
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
    pub fn new(
        coefficients: Array2<u64>,
        poly_context: &Arc<PolyContext>,
        representation: Representation,
    ) -> Poly {
        Poly {
            coefficients,
            representation,
            context: poly_context.clone(),
        }
    }

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
        Poly::try_convert_from_i64_small(&v, poly_context, representation)
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
                    // TODO: switch between native and hexl
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
                    // TODO: switch between native and hexl
                    ntt.forward(coefficients.as_slice_mut().unwrap())
                });
                self.representation = Representation::Evaluation;
            } else {
            }
        } else {
            panic!("Unknown representation");
        }
    }

    /// Given polynomial in Q(X) returns Q(X^i) for substitution element i.
    /// In Evaluation form i must be an odd integer not a multiple of 2*degree.
    /// In Coefficient form i must be an integer not a multiple of 2*degree.
    pub fn substitute(&self, subs: &Substitution) -> Poly {
        debug_assert!(subs.exponent % (self.context.degree * 2) != 0);
        debug_assert!(self.context.degree == subs.degree);
        let mut p = Poly::zero(&self.context, &self.representation);
        if self.representation == Representation::Evaluation {
            debug_assert!(subs.exponent & 1 == 1);
            azip!(
                p.coefficients.outer_iter_mut(),
                self.coefficients.outer_iter()
            )
            .par_for_each(|mut pv, qv| {
                izip!(self.context.bit_reverse.iter(), subs.power_bitrev.iter()).for_each(
                    |(br, pr)| {
                        pv[*br] = qv[*pr];
                    },
                );
            });
        } else if self.representation == Representation::Coefficient {
            let mut exponent = 0;
            let mask = self.context.degree - 1;
            for j in 0..self.context.degree {
                izip!(
                    self.coefficients.slice(s![.., j]),
                    p.coefficients.slice_mut(s![.., mask & exponent]),
                    self.context.moduli_ops.iter()
                )
                .for_each(|(qxi, pxi, modqi)| {
                    if exponent & self.context.degree != 0 {
                        *pxi = modqi.sub_mod_fast(*pxi, *qxi);
                    } else {
                        *pxi = modqi.add_mod_fast(*pxi, *qxi);
                    }
                });

                exponent += subs.exponent;
            }
        } else {
            panic!("Unknown polynomial representation!");
        }

        p
    }

    pub fn scale_and_round_decryption(
        &self,
        t: &Modulus,
        b: usize,
        t_qhat_inv_modq_divq_modt: &[u64],
        t_bqhat_inv_modq_divq_modt: &[u64],
        t_qhat_inv_modq_divq_frac: &[f64],
        t_bqhat_inv_modq_divq_frac: &[f64],
    ) -> Vec<u64> {
        debug_assert!(self.representation == Representation::Coefficient);

        let t_f64 = t.modulus().to_f64().unwrap();
        let t_inv = 1.0 / t_f64;

        let t_values = self
            .coefficients
            .axis_iter(Axis(1))
            .into_par_iter()
            .map(|rests| {
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

                    // FIXME: this will fail in debug mode since xi_lo and xi_hi are almost always greater than t (ie plaintext modulus)
                    rational_sum = t.add_mod_fast(rational_sum, t.mul_mod_fast(xi_lo, *rational));
                    rational_sum = t.add_mod_fast(rational_sum, t.mul_mod_fast(xi_hi, *brational));

                    fractional_sum += xi_lo.to_f64().unwrap() * fractional;
                    fractional_sum += xi_hi.to_f64().unwrap() * bfractional;
                });

                fractional_sum += rational_sum.to_f64().unwrap();

                // round
                fractional_sum += 0.5;

                let quotient = (fractional_sum * t_inv).floor();
                (fractional_sum - (quotient * t_f64)).to_u64().unwrap()
            })
            .collect::<Vec<u64>>();

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

        let mut o = Poly::zero(out_context, &Representation::Coefficient);

        azip!(
            self.coefficients.axis_iter(Axis(1)).into_producer(),
            o.coefficients.axis_iter_mut(Axis(1)).into_producer()
        )
        .par_for_each(|pq_rests, mut o_rests| {
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
                        s += *input.get(j).unwrap() as u128
                            * *to_s_hat_inv_mods_divs_modo.get((i, j)).unwrap() as u128;
                    }

                    s += *pq_rests.get(output_offset + i).unwrap() as u128
                        * *to_s_hat_inv_mods_divs_modo.get((i, input_size)).unwrap() as u128;

                    let oxi = o_rests.get_mut(i).unwrap();
                    *oxi = modo.barret_reduction_u128(s);
                }
            }
            // println!("inner time1: {:?}", now.elapsed());
        });

        o
    }

    /// Given a polynomial in context with moduli Q returns a polynomial in context with moduli P by calculating [round(P/Q([poly]_Q))]_P
    ///
    /// This function additionally estimates the value of $u and subtracts it from summation of each $p_{j}. Hence, the noise accumulation
    /// due to this function is reduced to 0, otherwise $noise = \norm u_{\inf} = \frac{k}{2}$ where $k is no. of moduli in $Q. This is important
    /// because this function is called 2 times in ciphertext multiplication and directly impacts noise grwoth. However, as you did expect,
    /// this introduces some performance overhead. For ex, for loq=900 and n=2^15 performance regresses by 10-12%.
    ///
    /// Check Appendix E of 2021/204 for reference.
    pub fn fast_conv_p_over_q(
        &self,
        p_context: &Arc<PolyContext>,
        neg_pq_hat_inv_modq: &[u64],
        neg_pq_hat_inv_modq_shoup: &[u64],
        q_inv: &[f64],
        q_inv_modp: &Array2<u64>,
    ) -> Poly {
        debug_assert!(self.representation == Representation::Coefficient);

        let mut p = Poly::zero(p_context, &Representation::Coefficient);

        azip!(
            p.coefficients.axis_iter_mut(Axis(1)).into_producer(),
            self.coefficients.axis_iter(Axis(1)).into_producer()
        )
        .par_for_each(|mut p_rests, q_rests| {
            let mut nu = 0.5;
            let xiv = izip!(
                q_rests.iter(),
                neg_pq_hat_inv_modq.iter(),
                neg_pq_hat_inv_modq_shoup.iter(),
                q_inv.iter(),
                self.context.moduli_ops.iter()
            )
            .map(
                |(xi, neg_pqi_hat_inv, neg_pq_hat_inv_shoup, qi_inv, modqi)| {
                    let tmp = modqi.mul_mod_shoup(*xi, *neg_pqi_hat_inv, *neg_pq_hat_inv_shoup);
                    nu += tmp as f64 * qi_inv;
                    tmp
                },
            )
            .collect_vec();

            izip!(
                p_rests.iter_mut(),
                p_context.moduli_ops.iter(),
                q_inv_modp.outer_iter()
            )
            .for_each(|(pxi, modpj, q_inv_modpj)| {
                let mut tmp = 0u128;
                izip!(xiv.iter(), q_inv_modpj.iter()).for_each(|(v, qi_inv_modpj)| {
                    tmp += *v as u128 * *qi_inv_modpj as u128;
                });
                tmp -= nu as u128;
                *pxi = modpj.barret_reduction_u128(tmp);
            })
        });

        p

        // Note: Below is the old code for the function and I have kept it around because it baffles me!
        // This version is 200% times slower than the current version (above) and I am not sure why?
        // The current version maps multiplication of q_rests with neg_pqi_hat_inv for each modqi into
        // a local vector and then performs calculation per p_rests, that is p_rests is outer loop and local vector is inner
        // loop. What baffles me is that current version somewhat does a bit extra amount of work than
        // this version, and is still a lot faster! I suspect the reason for this is cache locality. Hence,
        // I wanted to keep this code around and check whether I can someday resolve this issue and use this
        // version!
        // let mut p = Poly::zero(p_context, &Representation::Coefficient);

        // azip!(
        //     p.coefficients.axis_iter_mut(Axis(1)).into_producer(),
        //     self.coefficients.axis_iter(Axis(1)).into_producer()
        // )
        // .par_for_each(|mut p_rests, q_rests| {
        //     izip!(
        //         q_rests.iter(),
        //         q_inv_modp.outer_iter(),
        //         neg_pq_hat_inv_modq.iter(),
        //         neg_pq_hat_inv_modq_shoup.iter(),
        //         self.context.moduli_ops.iter()
        //     )
        //     .for_each(
        //         |(xi, qi_inv_modp, neg_pqi_hat_inv, neg_pq_hat_inv_shoup, modqi)| {
        //             // TODO: replacing mul_mod_fast with mul_mod_shoup only increases perf by 4%
        //             let xi_v = modqi.mul_mod_shoup(*xi, *neg_pqi_hat_inv, *neg_pq_hat_inv_shoup);
        //             izip!(
        //                 p_rests.iter_mut(),
        //                 qi_inv_modp.iter(),
        //                 p_context.moduli_ops.iter()
        //             )
        //             .for_each(|(pxi, qi_inv, modpi)| {
        //                 let tmp = modpi.mul_mod_fast(xi_v, *qi_inv);
        //                 *pxi = modpi.add_mod_fast(*pxi, tmp);
        //             })
        //         },
        //     );
        // });
        // p
    }

    pub fn switch_crt_basis(
        &self,
        p_context: &Arc<PolyContext>,
        q_hat_modp: &Array2<u64>,
        q_hat_inv_modq: &[u64],
        q_hat_inv_modq_shoup: &[u64],
        q_inv: &[f64],
        alpha_modp: &Array2<u64>,
    ) -> Poly {
        debug_assert!(self.representation == Representation::Coefficient);

        let mut p = Poly::zero(p_context, &Representation::Coefficient);

        azip!(
            p.coefficients.axis_iter_mut(Axis(1)).into_producer(),
            self.coefficients.axis_iter(Axis(1)).into_producer()
        )
        .par_for_each(|mut p_rests, q_rests| {
            let mut xi_q_hat_inv_modq = vec![];
            let mut nu = 0.5f64;

            azip!(
                q_rests.into_producer(),
                q_hat_inv_modq.into_producer(),
                q_hat_inv_modq_shoup.into_producer(),
                q_inv.into_producer(),
                self.context.moduli_ops.into_producer()
            )
            .for_each(|xi, qi_hat_inv, qi_hat_inv_shoup, q_inv, modq| {
                // TODO: replacing mul_mod_fast with mul_mod_shoup only increases perf by 2.8% or so. I think treasure lies somewhere else
                let tmp = modq.mul_mod_shoup(*xi, *qi_hat_inv, *qi_hat_inv_shoup);
                xi_q_hat_inv_modq.push(tmp);

                nu += tmp as f64 * q_inv;
            });

            let alpha = alpha_modp.slice(s![nu as usize, ..]);

            azip!(
                p_rests.into_producer(),
                q_hat_modp.outer_iter().into_producer(),
                alpha.into_producer(),
                p_context.moduli_ops.into_producer()
            )
            .for_each(|pxj, q_hat_modpj, alpha_modpj, modpj| {
                let mut tmp = 0u128;
                izip!(xi_q_hat_inv_modq.iter(), q_hat_modpj.iter()).for_each(
                    |(xi_q_hat_inv, qi_hat_modpj)| {
                        // TODO: can FMA using HEXL
                        tmp += (*xi_q_hat_inv as u128 * *qi_hat_modpj as u128);
                    },
                );
                tmp -= *alpha_modpj as u128;
                *pxj = modpj.barret_reduction_u128(tmp);
            });
        });

        p
    }

    pub fn fast_expand_crt_basis_p_over_q(
        &self,
        p_context: &Arc<PolyContext>,
        pq_context: &Arc<PolyContext>,
        neg_pq_hat_inv_modq: &[u64],
        neg_pq_hat_inv_modq_shoup: &[u64],
        q_inv: &[f64],
        q_inv_modp: &Array2<u64>,
        p_hat_modq: &Array2<u64>,
        p_hat_inv_modp: &[u64],
        p_hat_inv_modp_shoup: &[u64],
        p_inv: &[f64],
        alpha_modq: &Array2<u64>,
    ) -> Poly {
        // if self is not in coefficient, then convert it
        let p = if self.representation == Representation::Coefficient {
            self.fast_conv_p_over_q(
                p_context,
                neg_pq_hat_inv_modq,
                neg_pq_hat_inv_modq_shoup,
                q_inv,
                q_inv_modp,
            )
        } else {
            let mut q = self.clone();
            q.change_representation(Representation::Coefficient);
            q.fast_conv_p_over_q(
                p_context,
                neg_pq_hat_inv_modq,
                neg_pq_hat_inv_modq_shoup,
                q_inv,
                q_inv_modp,
            )
        };

        // switch p to q
        let q = p.switch_crt_basis(
            &self.context,
            p_hat_modq,
            p_hat_inv_modp,
            p_hat_inv_modp_shoup,
            p_inv,
            alpha_modq,
        );

        let p_size = p_context.moduli.len();
        // output should always be in coefficient form
        let mut pq = Poly::zero(pq_context, &Representation::Coefficient);
        pq.coefficients
            .slice_mut(s![..p_size, ..])
            .assign(&p.coefficients);
        pq.coefficients
            .slice_mut(s![p_size.., ..])
            .assign(&q.coefficients);
        // izip!(
        //     pq.coefficients.outer_iter_mut(),
        //     p.coefficients.outer_iter()
        // )
        // .for_each(|(mut pq_row, p_row)| {
        //     pq_row
        //         .as_slice_mut()
        //         .unwrap()
        //         .copy_from_slice(p_row.as_slice().unwrap());
        // });
        // izip!(
        //     pq.coefficients
        //         .outer_iter_mut()
        //         .skip(p_context.moduli.len()),
        //     q.coefficients.outer_iter()
        // )
        // .for_each(|(mut pq_row, q_row)| {
        //     pq_row
        //         .as_slice_mut()
        //         .unwrap()
        //         .copy_from_slice(q_row.as_slice().unwrap());
        // });

        pq
    }

    pub fn expand_crt_basis(
        &self,
        pq_context: &Arc<PolyContext>,
        p_context: &Arc<PolyContext>,
        q_hat_modp: &Array2<u64>,
        q_hat_inv_modq: &[u64],
        q_hat_inv_modq_shoup: &[u64],
        q_inv: &[f64],
        alpha_modp: &Array2<u64>,
    ) -> Poly {
        let mut p = if self.representation == Representation::Coefficient {
            self.switch_crt_basis(
                p_context,
                q_hat_modp,
                q_hat_inv_modq,
                q_hat_inv_modq_shoup,
                q_inv,
                alpha_modp,
            )
        } else {
            let mut q = self.clone();
            q.change_representation(Representation::Coefficient);
            q.switch_crt_basis(
                p_context,
                q_hat_modp,
                q_hat_inv_modq,
                q_hat_inv_modq_shoup,
                q_inv,
                alpha_modp,
            )
        };
        p.change_representation(self.representation.clone());

        let p_size = p_context.moduli.len();
        let mut pq = Poly::zero(pq_context, &self.representation);
        pq.coefficients
            .slice_mut(s![..p_size, ..])
            .assign(&p.coefficients);
        pq.coefficients
            .slice_mut(s![p_size.., ..])
            .assign(&self.coefficients);

        pq
    }

    /// Switches CRT basis from Q to P approximately.
    ///
    /// Note: the result is approximate since overflow is ignored.
    pub fn approx_switch_crt_basis(
        q_coefficients: ArrayView2<u64>,
        q_moduli_ops: &[Modulus],
        degree: usize,
        q_hat_inv_modq: &[u64],
        q_hat_modp: &Array2<u64>,
        p_moduli_ops: &[Modulus],
    ) -> Array2<u64> {
        let mut p = Array2::<u64>::zeros((p_moduli_ops.len(), degree));

        // TODO: the computation within each loop isn't much. Does it make sense to parallelize this by Axis(1) or shall
        // we switch to parallelizing by Axis(0)
        azip!(p.axis_iter_mut(Axis(1)), q_coefficients.axis_iter(Axis(1))).par_for_each(
            |mut p_rests, q_rests| {
                let mut sum = vec![0u128; p_moduli_ops.len()];
                izip!(
                    q_rests.iter(),
                    q_hat_inv_modq.iter(),
                    q_hat_modp.outer_iter(),
                    q_moduli_ops.iter()
                )
                .for_each(|(xi, qi_hat_inv_modqi, qi_hat_modp, modqi)| {
                    // TODO: change this to mul shoup
                    let tmp = modqi.mul_mod_fast(*xi, *qi_hat_inv_modqi);
                    izip!(sum.iter_mut(), qi_hat_modp.iter()).for_each(|(vj, qi_hat_modpj)| {
                        *vj += (tmp as u128 * *qi_hat_modpj as u128)
                    });
                });

                izip!(p_rests.iter_mut(), sum.iter(), p_moduli_ops.iter())
                    .for_each(|(vj, vj_u128, modpj)| *vj = modpj.barret_reduction_u128(*vj_u128));
            },
        );
        p
    }

    /// Approx mod down
    ///
    /// Switches modulus from QP to Q and divides the result by P.
    /// Uses approx mod switch to switch from P to Q resulting in additional uP. However,
    /// we get rid uP by dividing the final value by P.
    pub fn approx_mod_down(
        &mut self,
        q_context: &Arc<PolyContext>,
        p_context: &Arc<PolyContext>,
        p_hat_inv_modp: &[u64],
        p_hat_modq: &Array2<u64>,
        p_inv_modq: &[u64],
    ) {
        debug_assert!(q_context.moduli.len() + p_context.moduli.len() == self.context.moduli.len());
        debug_assert!(self.representation == Representation::Evaluation);

        let q_size = q_context.moduli.len();

        // Change P part of QP from `Evaluation` to `Coefficient` representation
        let mut p_to_q_coefficients = Array2::zeros((q_size, self.context.degree));
        let mut p_coefficients = self.coefficients.slice_mut(s![q_size.., ..]);
        debug_assert!(p_coefficients.shape()[0] == p_context.ntt_ops.len());
        azip!(
            p_coefficients.outer_iter_mut(),
            // skip first `q_size` ntt ops
            p_context.ntt_ops.into_producer()
        )
        .par_for_each(|mut v, ntt_op| {
            ntt_op.backward(v.as_slice_mut().unwrap());
        });

        azip!(
            p_to_q_coefficients.axis_iter_mut(Axis(1)),
            p_coefficients.axis_iter(Axis(1))
        )
        .par_for_each(|mut p_to_q_rests, p_rests| {
            let mut sum = vec![0u128; q_size];

            // TODO: the computation within each loop isn't much. Does it make sense to parallelize this by Axis(1) or shall
            // we switch to parallelizing by Axis(0)
            izip!(
                p_rests.iter(),
                p_hat_inv_modp.iter(),
                p_hat_modq.outer_iter(),
                p_context.moduli_ops.iter()
            )
            .for_each(|(xi, pi_hat_inv_modpi, pi_hat_modq, modpi)| {
                // TODO: change this to mul shoup
                let tmp = modpi.mul_mod_fast(*xi, *pi_hat_inv_modpi) as u128;
                izip!(sum.iter_mut(), pi_hat_modq.iter()).for_each(|(vj, pi_hat_modqj)| {
                    *vj += tmp * *pi_hat_modqj as u128;
                });
            });

            izip!(
                p_to_q_rests.iter_mut(),
                sum.iter(),
                q_context.moduli_ops.iter()
            )
            .for_each(|(xi, xi_u128, modq)| {
                *xi = modq.barret_reduction_u128(*xi_u128);
            });
        });

        // Change P switched to Q part from `Coefficient` to `Evaluation` representation
        // Reason to switch from coefficient to evaluation form becomes apparent in next step when we multiply all values by 1/P
        azip!(
            p_to_q_coefficients.outer_iter_mut(),
            q_context.ntt_ops.into_producer()
        )
        .par_for_each(|mut v, ntt_op| {
            ntt_op.forward(v.as_slice_mut().unwrap());
        });

        self.coefficients.slice_collapse(s![..q_size, ..]);
        debug_assert!(self.coefficients.shape()[0] == q_size);
        // TODO: why is this not parallelized?
        izip!(
            self.coefficients.outer_iter_mut(),
            p_to_q_coefficients.outer_iter(),
            q_context.moduli_ops.iter(),
            p_inv_modq.iter(),
        )
        .for_each(|((mut v, switched_v, modqi, p_inv_modqi))| {
            modqi.sub_mod_fast_vec(v.as_slice_mut().unwrap(), switched_v.as_slice().unwrap());
            modqi.scalar_mul_mod_fast_vec(v.as_slice_mut().unwrap(), *p_inv_modqi);
        });

        // Switch ctx from QP to Q
        self.context = q_context.clone();
    }
}

impl AddAssign<&Poly> for Poly {
    fn add_assign(&mut self, rhs: &Poly) {
        // Note: Use debug_assert instead of assert since it takes significantly longer if the trick of just checking arc pointers in stack fails.
        debug_assert!(self.context == rhs.context);
        debug_assert!(self.representation == rhs.representation);
        izip!(
            self.coefficients.outer_iter_mut(),
            rhs.coefficients.outer_iter(),
            self.context.moduli_ops.iter()
        )
        .for_each(|(mut p1, p2, q)| {
            q.add_mod_fast_vec(p1.as_slice_mut().unwrap(), p2.as_slice().unwrap())
        });
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
        debug_assert!(self.representation == rhs.representation);
        izip!(
            self.coefficients.outer_iter_mut(),
            rhs.coefficients.outer_iter(),
            self.context.moduli_ops.iter()
        )
        .for_each(|(mut p1, p2, q)| {
            q.sub_mod_fast_vec(p1.as_slice_mut().unwrap(), p2.as_slice().unwrap())
        });
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

        azip!(
            self.coefficients.outer_iter_mut().into_producer(),
            rhs.coefficients.outer_iter().into_producer(),
            self.context.moduli_ops.into_producer()
        )
        .for_each(|mut p, p2, modqi| {
            modqi.mul_mod_fast_vec(p.as_slice_mut().unwrap(), p2.as_slice().unwrap());
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

    /// Constructs a polynomial with given i64 values with small bound and assumes the given representation.  
    ///
    /// Panics if length of values is not equal to polynomial degree
    pub fn try_convert_from_i64_small(
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
                .copy_from_slice(&qi.reduce_vec_i64_small(values).as_slice());
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
    use super::*;
    use crate::{nb_theory::generate_primes_vec, parameters::BfvParameters};
    use num_bigint::ToBigInt;
    use num_bigint_dig::UniformBigUint;
    use num_traits::{FromPrimitive, Zero};
    use rand::{
        distributions::{uniform::UniformSampler, Uniform},
        thread_rng, Rng,
    };

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
    fn substitution_works() {
        let bfv_params = BfvParameters::default(1, 1 << 3);

        let ctx = bfv_params.ciphertext_poly_contexts[0].clone();
        let mut rng = thread_rng();

        let p = Poly::random(&ctx, &Representation::Coefficient, &mut rng);
        let mut p_ntt = p.clone();
        p_ntt.change_representation(Representation::Evaluation);

        // substitution by 1 should not change p
        let subs = Substitution::new(1, bfv_params.polynomial_degree);
        assert_eq!(p, p.substitute(&subs));
        assert_eq!(p_ntt, p_ntt.substitute(&subs));

        for exp in [3, 5, 9] {
            let subs = Substitution::new(exp, bfv_params.polynomial_degree);

            let p = Poly::random(&ctx, &Representation::Coefficient, &mut rng);
            let mut p_ntt = p.clone();
            p_ntt.change_representation(Representation::Evaluation);

            let p_subs = p.substitute(&subs);
            // substitue using biguints
            let p_biguint = Vec::<BigUint>::from(&p);
            let mut p_biguint_subs = vec![BigUint::zero(); bfv_params.polynomial_degree];
            p_biguint.iter().enumerate().for_each(|(i, v)| {
                let wraps = (i * subs.exponent) / bfv_params.polynomial_degree;
                let index = (i * subs.exponent) % bfv_params.polynomial_degree;
                if (wraps & 1 == 1) {
                    p_biguint_subs[index] += (p.context.modulus() - v);
                } else {
                    p_biguint_subs[index] += v;
                }
            });
            assert_eq!(p_biguint_subs, Vec::<BigUint>::from(&p_subs));

            // check subs in evaluation form
            let p_subs_ntt = p_ntt.substitute(&subs);
            let mut p_subs_ntt_clone = p_subs_ntt.clone();
            p_subs_ntt_clone.change_representation(Representation::Coefficient);
            assert_eq!(p_subs, p_subs_ntt_clone);

            // substitution by [exp^-1]_(2*degree) on polynomial that was substitution bu exp
            // must result to original polynomial
            let exp_inv = BigUintDig::from(exp)
                .mod_inverse(BigUintDig::from(2 * ctx.degree))
                .unwrap()
                .to_usize()
                .unwrap();
            let inv_subs = Substitution::new(exp_inv, bfv_params.polynomial_degree);
            assert_eq!(p, p_subs.substitute(&inv_subs));
            assert_eq!(p_ntt, p_subs_ntt.substitute(&inv_subs));
        }
    }

    #[test]
    // FIXME: fails in debug mode. Check fn to see why.
    fn test_scale_and_round_decryption() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(10)
            .build_global()
            .unwrap();
        let mut rng = thread_rng();
        let bfv_params = BfvParameters::new(&[60, 60, 60, 60], 65537, 8);

        let top_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&top_context, &Representation::Coefficient, &mut rng);

        // let's scale q_poly by t/Q and switch its context from Q to t.
        let t_coeffs = q_poly.scale_and_round_decryption(
            &Modulus::new(bfv_params.plaintext_modulus),
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

    // Tests for Main Polynomial operations //
    #[test]
    pub fn test_fast_conv_p_over_q() {
        let mut rng = thread_rng();
        let bfv_params = BfvParameters::new(
            &[
                60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60,
            ],
            65537,
            1 << 8,
        );

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let p_poly = q_poly.fast_conv_p_over_q(
            &p_context,
            &bfv_params.neg_pql_hat_inv_modql[0],
            &bfv_params.neg_pql_hat_inv_modql_shoup[0],
            &bfv_params.ql_inv[0],
            &bfv_params.ql_inv_modp[0],
        );
        println!("time: {:?}", now.elapsed());

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
        let bfv_params = BfvParameters::new(&[60, 60], 65537, 8);

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let p_poly = q_poly.switch_crt_basis(
            &p_context,
            &bfv_params.ql_hat_modp[0],
            &bfv_params.ql_hat_inv_modql[0],
            &bfv_params.ql_hat_inv_modql_shoup[0],
            &bfv_params.ql_inv[0],
            &bfv_params.alphal_modp[0],
        );
        println!("time: {:?}", now.elapsed());

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
            65537,
            1 << 3,
        );

        let q_context = bfv_params.ciphertext_poly_contexts[0].clone();
        let p_context = bfv_params.extension_poly_contexts[0].clone();
        let pq_context = bfv_params.pq_poly_contexts[0].clone();
        let mut q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let pq_poly = q_poly.fast_expand_crt_basis_p_over_q(
            &p_context,
            &pq_context,
            &bfv_params.neg_pql_hat_inv_modql[0],
            &bfv_params.neg_pql_hat_inv_modql_shoup[0],
            &bfv_params.ql_inv[0],
            &bfv_params.ql_inv_modp[0],
            &bfv_params.pl_hat_modq[0],
            &bfv_params.pl_hat_inv_modpl[0],
            &bfv_params.pl_hat_inv_modpl_shoup[0],
            &bfv_params.pl_inv[0],
            &bfv_params.alphal_modq[0],
        );
        println!("time: {:?}", now.elapsed());

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
            65537,
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
                // dbg!(diff.bits());
                assert!(diff.bits() <= 1);
            },
        );
    }

    #[test]
    pub fn test_approx_switch_crt_basis() {
        let mut rng = thread_rng();
        let polynomial_degree = 8;
        let p_moduli = generate_primes_vec(&vec![60, 60, 60, 60, 60, 60], polynomial_degree, &[]);
        let q_moduli = p_moduli[..3].to_vec();

        let q_context = Arc::new(PolyContext::new(&q_moduli, polynomial_degree));
        let p_context = Arc::new(PolyContext::new(&p_moduli, polynomial_degree));

        let mut p_poly = Poly::zero(&p_context, &Representation::Coefficient);

        // Pre-computation
        let mut q_hat_inv_modq = vec![];
        let mut q_hat_modp = vec![];
        let q = q_context.modulus();
        let q_dig = q_context.modulus_dig();
        izip!(q_context.moduli.iter()).for_each(|(qi)| {
            let qi_hat_inv_modqi = (&q_dig / *qi)
                .mod_inverse(BigUintDig::from_u64(*qi).unwrap())
                .unwrap()
                .to_biguint()
                .unwrap()
                .to_u64()
                .unwrap();

            q_hat_inv_modq.push(qi_hat_inv_modqi);

            izip!(p_moduli.iter())
                .for_each(|pj| q_hat_modp.push(((&q / qi) % pj).to_u64().unwrap()));
        });
        let q_hat_modp =
            Array2::<u64>::from_shape_vec((q_context.moduli.len(), p_moduli.len()), q_hat_modp)
                .unwrap();

        let mut rng = thread_rng();
        let q_poly = Poly::random(&q_context, &Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let p_coefficients = Poly::approx_switch_crt_basis(
            q_poly.coefficients.view(),
            &q_context.moduli_ops,
            q_context.degree,
            &q_hat_inv_modq,
            &q_hat_modp,
            &p_context.moduli_ops,
        );
        p_poly.coefficients = p_coefficients;
        println!("time: {:?}", now.elapsed());

        // dbg!(&p_poly.coefficients);

        let q = q_context.modulus();
        let p = p_context.modulus();

        let p_expected: Vec<BigUint> = Vec::<BigUint>::from(&q_poly)
            .iter()
            .map(|xi| {
                if xi > &(&q >> 1) {
                    &p - ((&q - xi) % &p)
                } else {
                    xi % &p
                }
            })
            .collect_vec();
        izip!(Vec::<BigUint>::from(&p_poly).iter(), p_expected.iter()).for_each(|(r, e)| {
            let mut diff = r.to_bigint().unwrap() - e.to_bigint().unwrap();
            dbg!(r, e, diff.bits());
        })
    }

    #[test]
    pub fn test_approx_mod_down() {
        let mut rng = thread_rng();
        let polynomial_degree = 1 << 3;
        let q_moduli = generate_primes_vec(&vec![60, 60, 60, 60, 60, 60], polynomial_degree, &[]);
        let p_moduli = generate_primes_vec(&vec![60, 60], polynomial_degree, &q_moduli);
        let qp_moduli = [q_moduli.clone(), p_moduli.clone()].concat();

        let q_context = Arc::new(PolyContext::new(&q_moduli, polynomial_degree));
        let p_context = Arc::new(PolyContext::new(&p_moduli, polynomial_degree));
        let qp_context = Arc::new(PolyContext::new(&qp_moduli, polynomial_degree));

        let q_size = q_context.moduli.len();
        let p_size = p_context.moduli.len();
        let qp_size = q_size + p_size;

        // Pre computation
        let p = p_context.modulus();
        let p_dig = p_context.modulus_dig();
        let mut p_hat_inv_modp = vec![];
        let mut p_hat_modq = vec![];
        p_context.moduli.iter().for_each(|(pi)| {
            p_hat_inv_modp.push(
                (&p_dig / pi)
                    .mod_inverse(BigUintDig::from_u64(*pi).unwrap())
                    .unwrap()
                    .to_biguint()
                    .unwrap()
                    .to_u64()
                    .unwrap(),
            );

            // pi_hat_modq
            let p_hat = &p / pi;
            q_context
                .moduli
                .iter()
                .for_each(|qi| p_hat_modq.push((&p_hat % qi).to_u64().unwrap()));
        });
        let p_hat_modq =
            Array2::from_shape_vec((p_context.moduli.len(), q_context.moduli.len()), p_hat_modq)
                .unwrap();
        let mut p_inv_modq = vec![];
        q_context.moduli.iter().for_each(|qi| {
            p_inv_modq.push(
                p_dig
                    .clone()
                    .mod_inverse(BigUintDig::from_u64(*qi).unwrap())
                    .unwrap()
                    .to_biguint()
                    .unwrap()
                    .to_u64()
                    .unwrap(),
            );
        });

        let mut qp_poly = Poly::random(&qp_context, &Representation::Evaluation, &mut rng);
        let mut q_res = qp_poly.clone();

        let now = std::time::Instant::now();
        q_res.approx_mod_down(
            &q_context,
            &p_context,
            &p_hat_inv_modp,
            &p_hat_modq,
            &p_inv_modq,
        );
        println!("time: {:?}", now.elapsed());

        q_res.change_representation(Representation::Coefficient);

        qp_poly.change_representation(Representation::Coefficient);

        let q_poly = {
            let coeff = qp_poly.coefficients.slice(s![..q_size, ..]).to_owned();
            assert!(coeff.shape()[0] == q_size);
            Poly::new(coeff, &q_context, Representation::Coefficient)
        };
        let p_poly = {
            let coeff = qp_poly.coefficients.slice(s![q_size.., ..]).to_owned();
            assert!(coeff.shape()[0] == p_size);
            Poly::new(coeff, &p_context, Representation::Coefficient)
        };

        let qp = qp_context.modulus();
        let q = q_context.modulus();
        let p = p_context.modulus();
        let p_inv_modq = {
            let dig = p_context
                .modulus_dig()
                .mod_inverse(BigUintDig::from_bytes_le(&q.to_bytes_le()))
                .unwrap()
                .to_biguint()
                .unwrap();

            BigUint::from_bytes_le(&dig.to_bytes_le())
        };

        let p_switched = Vec::<BigUint>::from(&p_poly)
            .iter()
            .map(|xi| {
                if xi > &(&p >> 1) {
                    &q - ((&p - xi) % &q)
                } else {
                    xi % &q
                }
            })
            .collect_vec();
        // convert p_poly to q
        let q_expected = izip!(Vec::<BigUint>::from(&q_poly).iter(), p_switched.iter())
            .map(|(qv, pv)| ((qv + (&q - pv)) * &p_inv_modq) % &q)
            .collect_vec();

        let q_res: Vec<BigUint> = Vec::<BigUint>::from(&q_res)
            .iter()
            .map(|xi| xi.clone())
            .collect_vec();

        izip!(q_res.iter(), q_expected.iter()).for_each(|(res, expected)| {
            let diff: BigInt = res.to_bigint().unwrap() - expected.to_bigint().unwrap();
            // assert!(diff <= BigInt::one());
            dbg!(diff);
        });
    }

    #[test]
    fn test_one() {
        let mut arr = Array2::<u64>::zeros((5, 5));
        arr.slice_collapse(s![..3, ..]);
        dbg!(arr.shape());
    }
}
