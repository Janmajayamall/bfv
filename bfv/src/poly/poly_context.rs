use crate::Poly;
use crate::{modulus::Modulus, utils::mod_inverse_biguint_u64, Representation, Substitution};
use crypto_bigint::U192;
use fhe_util::sample_vec_cbd;
use itertools::{izip, Itertools};
use ndarray::{azip, s, Array2, ArrayView2, Axis, IntoNdProducer};
use num_bigint::BigUint;
use num_traits::{identities::One, ToPrimitive, Zero};
use rand::{CryptoRng, RngCore};
use seq_macro::seq;
use std::mem::{self, MaybeUninit};
use traits::Ntt;

#[derive(PartialEq)]
pub struct PolyContext<'a, T: Ntt> {
    pub(crate) moduli_ops: (&'a [Modulus], &'a [Modulus]),
    pub(crate) ntt_ops: (&'a [T], &'a [T]),
    pub(crate) moduli_count: usize,
    pub(crate) degree: usize,
}

impl<T> PolyContext<'_, T>
where
    T: Ntt,
{
    pub fn iter_moduli_ops(&self) -> impl Iterator<Item = &Modulus> {
        self.moduli_ops.0.iter().chain(self.moduli_ops.1.iter())
    }

    pub fn iter_ntt_ops(&self) -> impl Iterator<Item = &T> {
        self.ntt_ops.0.iter().chain(self.ntt_ops.1.iter())
    }

    pub fn big_q(&self) -> BigUint {
        let mut q = BigUint::one();
        self.iter_moduli_ops().for_each(|qi| {
            q *= qi.modulus();
        });
        q
    }

    pub fn moduli_ops(&self) -> &[Modulus] {
        debug_assert!(self.moduli_ops.1.len() == 0);
        self.moduli_ops.0
    }

    pub fn ntt_ops(&self) -> &[T] {
        debug_assert!(self.ntt_ops.1.len() == 0);
        self.ntt_ops.0
    }

    pub fn moduli_count(&self) -> usize {
        self.moduli_count
    }
}

impl<T> PolyContext<'_, T>
where
    T: Ntt,
{
    pub fn new(&self, coefficients: Array2<u64>, representation: Representation) -> Poly {
        Poly {
            coefficients,
            representation,
        }
    }

    /// Creates zero polynomial with a given context and representation
    pub fn zero(&self, representation: Representation) -> Poly {
        Poly {
            coefficients: Array2::zeros((self.moduli_count, self.degree)),
            representation: representation,
        }
    }

    /// Creates a polynomial with random values for given context and representation
    pub fn random<R: RngCore + CryptoRng>(
        &self,
        representation: Representation,
        rng: &mut R,
    ) -> Poly {
        let mut poly = self.zero(representation);
        izip!(poly.coefficients.outer_iter_mut(), self.iter_moduli_ops()).for_each(
            |(mut coefficients, q)| {
                coefficients
                    .as_slice_mut()
                    .unwrap()
                    .copy_from_slice(q.random_vec(self.degree, rng).as_slice());
            },
        );
        poly
    }

    /// Creates a polynomial with random values sampled from gaussian distribution with given variance
    pub fn random_gaussian<R: CryptoRng + RngCore>(
        &self,
        representation: Representation,
        variance: usize,
        rng: &mut R,
    ) -> Poly {
        // TODO: replace this
        let v = sample_vec_cbd(self.degree, variance, rng).unwrap();
        self.try_convert_from_i64_small(&v, representation)
    }

    /// Changes representation of the polynomial to `to` representation
    pub fn change_representation(&self, poly: &mut Poly, to: Representation) {
        if poly.representation == Representation::Evaluation {
            if to == Representation::Coefficient {
                izip!(poly.coefficients.outer_iter_mut(), self.iter_ntt_ops()).for_each(
                    |(mut coefficients, ntt)| ntt.backward(coefficients.as_slice_mut().unwrap()),
                );
                poly.representation = Representation::Coefficient;
            } else {
            }
        } else if poly.representation == Representation::Coefficient {
            if to == Representation::Evaluation {
                izip!(poly.coefficients.outer_iter_mut(), self.iter_ntt_ops()).for_each(
                    |(mut coefficients, ntt)| {
                        // TODO: switch between native and hexl
                        ntt.forward(coefficients.as_slice_mut().unwrap())
                    },
                );
                poly.representation = Representation::Evaluation;
            } else {
            }
        } else {
            panic!("Unknown representation");
        }
    }

    /// Given polynomial in Q(X) returns Q(X^i) for substitution element i.
    /// In Evaluation form i must be an odd integer not a multiple of 2*degree.
    /// In Coefficient form i must be an integer not a multiple of 2*degree.
    pub fn substitute(&self, poly: &Poly, subs: &Substitution) -> Poly {
        debug_assert!(subs.exponent % (self.degree * 2) != 0);
        debug_assert!(self.degree == subs.degree);
        let mut p = self.zero(poly.representation.clone());
        if poly.representation == Representation::Evaluation {
            debug_assert!(subs.exponent & 1 == 1);
            izip!(
                p.coefficients.outer_iter_mut(),
                poly.coefficients.outer_iter()
            )
            .for_each(|(mut pv, qv)| {
                izip!(subs.bit_rev.iter(), subs.power_bitrev.iter()).for_each(|(br, pr)| {
                    pv[*br] = qv[*pr];
                });
            });
        } else if poly.representation == Representation::Coefficient {
            let mut exponent = 0;
            let mask = self.degree - 1;
            for j in 0..self.degree {
                izip!(
                    poly.coefficients.slice(s![.., j]),
                    p.coefficients.slice_mut(s![.., mask & exponent]),
                    self.iter_moduli_ops()
                )
                .for_each(|(qxi, pxi, modqi)| {
                    if exponent & self.degree != 0 {
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
        q_poly: &Poly,
        t: &Modulus,
        b: usize,
        t_qhat_inv_modq_divq_modt: &[u64],
        t_bqhat_inv_modq_divq_modt: &[u64],
        t_qhat_inv_modq_divq_frac: &[f64],
        t_bqhat_inv_modq_divq_frac: &[f64],
    ) -> Vec<u64> {
        debug_assert!(q_poly.representation == Representation::Coefficient);

        let t_f64 = t.modulus().to_f64().unwrap();
        let t_inv = 1.0 / t_f64;

        let t_values = q_poly
            .coefficients
            .axis_iter(Axis(1))
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
        pq_poly: &Poly,
        out_context: &PolyContext<'_, T>,
        p_context: &PolyContext<'_, T>,
        q_context: &PolyContext<'_, T>,
        to_s_hat_inv_mods_divs_modo: &Array2<u64>,
        to_s_hat_inv_mods_divs_frachi: &[u64],
        to_s_hat_inv_mods_divs_fraclo: &[u64],
    ) -> Poly {
        debug_assert!(pq_poly.representation == Representation::Coefficient);
        let mut input_offset = 0;
        let mut output_offset = 0;
        let mut input_size = 0;
        let mut output_size = 0;
        if out_context == p_context {
            input_offset = p_context.moduli_count;
            input_size = q_context.moduli_count;
            output_size = p_context.moduli_count;
        } else {
            output_offset = p_context.moduli_count;
            input_size = p_context.moduli_count;
            output_size = q_context.moduli_count;
        }

        let degree = out_context.degree;
        let modos = out_context.moduli_ops();

        let mut o_coeffs = Array2::<u64>::uninit((output_size, degree));
        unsafe {
            for ri in (0..degree).step_by(8) {
                seq!(N in 0..8 {
                    let mut frac~N = U192::ZERO;
                });

                for i in 0..input_size {
                    let fhi = *to_s_hat_inv_mods_divs_frachi.get_unchecked(i) as u128;
                    let flo = *to_s_hat_inv_mods_divs_fraclo.get_unchecked(i) as u128;

                    seq!(N in 0..8 {
                        let xi = *pq_poly.coefficients.uget((i + input_offset, ri+N));
                        let lo = xi as u128 * flo;
                        let hi = xi as u128 * fhi + (lo >> 64);
                        frac~N = frac~N.wrapping_add(&U192::from_words([
                            lo as u64,
                            hi as u64,
                            (hi >> 64) as u64,
                        ]));
                    });
                }

                seq!(N in 0..8 {
                    let frac~N = frac~N.shr_vartime(127).as_words()[0] as u128;
                });

                for j in 0..output_size {
                    seq!(N in 0..8 {
                        let mut tmp~N = frac~N;
                    });

                    for i in 0..input_size {
                        let op = *to_s_hat_inv_mods_divs_modo.uget((j, i)) as u128;

                        seq!(N in 0..8 {
                            tmp~N += *pq_poly.coefficients.uget((i + input_offset, ri+N)) as u128 * op;
                        });
                    }

                    let modoj = modos.get_unchecked(j);

                    let op = *to_s_hat_inv_mods_divs_modo.uget((j, input_size)) as u128;

                    seq!(N in 0..8 {
                        tmp~N += *pq_poly.coefficients.uget((j + output_offset, ri+N)) as u128 * op;
                        let pxj = modoj.barret_reduction_u128(tmp~N);
                        o_coeffs.uget_mut((j, ri+N)).write(pxj);
                    });
                }
            }
        }

        unsafe {
            let o_coeffs = o_coeffs.assume_init();
            return self.new(o_coeffs, Representation::Coefficient);
        }
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
        q_poly: &Poly,
        p_context: &PolyContext<'_, T>,
        neg_pq_hat_inv_modq: &[u64],
        neg_pq_hat_inv_modq_shoup: &[u64],
        q_inv: &[f64],
        q_inv_modp: &Array2<u64>,
    ) -> Poly {
        debug_assert!(q_poly.representation == Representation::Coefficient);

        let q_size = self.moduli_count;
        let p_size = p_context.moduli_count;
        let degree = self.degree;

        let modqs = self.moduli_ops();
        let modps = p_context.moduli_ops();

        let mut p_coeffs = Array2::<u64>::uninit((p_size, degree));
        unsafe {
            //
            for ri in (0..degree).step_by(8) {
                let mut xiv = Vec::with_capacity(q_size * 8);
                let uninit_xiv = xiv.spare_capacity_mut();

                seq!(N in 0..8 {
                    let mut nu~N = 0.5f64;
                });

                for i in 0..q_size {
                    let modqi = modqs.get_unchecked(i);
                    let op = *neg_pq_hat_inv_modq.get_unchecked(i);
                    let op_shoup = *neg_pq_hat_inv_modq_shoup.get_unchecked(i);
                    let qi_inv = q_inv.get_unchecked(i);
                    seq!(N in 0..8 {
                        let tmp~N = modqi.mul_mod_shoup(*q_poly.coefficients.uget((i, ri+N)), op, op_shoup);
                        nu~N += tmp~N as f64 * qi_inv;
                        uninit_xiv.get_unchecked_mut(i*8+N).write(tmp~N);
                    });
                }

                xiv.set_len(q_size * 8);
                seq!(N in 0..8 {
                    let nu~N = nu~N as u64;
                });

                for j in 0..p_size {
                    seq!(N in 0..8 {
                        let mut tmp~N = 0u128;
                    });
                    for i in 0..q_size {
                        let op = *q_inv_modp.uget((j, i)) as u128;

                        seq!(N in 0..8 {
                            tmp~N += *xiv.get_unchecked(i * 8 + N) as u128 * op;
                        });
                    }

                    let modpj = modps.get_unchecked(j);
                    seq!(N in 0..8 {
                        let pxj = p_coeffs.uget_mut((j, ri+N)).write(modpj.barret_reduction_u128(tmp~N));
                        *pxj = modpj.sub_mod_fast(*pxj, nu~N);
                    });
                }
            }
        }

        unsafe {
            let p_coeffs = p_coeffs.assume_init();
            return p_context.new(p_coeffs, Representation::Coefficient);
        }
    }

    pub fn switch_crt_basis(
        &self,
        q_poly: &Poly,
        p_context: &PolyContext<'_, T>,
        q_hat_modp: &Array2<u64>,
        q_hat_inv_modq: &[u64],
        q_hat_inv_modq_shoup: &[u64],
        q_inv: &[f64],
        alpha_modp: &Array2<u64>,
    ) -> Poly {
        debug_assert!(q_poly.representation == Representation::Coefficient);

        let q_size = self.moduli_count;
        let p_size = p_context.moduli_count;
        let degree = self.degree;

        let modq_ops = self.moduli_ops();
        let modp_ops = p_context.moduli_ops();

        let mut p_coeffs = Array2::uninit((p_size, degree));

        unsafe {
            for ri in (0..degree).step_by(8) {
                let mut xiq = Vec::with_capacity(q_size * 8);
                let uninit = xiq.spare_capacity_mut();

                seq!(N in 0..8{
                    // let mut xiq~N = Vec::with_capacity(q_size);
                    let mut nu~N = 0.5f64;
                });

                for i in 0..q_size {
                    let mod_ref = modq_ops.get_unchecked(i);
                    let op = q_hat_inv_modq.get_unchecked(i);
                    let op_shoup = q_hat_inv_modq_shoup.get_unchecked(i);
                    let q_invi = q_inv.get_unchecked(i);
                    seq!(N in 0..8{
                        let tmp~N = mod_ref.mul_mod_shoup(
                            *q_poly.coefficients.uget((i,ri + N)),
                            *op,
                            *op_shoup
                        );
                        nu~N += tmp~N as f64 * *q_invi;
                        uninit.get_unchecked_mut(i*8+N).write(tmp~N);
                    });
                }

                xiq.set_len(q_size * 8);

                for j in 0..p_size {
                    // Why not set `tmp` as a vec of u128? Apparently calling `drop_in_place` afterwards on
                    // `tmp` if it were a vec of u128s is more expensive than using tmp as 8 different variables.
                    seq!(N in 0..8{
                        let mut tmp~N = 0u128;
                    });

                    for i in 0..q_size {
                        let op2 = *q_hat_modp.uget((j, i)) as u128;

                        seq!(N in 0..8 {
                            tmp~N += *xiq.get_unchecked(i * 8 + N) as u128 * op2;

                        });
                    }

                    let modpj = modp_ops.get_unchecked(j);

                    seq!(N in 0..8 {
                        let tmp = modpj.sub_mod_fast(modpj.barret_reduction_u128(tmp~N), *alpha_modp.uget((j,nu~N as usize)));
                        p_coeffs.uget_mut((j, ri + N)).write(tmp);
                    });
                }
            }
        }

        unsafe {
            let p_coeffs = p_coeffs.assume_init();
            return p_context.new(p_coeffs, Representation::Coefficient);
        }
    }

    pub fn fast_expand_crt_basis_p_over_q(
        &self,
        q_poly: &Poly,
        p_context: &PolyContext<'_, T>,
        pq_context: &PolyContext<'_, T>,
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
        let p = if q_poly.representation == Representation::Coefficient {
            self.fast_conv_p_over_q(
                q_poly,
                p_context,
                neg_pq_hat_inv_modq,
                neg_pq_hat_inv_modq_shoup,
                q_inv,
                q_inv_modp,
            )
        } else {
            let mut q = q_poly.clone();
            self.change_representation(&mut q, Representation::Coefficient);
            self.fast_conv_p_over_q(
                &q,
                p_context,
                neg_pq_hat_inv_modq,
                neg_pq_hat_inv_modq_shoup,
                q_inv,
                q_inv_modp,
            )
        };

        // switch p to q
        let q = p_context.switch_crt_basis(
            &p,
            &self,
            p_hat_modq,
            p_hat_inv_modp,
            p_hat_inv_modp_shoup,
            p_inv,
            alpha_modq,
        );

        let p_size = p_context.moduli_count;
        // output should always be in coefficient form
        let mut pq_coeffs = Array2::zeros((pq_context.moduli_count, pq_context.degree));
        pq_coeffs
            .slice_mut(s![..p_size, ..])
            .assign(&p.coefficients);
        pq_coeffs
            .slice_mut(s![p_size.., ..])
            .assign(&q.coefficients);

        pq_context.new(pq_coeffs, Representation::Coefficient)
    }

    pub fn expand_crt_basis(
        &self,
        q_poly: &Poly,
        pq_context: &PolyContext<'_, T>,
        p_context: &PolyContext<'_, T>,
        q_hat_modp: &Array2<u64>,
        q_hat_inv_modq: &[u64],
        q_hat_inv_modq_shoup: &[u64],
        q_inv: &[f64],
        alpha_modp: &Array2<u64>,
    ) -> Poly {
        let mut p = if q_poly.representation == Representation::Coefficient {
            self.switch_crt_basis(
                q_poly,
                p_context,
                q_hat_modp,
                q_hat_inv_modq,
                q_hat_inv_modq_shoup,
                q_inv,
                alpha_modp,
            )
        } else {
            let mut q = q_poly.clone();
            self.change_representation(&mut q, Representation::Coefficient);
            self.switch_crt_basis(
                &q,
                p_context,
                q_hat_modp,
                q_hat_inv_modq,
                q_hat_inv_modq_shoup,
                q_inv,
                alpha_modp,
            )
        };
        p_context.change_representation(&mut p, q_poly.representation.clone());

        let p_size = p_context.moduli_count;
        let mut pq_coeffs = Array2::zeros((pq_context.moduli_count, pq_context.degree));
        pq_coeffs
            .slice_mut(s![..p_size, ..])
            .assign(&p.coefficients);
        pq_coeffs
            .slice_mut(s![p_size.., ..])
            .assign(&q_poly.coefficients);

        pq_context.new(pq_coeffs, q_poly.representation.clone())
    }

    /// Switches CRT basis from Q to P approximately.
    ///
    /// Note: the result is approximate since overflow is ignored.
    pub fn approx_switch_crt_basis(
        q_coefficients: &ArrayView2<u64>,
        q_moduli_ops: &[Modulus],
        degree: usize,
        q_hat_inv_modq: &[u64],
        q_hat_modp: &Array2<u64>,
        p_moduli_ops: &[Modulus],
    ) -> Array2<u64> {
        debug_assert!(q_moduli_ops.len() == q_coefficients.shape()[0]);
        debug_assert!(q_moduli_ops.len() <= 3);

        let mut p_coeffs = Array2::<u64>::uninit((p_moduli_ops.len(), degree));

        let p_size = p_moduli_ops.len();
        let q_size = q_coefficients.shape()[0];
        unsafe {
            for ri in (0..degree).step_by(8) {
                let mut tmp: [MaybeUninit<u64>; 3 * 8] = MaybeUninit::uninit().assume_init();

                for i in 0..q_size {
                    let modq = q_moduli_ops.get_unchecked(i);
                    let op = *q_hat_inv_modq.get_unchecked(i);

                    seq!(N in 0..8 {
                        tmp.get_unchecked_mut(i*8+N)
                        .write(modq.mul_mod_fast(*q_coefficients.uget((i, ri+N)), op));
                    });
                }
                // tmp.set_len(q_size);

                let tmp = mem::transmute::<_, [u64; 3 * 8]>(tmp);
                for j in 0..p_size {
                    seq!(N in 0..8 {
                        let mut s~N = 0u128;
                    });

                    for i in 0..q_size {
                        let op = *q_hat_modp.uget((j, i)) as u128;
                        seq!(N in 0..8 {
                            s~N += *tmp.get_unchecked(i*8+N) as u128 * op;
                        });
                    }

                    let modpj = p_moduli_ops.get_unchecked(j);
                    seq!(N in 0..8 {
                        p_coeffs
                        .uget_mut((j, ri+N))
                        .write(modpj.barret_reduction_u128(s~N));
                    });
                }
            }
        }

        unsafe {
            return p_coeffs.assume_init();
        }
    }

    /// Approx mod down
    ///
    /// Switches modulus from QP to Q and divides the result by P.
    /// Uses approx mod switch to switch from P to Q resulting in additional uP. However,
    /// we get rid uP by dividing the final value by P.
    pub fn approx_mod_down(
        &self,
        qp_poly: &mut Poly,
        q_context: &PolyContext<'_, T>,
        p_context: &PolyContext<'_, T>,
        p_hat_inv_modp: &[u64],
        p_hat_modq: &Array2<u64>,
        p_inv_modq: &[u64],
    ) {
        debug_assert!(q_context.moduli_count + p_context.moduli_count == self.moduli_count);
        debug_assert!(qp_poly.representation == Representation::Evaluation);

        let q_size = q_context.moduli_count;

        // Change P part of QP from `Evaluation` to `Coefficient` representation
        let mut p_coefficients = qp_poly.coefficients.slice_mut(s![q_size.., ..]);
        debug_assert!(p_coefficients.shape()[0] == p_context.moduli_count);
        izip!(p_coefficients.outer_iter_mut(), p_context.iter_ntt_ops()).for_each(
            |(mut v, ntt_op)| {
                ntt_op.backward(v.as_slice_mut().unwrap());
            },
        );

        let mut p_to_q_coefficients = PolyContext::<T>::approx_switch_crt_basis(
            &p_coefficients.view(),
            p_context.moduli_ops(),
            self.degree,
            p_hat_inv_modp,
            p_hat_modq,
            q_context.moduli_ops(),
        );

        // Change P switched to Q part from `Coefficient` to `Evaluation` representation
        // Reason to switch from coefficient to evaluation form becomes apparent in next step when we multiply all values by 1/P
        izip!(
            p_to_q_coefficients.outer_iter_mut(),
            q_context.iter_ntt_ops()
        )
        .for_each(|(mut v, ntt_op)| {
            ntt_op.forward(v.as_slice_mut().unwrap());
        });

        qp_poly.coefficients.slice_collapse(s![..q_size, ..]);
        debug_assert!(qp_poly.coefficients.shape()[0] == q_size);

        izip!(
            qp_poly.coefficients.outer_iter_mut(),
            p_to_q_coefficients.outer_iter(),
            q_context.iter_moduli_ops(),
            p_inv_modq.iter(),
        )
        .for_each(|(mut v, switched_v, modqi, p_inv_modqi)| {
            modqi.sub_mod_fast_vec(v.as_slice_mut().unwrap(), switched_v.as_slice().unwrap());
            modqi.scalar_mul_mod_fast_vec(v.as_slice_mut().unwrap(), *p_inv_modqi);
        });
    }

    /// Switches polynomial from Q to Q' and scales by 1/qn where Q = q0*q1*q2...*qn and Q' = q0*q1*q2...*q(n-1).
    ///
    /// Works for both coefficient and evaluation representation, but latter is expensive since you need to pay for
    /// 1 + (n-1) NTT ops.
    pub fn mod_down_next(&self, poly: &mut Poly, last_qi_inv_modq: &[u64]) {
        let (mut coeffs, mut p) = poly
            .coefficients
            .view_mut()
            .split_at(Axis(0), self.moduli_count - 1);

        if poly.representation == Representation::Evaluation {
            // TODO: can we change this to lazy?
            self.ntt_ops()
                .last()
                .unwrap()
                .backward(p.as_slice_mut().unwrap());
        }

        izip!(
            coeffs.outer_iter_mut(),
            self.iter_moduli_ops(),
            self.iter_ntt_ops(),
            last_qi_inv_modq.into_producer()
        )
        .for_each(|(mut ceoffs, modqi, nttop, last_qi_modqi)| {
            let mut tmp = p.to_owned();
            modqi.reduce_vec(tmp.as_slice_mut().unwrap());

            if poly.representation == Representation::Evaluation {
                //TODO can we make this lazy as well?
                nttop.forward(tmp.as_slice_mut().unwrap());
            }

            modqi.sub_mod_fast_vec(ceoffs.as_slice_mut().unwrap(), tmp.as_slice().unwrap());
            modqi.scalar_mul_mod_fast_vec(ceoffs.as_slice_mut().unwrap(), *last_qi_modqi);
        });
        poly.coefficients.slice_collapse(s![..-1, ..]);
    }

    pub fn fma_reverse_inplace(&self, p0: &mut Poly, p1: &Poly, p2: &Poly) {
        izip!(
            p0.coefficients.outer_iter_mut(),
            p1.coefficients.outer_iter(),
            p2.coefficients.outer_iter(),
            self.iter_moduli_ops()
        )
        .for_each(|(mut a, b, c, modqi)| {
            modqi.fma_reverse_vec(
                a.as_slice_mut().unwrap(),
                b.as_slice().unwrap(),
                c.as_slice().unwrap(),
            )
        });
    }

    pub fn fma_reverse(&self, p0: &Poly, p1: &Poly, p2: &Poly) -> Poly {
        let mut p = p0.clone();
        self.fma_reverse_inplace(&mut p, p1, p2);
        p
    }

    /// Subtract self from a
    pub fn sub_reversed_inplace(&self, p0: &mut Poly, p1: &Poly) {
        debug_assert!(p0.representation == p1.representation);
        izip!(
            p0.coefficients.outer_iter_mut(),
            p1.coefficients.outer_iter(),
            self.iter_moduli_ops()
        )
        .for_each(|(mut a, b, modqi)| {
            modqi.sub_mod_fast_vec_reversed(a.as_slice_mut().unwrap(), b.as_slice().unwrap());
        });
    }

    pub fn neg_assign(&self, p0: &mut Poly) {
        izip!(p0.coefficients.outer_iter_mut(), self.iter_moduli_ops()).for_each(
            |(mut coeffs, modqi)| {
                modqi.neg_mod_fast_vec(coeffs.as_slice_mut().unwrap());
            },
        );
    }

    pub fn neg(self, p0: Poly) -> Poly {
        let mut p0 = p0.clone();
        self.neg_assign(&mut p0);
        p0
    }

    pub fn add_assign(&self, lhs: &mut Poly, rhs: &Poly) {
        debug_assert!(lhs.representation == rhs.representation);

        izip!(
            lhs.coefficients.outer_iter_mut(),
            rhs.coefficients.outer_iter(),
            self.iter_moduli_ops()
        )
        .for_each(|(mut p1, p2, q)| {
            q.add_mod_fast_vec(p1.as_slice_mut().unwrap(), p2.as_slice().unwrap())
        });
    }

    pub fn add(&self, lhs: &Poly, rhs: &Poly) -> Poly {
        let mut lhs = lhs.clone();
        self.add_assign(&mut lhs, rhs);
        lhs
    }

    pub fn sub_assign(&self, lhs: &mut Poly, rhs: &Poly) {
        debug_assert!(lhs.representation == rhs.representation);
        izip!(
            lhs.coefficients.outer_iter_mut(),
            rhs.coefficients.outer_iter(),
            self.iter_moduli_ops()
        )
        .for_each(|(mut p1, p2, q)| {
            q.sub_mod_fast_vec(p1.as_slice_mut().unwrap(), p2.as_slice().unwrap())
        });
    }

    pub fn sub(&self, lhs: &Poly, rhs: &Poly) -> Poly {
        let mut lhs = lhs.clone();
        self.sub_assign(&mut lhs, rhs);
        lhs
    }

    pub fn mul_assign(&self, lhs: &mut Poly, rhs: &Poly) {
        assert!(lhs.representation == rhs.representation);
        assert!(lhs.representation == Representation::Evaluation);

        izip!(
            lhs.coefficients.outer_iter_mut(),
            rhs.coefficients.outer_iter(),
            self.iter_moduli_ops()
        )
        .for_each(|(mut p, p2, modqi)| {
            // let now = std::time::Instant::now();
            modqi.mul_mod_fast_vec(p.as_slice_mut().unwrap(), p2.as_slice().unwrap());
            // println!("Time: {:?}", now.elapsed());
        });
    }

    pub fn mul(&self, lhs: &Poly, rhs: &Poly) -> Poly {
        let mut lhs = lhs.clone();
        self.mul_assign(&mut lhs, rhs);
        lhs
    }
}

impl<T> PolyContext<'_, T>
where
    T: Ntt,
{
    pub fn try_convert_from_u64(&self, values: &[u64], representation: Representation) -> Poly {
        assert!(values.len() == self.degree);
        let mut p = self.zero(representation);
        izip!(p.coefficients.outer_iter_mut(), self.iter_moduli_ops()).for_each(
            |(mut qi_values, qi)| {
                let mut xi = values.to_vec();
                qi.reduce_vec(&mut xi);
                qi_values.as_slice_mut().unwrap().copy_from_slice(&xi);
            },
        );
        p
    }

    /// Constructs a polynomial with given BigUint values. It simply reduces each BigUint coefficient by every modulus in poly_context and assumes the specified representation.
    ///
    /// values length should be smaller than or equal to poly_context degree. Values at index beyond polynomial degree are ignored.
    pub fn try_convert_from_biguint(
        &self,
        values: &[BigUint],
        representation: Representation,
    ) -> Poly {
        debug_assert!(self.degree >= values.len());
        let mut poly = self.zero(representation);

        izip!(values.iter(), poly.coefficients.axis_iter_mut(Axis(1))).for_each(
            |(v, mut rests)| {
                izip!(rests.iter_mut(), self.iter_moduli_ops()).for_each(|(xi, qi)| {
                    *xi = (v % qi.modulus()).to_u64().unwrap();
                })
            },
        );

        poly
    }

    /// Constructs a polynomial with given i64 values with small bound and assumes the given representation.  
    ///
    /// Panics if length of values is not equal to polynomial degree
    pub fn try_convert_from_i64_small(
        &self,
        values: &[i64],
        representation: Representation,
    ) -> Poly {
        assert!(values.len() == self.degree);
        let mut p = self.zero(representation);
        izip!(p.coefficients.outer_iter_mut(), self.iter_moduli_ops()).for_each(
            |(mut qi_values, qi)| {
                qi_values
                    .as_slice_mut()
                    .unwrap()
                    .copy_from_slice(&qi.reduce_vec_i64_small(values).as_slice());
            },
        );
        p
    }

    pub fn try_convert_to_biguint(&self, p: &Poly) -> Vec<BigUint> {
        assert!(p.representation == Representation::Coefficient);

        let big_q = self.big_q();
        let mut q_hat = vec![];
        let mut q_hat_inv = vec![];
        self.iter_moduli_ops().for_each(|modqi| {
            let qi = modqi.modulus();
            let qi_hat = &big_q / qi;
            let qi_hat_inv = mod_inverse_biguint_u64(&qi_hat, qi);
            q_hat.push(qi_hat);
            q_hat_inv.push(qi_hat_inv);
        });

        let mut values = vec![];
        p.coefficients.axis_iter(Axis(1)).for_each(|rests| {
            let mut v = BigUint::zero();
            izip!(
                rests.iter(),
                q_hat.iter(),
                q_hat_inv.iter(),
                self.iter_moduli_ops()
            )
            .for_each(|(xi, qi_hat, qi_hat_inv, modqi)| {
                v += ((xi * qi_hat_inv) % modqi.modulus()) * qi_hat;
            });
            values.push(v % &big_q);
        });
        values
    }
}

mod tests {
    use super::*;
    use crate::{nb_theory::generate_primes_vec, parameters::BfvParameters, PolyType};
    use fhe_math::zq::ntt::NttOperator;
    use num_bigint::{BigInt, ToBigInt};
    use num_bigint_dig::UniformBigUint;
    use num_traits::{FromPrimitive, Zero};
    use rand::{distributions::Uniform, rngs::ThreadRng, thread_rng, Rng};

    #[test]
    pub fn test_poly_to_biguint() {
        let rng = thread_rng();
        let values = rng
            .sample_iter(Uniform::new(0u128, 1 << 127))
            .take(1 << 4)
            .map(BigUint::from)
            .collect_vec();

        let params = BfvParameters::default(10, 1 << 4);
        let poly_ctx = params.poly_ctx(&PolyType::Q, 0);
        let q_poly = poly_ctx.try_convert_from_biguint(&values, Representation::Coefficient);

        assert_eq!(values, poly_ctx.try_convert_to_biguint(&q_poly));
    }

    #[test]
    fn substitution_works() {
        let params = BfvParameters::default(1, 1 << 3);

        let ctx = params.poly_ctx(&PolyType::Q, 0);
        let mut rng = thread_rng();

        let p = ctx.random(Representation::Coefficient, &mut rng);
        let mut p_ntt = p.clone();
        ctx.change_representation(&mut p_ntt, Representation::Evaluation);

        // substitution by 1 should not change p
        let subs = Substitution::new(1, params.degree);
        assert_eq!(p, ctx.substitute(&p, &subs));
        assert_eq!(p_ntt, ctx.substitute(&p_ntt, &subs));

        for exp in [3, 5, 9] {
            let subs = Substitution::new(exp, params.degree);

            let p = ctx.random(Representation::Coefficient, &mut rng);
            let mut p_ntt = p.clone();
            ctx.change_representation(&mut p_ntt, Representation::Evaluation);

            let p_subs = ctx.substitute(&p, &subs);
            // substitue using biguints
            let p_biguint = ctx.try_convert_to_biguint(&p);
            let mut p_biguint_subs = vec![BigUint::zero(); params.degree];
            p_biguint.iter().enumerate().for_each(|(i, v)| {
                let wraps = (i * subs.exponent) / params.degree;
                let index = (i * subs.exponent) % params.degree;
                if (wraps & 1 == 1) {
                    p_biguint_subs[index] += (ctx.big_q() - v);
                } else {
                    p_biguint_subs[index] += v;
                }
            });
            assert_eq!(p_biguint_subs, ctx.try_convert_to_biguint(&p_subs));

            // check subs in evaluation form
            let p_subs_ntt = ctx.substitute(&p_ntt, &subs);
            let mut p_subs_ntt_clone = p_subs_ntt.clone();
            ctx.change_representation(&mut p_subs_ntt_clone, Representation::Coefficient);
            assert_eq!(p_subs, p_subs_ntt_clone);

            // substitution by [exp^-1]_(2*degree) on polynomial that was substitution bu exp
            // must result to original polynomial
            let exp_inv = Modulus::new((2 * ctx.degree) as u64).inv(exp as u64) as usize;
            let inv_subs = Substitution::new(exp_inv, params.degree);
            assert_eq!(p, ctx.substitute(&p_subs, &inv_subs));
            assert_eq!(p_ntt, ctx.substitute(&p_subs_ntt, &inv_subs));
        }
    }

    #[test]
    // FIXME: fails in debug mode. Check fn to see why.
    fn test_scale_and_round_decryption() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(10, 1 << 3);

        let top_context = params.poly_ctx(&PolyType::Q, 0);
        let mut q_poly = top_context.random(Representation::Coefficient, &mut rng);

        // let's scale q_poly by t/Q and switch its context from Q to t.
        let t_coeffs = top_context.scale_and_round_decryption(
            &q_poly,
            &Modulus::new(params.plaintext_modulus),
            params.max_bit_size_by2,
            &params.t_ql_hat_inv_modql_divql_modt[0],
            &params.t_bql_hat_inv_modql_divql_modt[0],
            &params.t_ql_hat_inv_modql_divql_frac[0],
            &params.t_bql_hat_inv_modql_divql_frac[0],
        );

        let q = top_context.big_q();
        let t = params.plaintext_modulus;
        let t_expected = izip!(top_context.try_convert_to_biguint(&q_poly))
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
        let params = BfvParameters::default(10, 1 << 4);

        let q_context = params.poly_ctx(&PolyType::Q, 0);
        let p_context = params.poly_ctx(&PolyType::P, 0);
        let mut q_poly = q_context.random(Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let p_poly = q_context.fast_conv_p_over_q(
            &q_poly,
            &p_context,
            &params.neg_pql_hat_inv_modql[0],
            &params.neg_pql_hat_inv_modql_shoup[0],
            &params.ql_inv[0],
            &params.ql_inv_modpl[0],
        );
        println!("time: {:?}", now.elapsed());

        let q = q_context.big_q();
        let p = p_context.big_q();
        let p_expected: Vec<BigUint> = q_context
            .try_convert_to_biguint(&q_poly)
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

        assert_eq!(p_context.try_convert_to_biguint(&p_poly), p_expected);
    }

    #[test]
    pub fn test_switch_crt_basis() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(10, 1 << 4);

        let q_context = params.poly_ctx(&PolyType::Q, 0);
        let p_context = params.poly_ctx(&PolyType::P, 0);
        let mut q_poly = q_context.random(Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let p_poly = q_context.switch_crt_basis(
            &q_poly,
            &p_context,
            &params.ql_hat_modpl[0],
            &params.ql_hat_inv_modql[0],
            &params.ql_hat_inv_modql_shoup[0],
            &params.ql_inv[0],
            &params.alphal_modpl[0],
        );
        println!("time: {:?}", now.elapsed());

        let q = q_context.big_q();
        let p = p_context.big_q();
        let p_expected: Vec<BigUint> = q_context
            .try_convert_to_biguint(&q_poly)
            .iter()
            .map(|xi| {
                if xi >= &(&q >> 1) {
                    &p - ((&q - xi) % &p)
                } else {
                    xi % &p
                }
            })
            .collect();

        assert_eq!(p_expected, p_context.try_convert_to_biguint(&p_poly));
    }

    #[test]
    pub fn test_fast_expand_crt_basis_p_over_q() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(10, 1 << 4);

        let q_context = params.poly_ctx(&PolyType::Q, 0);
        let p_context = params.poly_ctx(&PolyType::P, 0);
        let pq_context = params.poly_ctx(&PolyType::PQ, 0);
        let mut q_poly = q_context.random(Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let pq_poly = q_context.fast_expand_crt_basis_p_over_q(
            &q_poly,
            &p_context,
            &pq_context,
            &params.neg_pql_hat_inv_modql[0],
            &params.neg_pql_hat_inv_modql_shoup[0],
            &params.ql_inv[0],
            &params.ql_inv_modpl[0],
            &params.pl_hat_modql[0],
            &params.pl_hat_inv_modpl[0],
            &params.pl_hat_inv_modpl_shoup[0],
            &params.pl_inv[0],
            &params.alphal_modql[0],
        );
        println!("time: {:?}", now.elapsed());

        let q = q_context.big_q();
        let p = p_context.big_q();
        let pq = pq_context.big_q();
        let pq_expected: Vec<BigUint> = q_context
            .try_convert_to_biguint(&q_poly)
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

        assert_eq!(pq_context.try_convert_to_biguint(&pq_poly), pq_expected);
    }

    #[test]
    pub fn test_scale_and_round() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(10, 1 << 4);

        let q_context = params.poly_ctx(&PolyType::Q, 0);
        let p_context = params.poly_ctx(&PolyType::P, 0);
        let pq_context = params.poly_ctx(&PolyType::PQ, 0);

        let pq_poly = pq_context.random(Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let q_poly = pq_context.scale_and_round(
            &pq_poly,
            &q_context,
            &p_context,
            &q_context,
            &params.tql_pl_hat_inv_modpl_divpl_modql[0],
            &params.tql_pl_hat_inv_modpl_divpl_frachi[0],
            &params.tql_pl_hat_inv_modpl_divpl_fraclo[0],
        );
        println!("time1: {:?}", now.elapsed());

        let t = params.plaintext_modulus;
        let p = p_context.big_q();
        let q = q_context.big_q();
        let pq = pq_context.big_q();
        let q_expected: Vec<BigUint> = pq_context
            .try_convert_to_biguint(&pq_poly)
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

        izip!(q_context.try_convert_to_biguint(&q_poly), q_expected.iter()).for_each(
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
        let params = BfvParameters::default(6, 1 << 4);

        let q_context = params.poly_ctx(&PolyType::Q, 3);
        let p_context = params.poly_ctx(&PolyType::P, 0);

        // Pre-computation
        let mut q_hat_inv_modq = vec![];
        let mut q_hat_modp = vec![];
        let q = q_context.big_q();
        izip!(q_context.iter_moduli_ops()).for_each(|(modqi)| {
            q_hat_inv_modq.push(
                mod_inverse_biguint_u64(&(&q / modqi.modulus()), modqi.modulus())
                    .to_u64()
                    .unwrap(),
            );
        });
        izip!(p_context.iter_moduli_ops()).for_each(|modpj| {
            izip!(q_context.iter_moduli_ops()).for_each(|(modqi)| {
                q_hat_modp.push(((&q / modqi.modulus()) % modpj.modulus()).to_u64().unwrap());
            });
        });
        let q_hat_modp = Array2::<u64>::from_shape_vec(
            (p_context.moduli_count, q_context.moduli_count),
            q_hat_modp,
        )
        .unwrap();

        let q_poly = q_context.random(Representation::Coefficient, &mut rng);
        let now = std::time::Instant::now();
        let p_coefficients = PolyContext::<NttOperator>::approx_switch_crt_basis(
            &q_poly.coefficients.view(),
            q_context.moduli_ops(),
            q_context.degree,
            &q_hat_inv_modq,
            &q_hat_modp,
            p_context.moduli_ops(),
        );
        let mut p_poly = p_context.new(p_coefficients, Representation::Coefficient);
        println!("time: {:?}", now.elapsed());

        // dbg!(&p_poly.coefficients);

        let q = q_context.big_q();
        let p = p_context.big_q();

        let p_expected: Vec<BigUint> = q_context
            .try_convert_to_biguint(&q_poly)
            .iter()
            .map(|xi| {
                if xi > &(&q >> 1) {
                    &p - ((&q - xi) % &p)
                } else {
                    xi % &p
                }
            })
            .collect_vec();

        izip!(
            p_context.try_convert_to_biguint(&p_poly).iter(),
            p_expected.iter()
        )
        .for_each(|(r, e)| {
            let mut diff = r.to_bigint().unwrap() - e.to_bigint().unwrap();
            dbg!(r, e, diff.bits());
        })
    }

    #[test]
    pub fn test_approx_mod_down() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(6, 1 << 4);

        let q_context = params.poly_ctx(&PolyType::Q, 0);
        let p_context = params.poly_ctx(&PolyType::SpecialP, 0);
        let qp_context = params.poly_ctx(&PolyType::QP, 0);

        let q_size = q_context.moduli_count;
        let p_size = p_context.moduli_count;
        let qp_size = qp_context.moduli_count;

        // Pre computation
        let p = p_context.big_q();
        let mut p_hat_inv_modp = vec![];
        let mut p_hat_modq = vec![];
        p_context.iter_moduli_ops().for_each(|(modpi)| {
            p_hat_inv_modp.push(
                mod_inverse_biguint_u64(&(&p / modpi.modulus()), modpi.modulus())
                    .to_u64()
                    .unwrap(),
            );
        });
        q_context.iter_moduli_ops().for_each(|modqj| {
            p_context.iter_moduli_ops().for_each(|(modpi)| {
                p_hat_modq.push(((&p / modpi.modulus()) % modqj.modulus()).to_u64().unwrap());
            });
        });
        let p_hat_modq =
            Array2::from_shape_vec((q_context.moduli_count, p_context.moduli_count), p_hat_modq)
                .unwrap();
        let mut p_inv_modq = vec![];
        q_context.iter_moduli_ops().for_each(|modqi| {
            p_inv_modq.push(
                mod_inverse_biguint_u64(&p, modqi.modulus())
                    .to_u64()
                    .unwrap(),
            );
        });

        let mut qp_poly = qp_context.random(Representation::Evaluation, &mut rng);
        let mut q_res = qp_poly.clone();

        let now = std::time::Instant::now();
        qp_context.approx_mod_down(
            &mut q_res,
            &q_context,
            &p_context,
            &p_hat_inv_modp,
            &p_hat_modq,
            &p_inv_modq,
        );
        println!("time: {:?}", now.elapsed());

        q_context.change_representation(&mut q_res, Representation::Coefficient);

        qp_context.change_representation(&mut qp_poly, Representation::Coefficient);

        let qp = qp_context.big_q();
        let q = q_context.big_q();
        let p = p_context.big_q();

        let q_expected: Vec<BigUint> = qp_context
            .try_convert_to_biguint(&qp_poly)
            .iter()
            .map(|xi| {
                if xi > &(&qp >> 1usize) {
                    if &qp & BigUint::one() == BigUint::zero() {
                        &q - (((&qp - xi) + ((&p >> 1) - 1usize)) / &p) % &q
                    } else {
                        &q - (((&qp - xi) + (&p >> 1)) / &p) % &q
                    }
                } else {
                    ((xi + (&p >> 1)) / &p) % &q
                }
            })
            .collect_vec();

        izip!(
            q_context.try_convert_to_biguint(&q_res).iter(),
            q_expected.iter()
        )
        .for_each(|(res, expected)| {
            let diff: BigInt = res.to_bigint().unwrap() - expected.to_bigint().unwrap();
            assert!(diff <= BigInt::one());
            // dbg!(diff);
        });
    }

    #[test]
    pub fn test_mod_down_next() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 4);
        let q_ctx = params.poly_ctx(&PolyType::Q, 0);
        let q_ctx_next = params.poly_ctx(&PolyType::Q, 1);

        let mut q_poly = q_ctx.random(Representation::Evaluation, &mut rng);

        let last_qi = q_ctx.moduli_ops().last().unwrap();

        let mut q_res_poly = q_poly.clone();
        q_ctx.mod_down_next(&mut q_res_poly, &params.lastq_inv_modql[0]);

        q_ctx_next.change_representation(&mut q_res_poly, Representation::Coefficient);
        q_ctx.change_representation(&mut q_poly, Representation::Coefficient);

        let q = q_ctx.big_q();
        let q_next = q_ctx_next.big_q();
        let p = last_qi.modulus();
        let q_expected: Vec<BigUint> = q_ctx
            .try_convert_to_biguint(&q_poly)
            .iter()
            .map(|xi| {
                if xi > &(&q >> 1usize) {
                    if &q_next & BigUint::one() == BigUint::zero() {
                        &q_next - (((&q - xi) + ((p >> 1) - 1u64)) / p) % &q_next
                    } else {
                        &q_next - (((&q - xi) + (p >> 1)) / p) % &q_next
                    }
                } else {
                    ((xi + (p >> 1)) / p) % &q_next
                }
            })
            .collect_vec();

        izip!(
            q_ctx_next.try_convert_to_biguint(&q_res_poly).iter(),
            q_expected.iter()
        )
        .for_each(|(res, expected)| {
            let diff: BigInt = res.to_bigint().unwrap() - expected.to_bigint().unwrap();
            assert!(diff <= BigInt::one());
            // dbg!(diff);
        });
    }
}
