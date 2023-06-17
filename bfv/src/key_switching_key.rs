use crate::modulus::Modulus;
use crate::BfvParameters;
use crate::{
    nb_theory::generate_prime,
    poly::{Poly, PolyContext, Representation},
    secret_key::SecretKey,
};
use crypto_bigint::rand_core::CryptoRngCore;
use fhe_math::zq::primes;
use itertools::{izip, Itertools};
use ndarray::{azip, par_azip, s, Array1, Array2, Array3, Axis, IntoNdProducer};
use num_bigint::{BigUint, ToBigInt};
use num_bigint_dig::BigUint as BigUintDig;
use num_bigint_dig::ModInverse;
use num_traits::{FromPrimitive, One, ToPrimitive};
use rand::{CryptoRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;
use traits::Ntt;

struct BVKeySwitchingKey<T: Ntt> {
    c0s: Box<[Poly<T>]>,
    c1s: Box<[Poly<T>]>,
    seed: <ChaCha8Rng as SeedableRng>::Seed,
    ciphertext_ctx: Arc<PolyContext<T>>,
    ksk_ctx: Arc<PolyContext<T>>,
}

impl<T> BVKeySwitchingKey<T>
where
    T: Ntt,
{
    pub fn new<R: CryptoRng + CryptoRngCore>(
        poly: &Poly<T>,
        sk: &SecretKey<T>,
        ciphertext_ctx: &Arc<PolyContext<T>>,
        rng: &mut R,
    ) -> BVKeySwitchingKey<T> {
        // check that ciphertext context has more than on moduli, otherwise key switching does not makes sense
        debug_assert!(ciphertext_ctx.moduli.len() > 1);

        let ksk_ctx = &poly.context;

        // c1s
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);
        let c1s = Self::generate_c1(ciphertext_ctx.moduli.len(), ksk_ctx, seed);
        let c0s = Self::generate_c0(ciphertext_ctx, ksk_ctx, poly, &c1s, sk, rng);

        BVKeySwitchingKey {
            c0s: c0s.into_boxed_slice(),
            c1s: c1s.into_boxed_slice(),
            seed,
            ciphertext_ctx: ciphertext_ctx.clone(),
            ksk_ctx: ksk_ctx.clone(),
        }
    }

    pub fn switch(&self, poly: &Poly<T>) -> Vec<Poly<T>> {
        debug_assert!(poly.context == self.ciphertext_ctx);
        debug_assert!(poly.representation == Representation::Coefficient);

        let mut p = Poly::try_convert_from_u64(
            poly.coefficients.slice(s![0, ..]).as_slice().unwrap(),
            &self.ksk_ctx,
            &Representation::Coefficient,
        );
        p.change_representation(Representation::Evaluation);
        let mut c1_out = &self.c1s[0] * &p;
        p *= &self.c0s[0];
        let mut c0_out = p;

        izip!(
            self.c0s.iter(),
            self.c1s.iter(),
            poly.coefficients.outer_iter()
        )
        .skip(1)
        .for_each(|(c0, c1, rests)| {
            let mut p = Poly::try_convert_from_u64(
                rests.as_slice().unwrap(),
                &self.ksk_ctx,
                &Representation::Coefficient,
            );
            p.change_representation(Representation::Evaluation);

            c1_out += &(c1 * &p);
            p *= c0;
            c0_out += &p;
        });

        vec![c0_out, c1_out]
    }

    pub fn generate_c1(
        count: usize,
        ksk_ctx: &Arc<PolyContext<T>>,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
    ) -> Vec<Poly<T>> {
        let mut rng = ChaCha8Rng::from_seed(seed);
        (0..count)
            .into_iter()
            .map(|_| Poly::random(ksk_ctx, &Representation::Evaluation, &mut rng))
            .collect_vec()
    }

    pub fn generate_c0<R: CryptoRng + CryptoRngCore>(
        ciphertext_ctx: &Arc<PolyContext<T>>,
        ksk_ctx: &Arc<PolyContext<T>>,
        poly: &Poly<T>,
        c1s: &[Poly<T>],
        sk: &SecretKey<T>,
        rng: &mut R,
    ) -> Vec<Poly<T>> {
        // encrypt g corresponding to every qi in ciphertext
        // make sure that you have enough c1s
        debug_assert!(ciphertext_ctx.moduli.len() == c1s.len());
        debug_assert!(poly.representation == Representation::Evaluation);

        let mut sk = Poly::try_convert_from_i64_small(
            &sk.coefficients,
            ksk_ctx,
            &Representation::Coefficient,
        );
        sk.change_representation(Representation::Evaluation);

        izip!(ciphertext_ctx.g.into_iter(), c1s.iter())
            .map(|(g, c1)| {
                let mut g = Poly::try_convert_from_biguint(
                    vec![g.clone(); ksk_ctx.degree].as_slice(),
                    ksk_ctx,
                    &Representation::Evaluation,
                );
                // m
                g *= poly;
                let mut e = Poly::random_gaussian(ksk_ctx, &Representation::Coefficient, 10, rng);
                e.change_representation(Representation::Evaluation);
                e += &g;
                e -= &(c1 * &sk);
                e
            })
            .collect_vec()
    }
}

pub struct HybridKeySwitchingKey<T: Ntt> {
    // ksk_ctx is q_ctx
    params: Arc<BfvParameters<T>>,
    ksk_ctx: Arc<PolyContext<T>>,
    qp_ctx: Arc<PolyContext<T>>,
    seed: <ChaCha8Rng as SeedableRng>::Seed,
    dnum: usize,

    // approx_switch_crt_basis //
    // I would want all these to be Array instead of simple vecs. But it's kind of tricky.
    // Array needs to a fixed shape. But since the last chunk of q_moduli may contain
    // less than alpha primes when q_moduli.len()%alpha isn't 0, we cannot gaurantee
    // the vector that we convert to Array will of desired shape. We can fix this using
    // dummy calues, but this increases the complexity and becomes harder to manage.
    qj_hat_inv_modqj_parts: Vec<Vec<u64>>,
    qj_moduli_parts: Vec<Vec<Modulus>>,
    qj_hat_modqpj_parts: Vec<Array2<u64>>,
    qpj_moduli_parts: Vec<Vec<Modulus>>,

    // approx_mod_down //
    p_hat_inv_modp: Array1<u64>,
    p_hat_modq: Array2<u64>,
    p_inv_modq: Array1<u64>,

    c0s: Box<[Poly<T>]>,
    c1s: Box<[Poly<T>]>,
}

impl<T> HybridKeySwitchingKey<T>
where
    T: Ntt,
{
    /// Warning: Ciphertext context needs to be as same as KeySwitching Context. This is not
    /// a limitation of hybrid key switching, instead a limitation of the way key switching is
    /// implemented here.
    /// Let's say ciphertext ctx = Q' and ksk ctx = Q. The extended ctx should be QP. To speed things
    /// up during `key_switch` operation, we assume Q == Q' because we extend poly from Qj to Q[..i*alpha] + Q[(i+1)*alpha..] + P.
    pub fn new<R: CryptoRng + CryptoRngCore>(
        poly: &Poly<T>,
        sk: &SecretKey<T>,
        ciphertext_ctx: &Arc<PolyContext<T>>,
        rng: &mut R,
    ) -> HybridKeySwitchingKey<T> {
        let params = sk.params.clone();
        let ksk_ctx = poly.context.clone();

        let alpha = params.alpha;
        let aux_bits = params.aux_bits;
        let mut qj = vec![];
        ciphertext_ctx
            .moduli
            .chunks(alpha)
            .for_each(|q_parts_moduli| {
                // Qj
                let mut qji = BigUint::one();
                q_parts_moduli.iter().for_each(|qi| {
                    qji *= *qi;
                });
                qj.push(qji);
            });

        // This is just for precaution. Ideally alpha*aux_bits must be greater than maxbits. But
        // having alpha*auxbits - maxbits greater than a few bits costs performance, since you end
        // up having an additional special prime for nothing. In our case, alpha and aux_bites are fixed
        // to 3 and 60. Hence, assuming that you have alpha consecutive ciphertext primes of size 40-60
        // (usually the case for lower levels) it must be the case that maxbits is around 90.
        let mut maxbits = qj[0].bits();
        qj.iter().skip(1).for_each(|q| {
            maxbits = std::cmp::max(maxbits, q.bits());
        });
        let ideal_special_prime_count = (maxbits as f64 / aux_bits as f64).ceil() as usize;
        debug_assert!(ideal_special_prime_count == params.alpha);

        // P is special prime
        let (p, p_dig) = {
            let mut p = BigUint::one();
            let mut p_dig = BigUintDig::one();
            params.special_moduli.iter().for_each(|pj| {
                p *= *pj;
                p_dig *= *pj;
            });
            (p, p_dig)
        };

        // Calculating g values + Pre-comp for switching Qj to QP.
        let q = ciphertext_ctx.modulus();
        let q_dig = ciphertext_ctx.modulus_dig();
        let q_moduli = ciphertext_ctx.moduli.clone();
        // g = P * Qj_hat * Qj_hat_inv_modQj
        let mut g = vec![];
        // FIXME: we use 2d Vec instead of Array2 because the last part may contain less than alpha qis.
        // But this isn't acceptable. Change this to Array2 and adjust for last part somehow.
        let mut qj_hat_inv_modqj_parts = vec![];
        let mut qj_hat_modqpj_parts = vec![];
        let mut qpj_moduli_parts = vec![];
        let mut qj_moduli_parts = vec![];

        // let mut q_moduli_ops = vec![];
        // q_moduli.iter().for_each(|qi| {
        //     q_moduli_ops.push(Arc::new(Modulus::new(*qi)));
        // });
        // let mut p_moduli_ops = vec![];
        // params.special_moduli.iter().for_each(|pi| {
        //     p_moduli_ops.push(Arc::new(Modulus::new(*pi)));
        // });

        let parts = (q_moduli.len() as f64 / alpha as f64).ceil() as usize;

        q_moduli
            .chunks(alpha)
            .enumerate()
            .for_each(|(chunk_index, qj_moduli)| {
                // Qj
                let mut qj = BigUint::one();
                let mut qj_dig = BigUintDig::one();

                qj_moduli.iter().for_each(|qi| {
                    qj *= *qi;
                    qj_dig *= *qi;
                });

                // Q/Qj
                let qj_hat = &q / &qj;

                // [(Q/Qj)^-1]_Qj
                let qj_hat_inv_modqj = BigUint::from_bytes_le(
                    &(&q_dig / &qj_dig)
                        .mod_inverse(&qj_dig)
                        .unwrap()
                        .to_biguint()
                        .unwrap()
                        .to_bytes_le(),
                );
                g.push(&p * qj_hat * qj_hat_inv_modqj);

                // precomp approx_switch_crt_basis
                let mut qj_hat_inv_modqj = vec![];
                qj_moduli.iter().for_each(|qji| {
                    let qji_hat_inv_modqji = (&qj_dig / *qji)
                        .mod_inverse(BigUintDig::from_u64(*qji).unwrap())
                        .unwrap()
                        .to_biguint()
                        .unwrap()
                        .to_u64()
                        .unwrap();
                    qj_hat_inv_modqj.push(qji_hat_inv_modqji);
                });
                qj_hat_inv_modqj_parts.push(qj_hat_inv_modqj);
                // let p_start = q_moduli[..alpha * chunk_index].to_vec();
                // let p_mid = {
                //     if (alpha * (chunk_index + 1)) < q_moduli.len() {
                //         q_moduli[(alpha * (chunk_index + 1))..].to_vec()
                //     } else {
                //         vec![]
                //     }
                // };
                qj_moduli_parts.push(
                    q_moduli[alpha * chunk_index
                        ..std::cmp::min(q_moduli.len(), alpha * (chunk_index + 1))]
                        .iter()
                        .map(|qi| Modulus::new(*qi))
                        .collect_vec(),
                );

                let qpj_moduli = [
                    q_moduli[..alpha * chunk_index].to_vec(), // qis before chunk's qjs
                    q_moduli[std::cmp::min(q_moduli.len(), alpha * (chunk_index + 1))..].to_vec(), // qis after chunk's qjs
                    params.special_moduli.clone(),
                ]
                .concat();

                // let p_whole = [p_start, p_mid, p_moduli.clone()].concat();

                let mut qj_hat_modqpj = vec![];
                qj_moduli.iter().for_each(|qji| {
                    qpj_moduli.iter().for_each(|modpj| {
                        qj_hat_modqpj.push(((&qj / qji) % modpj).to_u64().unwrap());
                    });
                });
                let qj_hat_modqpj = Array2::<u64>::from_shape_vec(
                    (qj_moduli.len(), qpj_moduli.len()),
                    qj_hat_modqpj,
                )
                .unwrap();
                qj_hat_modqpj_parts.push(qj_hat_modqpj);

                qpj_moduli_parts.push(
                    qpj_moduli
                        .iter()
                        .map(|modqpj| Modulus::new(*modqpj))
                        .collect_vec(),
                );
            });

        // Precompute for P to Q (used for approx_switch_crt_basis in approx_mod_down)
        let mut p_hat_inv_modp = vec![];
        let mut p_hat_modq = vec![];
        params.special_moduli.iter().for_each(|(pi)| {
            p_hat_inv_modp.push(
                (&p_dig / pi)
                    .mod_inverse(BigUintDig::from_u64(*pi).unwrap())
                    .unwrap()
                    .to_biguint()
                    .unwrap()
                    .to_u64()
                    .unwrap(),
            );

            // pj_hat_modq
            let p_hat = &p / pi;
            q_moduli
                .iter()
                .for_each(|qi| p_hat_modq.push((&p_hat % qi).to_u64().unwrap()));
        });
        let p_hat_inv_modp = Array1::from_shape_vec(alpha, p_hat_inv_modp).unwrap();
        let p_hat_modq = Array2::from_shape_vec((alpha, q_moduli.len()), p_hat_modq).unwrap();
        let mut p_inv_modq = vec![];
        // Precompute for dividing values in basis Q by P (approx_mod_down)
        q_moduli.iter().for_each(|qi| {
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
        let p_inv_modq = Array1::from_shape_vec(q_moduli.len(), p_inv_modq).unwrap();

        // ksk_ctx.moduli_ops.chunks(alpha).for_each(|q_mod_ops| {
        //     qj_moduli_parts.push(q_mod_ops.to_vec());
        // });

        // qp = q + p moduli
        let qp_ctx = Arc::new(PolyContext::new(
            &[q_moduli.to_vec(), params.special_moduli.clone()].concat(),
            ksk_ctx.degree,
        ));

        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);
        let c1s = Self::generate_c1(parts, &qp_ctx, seed);
        let c0s = Self::generate_c0(&c1s, &g, &poly, &sk, rng);

        HybridKeySwitchingKey {
            params: params,
            ksk_ctx: ksk_ctx.clone(),
            qp_ctx,
            seed,
            dnum: parts,

            // approx_switch_crt_basis //
            qj_hat_inv_modqj_parts,
            qj_moduli_parts,
            qj_hat_modqpj_parts,
            qpj_moduli_parts,

            // approx_mod_down //
            p_hat_inv_modp,
            p_hat_modq,
            p_inv_modq,

            c0s: c0s.into_boxed_slice(),
            c1s: c1s.into_boxed_slice(),
        }
    }

    pub fn switch(&self, poly: &Poly<T>) -> (Poly<T>, Poly<T>) {
        debug_assert!(poly.representation == Representation::Coefficient);
        debug_assert!(poly.context == self.ksk_ctx);

        let alpha = self.params.alpha;
        let dnum = self.dnum;

        // divide poly into parts and switch them from Qj to QP
        let mut poly_parts_qp = vec![];
        for i in 0..self.dnum {
            let mut qp_poly = Poly::zero(&self.qp_ctx, &Representation::Coefficient);

            let qj_coefficients = {
                if (i + 1) == self.dnum {
                    poly.coefficients.slice(s![(i * alpha).., ..])
                } else {
                    poly.coefficients
                        .slice(s![(i * alpha)..((i + 1) * alpha), ..])
                }
            };
            let mut parts_count = qj_coefficients.shape()[0];

            let mut p_whole_coefficients = Poly::<T>::approx_switch_crt_basis(
                &qj_coefficients,
                &self.qj_moduli_parts[i],
                poly.context.degree,
                &self.qj_hat_inv_modqj_parts[i],
                &self.qj_hat_modqpj_parts[i],
                &self.qpj_moduli_parts[i],
            );

            // TODO: can you parallelize following 3 operations ?
            // ..p_start
            azip!(
                qp_poly
                    .coefficients
                    .slice_mut(s![..(i * alpha), ..])
                    .outer_iter_mut()
                    .into_producer(),
                p_whole_coefficients
                    .slice(s![..(i * alpha), ..])
                    .outer_iter()
                    .into_producer()
            )
            .for_each(|mut qpi, pi| {
                qpi.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(pi.as_slice().unwrap());
            });

            // p_start..p_start+qj
            azip!(
                qp_poly
                    .coefficients
                    .slice_mut(s![(i * alpha)..(i * alpha + parts_count), ..])
                    .outer_iter_mut()
                    .into_producer(),
                qj_coefficients.outer_iter().into_producer()
            )
            .for_each(|mut qpi, qj| {
                qpi.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(qj.as_slice().unwrap());
            });

            // p_start+qj..
            azip!(
                qp_poly
                    .coefficients
                    .slice_mut(s![(i * alpha + parts_count).., ..])
                    .outer_iter_mut()
                    .into_producer(),
                p_whole_coefficients
                    .slice(s![i * alpha.., ..])
                    .outer_iter()
                    .into_producer()
            )
            .for_each(|mut qpi, pi| {
                qpi.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(pi.as_slice().unwrap());
            });

            qp_poly.change_representation(Representation::Evaluation);
            poly_parts_qp.push(qp_poly);
        }

        // perform key switching
        let mut c0_out = &poly_parts_qp[0] * &self.c0s[0];
        poly_parts_qp[0] *= &self.c1s[0];
        // FIXME: I find this clone unnecessary
        let mut c1_out = poly_parts_qp[0].clone();

        izip!(poly_parts_qp.iter_mut(), self.c0s.iter(), self.c1s.iter())
            .skip(1)
            .for_each(|(p, c0i, c1i)| {
                c0_out += &(c0i * p);
                *p *= &c1i;
                c1_out += &p
            });

        // switch results from QP to Q
        c0_out.approx_mod_down(
            &self.ksk_ctx,
            &self.params.special_moduli_ctx,
            &self.p_hat_inv_modp.as_slice().unwrap(),
            &self.p_hat_modq,
            &self.p_inv_modq.as_slice().unwrap(),
        );

        c1_out.approx_mod_down(
            &self.ksk_ctx,
            &self.params.special_moduli_ctx,
            &self.p_hat_inv_modp.as_slice().unwrap(),
            &self.p_hat_modq,
            &self.p_inv_modq.as_slice().unwrap(),
        );

        (c0_out, c1_out)
    }

    pub fn generate_c1(
        count: usize,
        qp_ctx: &Arc<PolyContext<T>>,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
    ) -> Vec<Poly<T>> {
        let mut rng = ChaCha8Rng::from_seed(seed);
        (0..count)
            .into_iter()
            .map(|_| Poly::random(qp_ctx, &Representation::Evaluation, &mut rng))
            .collect_vec()
    }

    pub fn generate_c0<R: CryptoRng + CryptoRngCore>(
        c1s: &[Poly<T>],
        g: &[BigUint],
        // ksk_ctx: &Arc<PolyContext>,
        poly: &Poly<T>,
        sk: &SecretKey<T>,
        rng: &mut R,
    ) -> Vec<Poly<T>> {
        debug_assert!(poly.representation == Representation::Evaluation);
        debug_assert!(g.len() == c1s.len());

        let qp_ctx = c1s[0].context.clone();
        // make sure special P exists in QP
        debug_assert!(poly.context.moduli.len() < qp_ctx.moduli.len());
        let c0s = izip!(c1s.iter(), g)
            .map(|(c1, g_part)| {
                let mut c0 = Poly::zero(&qp_ctx, &Representation::Evaluation);
                let mut e = Poly::random_gaussian(&qp_ctx, &Representation::Coefficient, 10, rng);
                e.change_representation(Representation::Evaluation);

                // An alternate to this will be to extend poly from Q to QP and calculate c0 = g * poly + e - (c1*sk). However there are two drawbacks to this:
                // 1. This will require pre-computation for switching Q to P and then extending Q to QP
                // 2. Notice that calculating P part will be useless since it will be multiplied by `g` afterwards. `g` is of form `P * (Q/Qj)^-1_Qj * (Q/Qj)` and will vanish
                // over `pi`.

                // Q parts
                // g = P * Qj_hat * Qj_hat_inv_modQj
                // [c0]_qi = [g * poly]_qi + [e]_qi - [c1s * sk]_qi
                izip!(
                    poly.context.moduli_ops.iter(),
                    poly.context.ntt_ops.iter(),
                    poly.coefficients.outer_iter(),
                    c0.coefficients.outer_iter_mut(),
                    c1.coefficients.outer_iter(),
                    e.coefficients.outer_iter(),
                )
                .for_each(|(modq, nttq, vqi, mut c0qi, c1qi, eqi)| {
                    let mut skqi = modq.reduce_vec_i64_small(&sk.coefficients);
                    nttq.forward(&mut skqi);

                    // [g * poly]_qi
                    c0qi.as_slice_mut()
                        .unwrap()
                        .copy_from_slice(vqi.as_slice().unwrap());
                    let g_u64 = (g_part % modq.modulus()).to_u64().unwrap();
                    modq.scalar_mul_mod_fast_vec(c0qi.as_slice_mut().unwrap(), g_u64);

                    // [g * poly]_qi + [e]_qi
                    modq.add_mod_fast_vec(c0qi.as_slice_mut().unwrap(), eqi.as_slice().unwrap());

                    // [c1s * sk]_qi
                    modq.mul_mod_fast_vec(&mut skqi, c1qi.as_slice().unwrap());

                    // [g * poly]_qi + [e]_qi - [c1s * sk]_qi
                    modq.sub_mod_fast_vec(c0qi.as_slice_mut().unwrap(), &skqi);
                });

                // P parts
                // [c0]_pi = [e]_pi - [c1s * sk]_pi
                // Note: `g` vanishes over pi
                let to_skip = poly.context.moduli.len();
                izip!(
                    qp_ctx.moduli_ops.iter().skip(to_skip),
                    qp_ctx.ntt_ops.iter().skip(to_skip),
                    c0.coefficients.outer_iter_mut().skip(to_skip),
                    c1.coefficients.outer_iter().skip(to_skip),
                    e.coefficients.outer_iter().skip(to_skip),
                )
                .for_each(|((modpi, nttpi, mut c0pi, c1pi, epi))| {
                    c0pi.as_slice_mut()
                        .unwrap()
                        .copy_from_slice(epi.as_slice().unwrap());

                    let mut skpi = modpi.reduce_vec_i64_small(&sk.coefficients);
                    nttpi.forward(&mut skpi);
                    modpi.mul_mod_fast_vec(&mut skpi, c1pi.as_slice().unwrap());

                    modpi.sub_mod_fast_vec(c0pi.as_slice_mut().unwrap(), &skpi);
                });

                c0
            })
            .collect_vec();
        c0s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::BfvParameters;
    use num_bigint::BigUint;
    use rand::thread_rng;
    use rayon::prelude::{IntoParallelIterator, ParallelBridge, ParallelIterator};

    #[test]
    fn key_switching_works() {
        let bfv_params = Arc::new(BfvParameters::default(15, 1 << 15));
        let ct_ctx = bfv_params.ciphertext_poly_contexts[0].clone();
        let ksk_ctx = ct_ctx.clone();

        let mut rng = thread_rng();

        let sk = SecretKey::random(&bfv_params, &mut rng);

        let poly = Poly::random(&ksk_ctx, &Representation::Evaluation, &mut rng);
        let ksk = BVKeySwitchingKey::new(&poly, &sk, &ct_ctx, &mut rng);

        let mut other_poly = Poly::random(&ct_ctx, &Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let cs = ksk.switch(&other_poly);
        println!("Time elapsed: {:?}", now.elapsed());

        let mut sk_poly = Poly::try_convert_from_i64_small(
            &sk.coefficients,
            &ksk_ctx,
            &Representation::Coefficient,
        );
        sk_poly.change_representation(Representation::Evaluation);
        let mut res = &cs[0] + &(&cs[1] * &sk_poly);

        // expected
        other_poly.change_representation(Representation::Evaluation);
        other_poly *= &poly;

        res -= &other_poly;
        res.change_representation(Representation::Coefficient);

        izip!(Vec::<BigUint>::from(&res).iter(),).for_each(|v| {
            let diff_bits = std::cmp::min(v.bits(), (ksk_ctx.modulus() - v).bits());
            // dbg!(&diff_bits);
            assert!(diff_bits <= 70);
        });
    }

    #[test]
    fn hybrid_key_switching() {
        let bfv_params = Arc::new(BfvParameters::default(5, 1 << 3));
        let ct_ctx = bfv_params.ciphertext_poly_contexts[0].clone();
        let ksk_ctx = ct_ctx.clone();

        let mut rng = thread_rng();

        let sk = SecretKey::random(&bfv_params, &mut rng);

        let poly = Poly::random(&ksk_ctx, &Representation::Evaluation, &mut rng);
        let ksk = HybridKeySwitchingKey::new(&poly, &sk, &ct_ctx, &mut rng);

        let mut other_poly = Poly::random(&ct_ctx, &Representation::Coefficient, &mut rng);
        let now = std::time::Instant::now();
        let (cs0, cs1) = ksk.switch(&other_poly);
        println!("Time elapsed: {:?}", now.elapsed());

        let mut sk_poly = Poly::try_convert_from_i64_small(
            &sk.coefficients,
            &ksk_ctx,
            &Representation::Coefficient,
        );
        sk_poly.change_representation(Representation::Evaluation);
        let mut res = &cs0 + &(&cs1 * &sk_poly);

        // expected
        other_poly.change_representation(Representation::Evaluation);
        other_poly *= &poly;

        res -= &other_poly;
        res.change_representation(Representation::Coefficient);
        izip!(Vec::<BigUint>::from(&res).iter(),).for_each(|v| {
            let diff_bits = std::cmp::min(v.bits(), (ksk_ctx.modulus() - v).bits());
            // dbg!(diff_bits);
        });
    }

    #[test]
    fn comp() {
        let bfv_params = Arc::new(BfvParameters::default(14, 1 << 15));
        let ctx = bfv_params.ciphertext_poly_contexts[0].clone();

        let mut rng = thread_rng();
        let mut p = Poly::random(&ctx, &Representation::Coefficient, &mut rng);
        let q = Poly::random(&ctx, &Representation::Coefficient, &mut rng);
        let modop = Modulus::new(32);

        // rayon::ThreadPoolBuilder::new()
        //     .num_threads(10)
        //     .build_global()
        //     .unwrap();

        let now = std::time::Instant::now();
        azip!(
            p.coefficients.axis_iter_mut(Axis(1)),
            q.coefficients.axis_iter(Axis(1))
        )
        .par_for_each(|mut a, b| {
            a[0] += modop.mul_mod_fast(a[0], b[0]);
            a[1] += modop.mul_mod_fast(a[1], b[1]);
            a[2] += modop.mul_mod_fast(a[2], b[2]);
            a[3] += modop.mul_mod_fast(a[5], b[5]);
            a[4] += modop.mul_mod_fast(a[3], b[3]);
            a[5] += modop.mul_mod_fast(a[4], b[4]);
        });
        println!("azip! took: {:?}", now.elapsed());

        let mut p = Poly::random(&ctx, &Representation::Coefficient, &mut rng);
        let q = Poly::random(&ctx, &Representation::Coefficient, &mut rng);
        let modop = Modulus::new(32);

        let now = std::time::Instant::now();
        izip!(
            p.coefficients.axis_iter_mut(Axis(1)),
            q.coefficients.axis_iter(Axis(1))
        )
        .for_each(|(mut a, b)| {
            a[0] += modop.mul_mod_fast(a[0], b[0]);
            a[1] += modop.mul_mod_fast(a[1], b[1]);
            a[2] += modop.mul_mod_fast(a[2], b[2]);
            a[3] += modop.mul_mod_fast(a[5], b[5]);
            a[4] += modop.mul_mod_fast(a[3], b[3]);
            a[5] += modop.mul_mod_fast(a[4], b[4]);
        });
        println!("izip! took: {:?}", now.elapsed());
    }
}
