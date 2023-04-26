use crate::{
    nb_theory::generate_prime,
    poly::{Poly, PolyContext, Representation},
    SecretKey,
};
use crypto_bigint::rand_core::CryptoRngCore;
use fhe_math::zq::Modulus;
use itertools::{izip, Itertools};
use ndarray::{s, Array2, Array3};
use num_bigint::BigUint;
use num_bigint_dig::BigUint as BigUintDig;
use num_bigint_dig::ModInverse;
use num_traits::{FromPrimitive, One, ToPrimitive};
use rand::{CryptoRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

struct BVKeySwitchingKey {
    c0s: Box<[Poly]>,
    c1s: Box<[Poly]>,
    seed: <ChaCha8Rng as SeedableRng>::Seed,
    ciphertext_ctx: Arc<PolyContext>,
    ksk_ctx: Arc<PolyContext>,
}

impl BVKeySwitchingKey {
    pub fn new<R: CryptoRng + CryptoRngCore>(
        poly: &Poly,
        sk: &SecretKey,
        ciphertext_ctx: &Arc<PolyContext>,
        rng: &mut R,
    ) -> BVKeySwitchingKey {
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

    pub fn switch(&self, poly: &Poly) -> Vec<Poly> {
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
        ksk_ctx: &Arc<PolyContext>,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
    ) -> Vec<Poly> {
        let mut rng = ChaCha8Rng::from_seed(seed);
        (0..count)
            .into_iter()
            .map(|_| Poly::random(ksk_ctx, &Representation::Evaluation, &mut rng))
            .collect_vec()
    }

    pub fn generate_c0<R: CryptoRng + CryptoRngCore>(
        ciphertext_ctx: &Arc<PolyContext>,
        ksk_ctx: &Arc<PolyContext>,
        poly: &Poly,
        c1s: &[Poly],
        sk: &SecretKey,
        rng: &mut R,
    ) -> Vec<Poly> {
        // encrypt g corresponding to every qi in ciphertext
        // make sure that you have enough c1s
        debug_assert!(ciphertext_ctx.moduli.len() == c1s.len());
        debug_assert!(poly.representation == Representation::Evaluation);

        let mut sk =
            Poly::try_convert_from_i64(&sk.coefficients, ksk_ctx, &Representation::Coefficient);
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

struct HybridKeySwitchingKey {
    ciphertext_ctx: Arc<PolyContext>,
    ksk_ctx: Arc<PolyContext>,
    qp_ctx: Arc<PolyContext>,
    seed: <ChaCha8Rng as SeedableRng>::Seed,
    q_hat_inv_modq_parts: Vec<Vec<u64>>,
    q_mod_ops_parts: Vec<Vec<Modulus>>,
    q_hat_modp_parts: Vec<Array2<u64>>,
    p_moduli_parts: Vec<Vec<u64>>,
    dnum: usize,
    alpha: usize,
    c0s: Box<[Poly]>,
    c1s: Box<[Poly]>,
}

impl HybridKeySwitchingKey {
    /// Warning: Ciphertext context needs to be as same as KeySwitching Context. This is not
    /// a limitation of hybrid key switching, instead a limitation of the way key switching is
    /// implemented here.
    /// Let's say ciphertext ctx = Q' and ksk ctx = Q. The extended ctx should be QP. To speed things
    /// up during `key_switch` operation, we assume Q == Q' because we extend poly from Qj to Q[..i*dnum] + Q[(i+1)*dnum..] + P.
    pub fn new<R: CryptoRng + CryptoRngCore>(
        poly: &Poly,
        sk: &SecretKey,
        ciphertext_ctx: &Arc<PolyContext>,
        rng: &mut R,
    ) -> HybridKeySwitchingKey {
        let dnum = 3;
        let aux_bits = 60;

        debug_assert!(ciphertext_ctx == &poly.context);

        let alpha = (ciphertext_ctx.moduli.len() + (dnum >> 2)) / dnum;

        let ksk_ctx = poly.context.clone();

        // generate special moduli P
        let mut qj = vec![];
        ciphertext_ctx
            .moduli
            .chunks(dnum)
            .for_each(|q_parts_moduli| {
                // Qj
                let mut qji = BigUint::one();
                q_parts_moduli.iter().for_each(|qi| {
                    qji *= *qi;
                });
                qj.push(qji);
            });
        let mut maxbits = qj[0].bits();
        qj.iter().skip(1).for_each(|q| {
            maxbits = std::cmp::max(maxbits, q.bits());
        });
        let size_p = (maxbits as f64 / aux_bits as f64).ceil() as usize;
        let mut p_moduli = vec![];
        // TODO: generate primes for P and then multiply P with g below
        let mut upper_bound = 1 << aux_bits;
        for _ in 0..size_p {
            loop {
                if let Some(prime) =
                    generate_prime(aux_bits, (2 * ksk_ctx.degree) as u64, 1 << aux_bits)
                {
                    if !p_moduli.contains(&prime) && !ksk_ctx.moduli.contains(&prime) {
                        upper_bound = prime;
                    } else {
                        p_moduli.push(prime);
                        break;
                    }
                } else {
                    panic!("Not enough primes for special moduli P in Hybrid key switching");
                }
            }
        }
        let p_ctx = Arc::new(PolyContext::new(&p_moduli, ksk_ctx.degree));
        let mut p = p_ctx.modulus();

        // TODO: move all pre-computation stuff to some other place.
        let q = ciphertext_ctx.modulus();
        let q_dig = ciphertext_ctx.modulus_dig();
        let q_moduli = ciphertext_ctx.moduli.clone();
        // g = P * Qj_hat * Qj_hat_inv_modQj
        let mut g = vec![];
        // FIXME: we use 2d Vec instead of Array2 because the last part may contain less than dnum qis.
        // But this isn't acceptable. Change this to Array2 and adjust for last part somehow.
        let mut q_hat_inv_modq_parts = vec![vec![]];
        let mut q_hat_modp_parts = vec![];
        let mut p_moduli_parts = vec![];
        let mut q_mod_ops_parts = vec![];
        q_moduli
            .chunks(dnum)
            .enumerate()
            .for_each(|(chunk_index, q_parts_moduli)| {
                // Qj
                let mut qj = BigUint::one();
                let mut qj_dig = BigUintDig::one();
                q_parts_moduli.iter().for_each(|qji| {
                    qj *= *qji;
                    qj_dig *= *qji;
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

                // for approx_switch_crt_basis
                let mut qj_hat_inv_modqj = vec![];
                q_parts_moduli.iter().for_each(|qji| {
                    let qji_hat_inv_modqji = (&qj_dig / *qji)
                        .mod_inverse(BigUintDig::from_u64(*qji).unwrap())
                        .unwrap()
                        .to_biguint()
                        .unwrap()
                        .to_u64()
                        .unwrap();
                    qj_hat_inv_modqj.push(qji_hat_inv_modqji);
                });
                q_hat_inv_modq_parts.push(qj_hat_inv_modqj);

                let p_start = q_moduli[..dnum * chunk_index].to_vec();
                let p_mid = {
                    if (dnum * (chunk_index + 1)) < q_moduli.len() {
                        q_moduli[(dnum * (chunk_index + 1))..].to_vec()
                    } else {
                        vec![]
                    }
                };
                let p_whole = [p_start, p_mid, p_moduli.clone()].concat();
                let mut q_hat_modp = vec![];
                q_parts_moduli.iter().for_each(|qji| {
                    p_whole.iter().for_each(|pk| {
                        q_hat_modp.push(((&qj / qji) % pk).to_u64().unwrap());
                    });
                });
                let q_hat_modp = Array2::<u64>::from_shape_vec(
                    (q_parts_moduli.len(), p_whole.len()),
                    q_hat_modp,
                )
                .unwrap();
                q_hat_modp_parts.push(q_hat_modp);
                p_moduli_parts.push(p_whole);
            });
        ciphertext_ctx
            .moduli_ops
            .chunks(dnum)
            .for_each(|q_mod_ops| {
                q_mod_ops_parts.push(q_mod_ops.to_vec());
            });

        let parts = g.len();

        // QP = ksk_ctx.modulus() + P;
        let qp_moduli = [ksk_ctx.moduli.clone(), p_ctx.moduli.clone()].concat();
        let qp_ctx = Arc::new(PolyContext::new(&qp_moduli, ksk_ctx.degree));

        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);
        let c1s = Self::generate_c1(parts, &qp_ctx, seed);
        let c0s = Self::generate_c0(&c1s, &g, &poly, &sk, rng);

        HybridKeySwitchingKey {
            ciphertext_ctx: ciphertext_ctx.clone(),
            ksk_ctx: ksk_ctx.clone(),
            qp_ctx: qp_ctx.clone(),
            seed,
            q_hat_inv_modq_parts,
            q_hat_modp_parts,
            p_moduli_parts,
            q_mod_ops_parts,
            dnum,
            alpha,
            c0s: c0s.into_boxed_slice(),
            c1s: c1s.into_boxed_slice(),
        }
    }

    pub fn key_switch(&self, poly: &Poly) {
        debug_assert!(poly.representation == Representation::Coefficient);
        debug_assert!(poly.context == self.ciphertext_ctx);

        // divide poly into parts and switch them from Qj to QP
        let mut poly_parts_qp = vec![];
        for i in 0..self.alpha {
            let mut qp_poly = Poly::zero(&self.qp_ctx, &Representation::Coefficient);

            let qj_coefficients = {
                if (i + 1) == self.alpha {
                    poly.coefficients
                        .slice(s![(i * self.dnum).., ..])
                        .to_owned()
                } else {
                    poly.coefficients
                        .slice(s![(i * self.dnum)..((i + 1) * self.dnum), ..])
                        .to_owned()
                }
            };
            let mut parts_count = qj_coefficients.shape()[0];

            let mut p_whole_coefficients = Poly::approx_switch_crt_basis(
                &qj_coefficients,
                &self.q_mod_ops_parts[i],
                poly.context.degree,
                &self.q_hat_inv_modq_parts[i],
                &self.q_hat_modp_parts[i],
                &self.p_moduli_parts[i],
            );

            // ..p_start
            izip!(
                qp_poly.coefficients.outer_iter_mut().take(i * self.dnum),
                p_whole_coefficients.outer_iter().take(i * self.dnum)
            )
            .for_each(|(mut qpi, pi)| {
                qpi.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(pi.as_slice().unwrap());
            });

            // p_start..p_start+qj
            izip!(
                qp_poly.coefficients.outer_iter_mut().skip(i * self.dnum),
                qj_coefficients.outer_iter()
            )
            .for_each(|(mut qpi, qj)| {
                qpi.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(qj.as_slice().unwrap());
            });

            // p_start+qj..
            izip!(
                qp_poly
                    .coefficients
                    .outer_iter_mut()
                    .skip(i * self.dnum + parts_count),
                p_whole_coefficients.outer_iter().skip(i * self.dnum)
            )
            .for_each(|(mut qpi, pi)| {
                qpi.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(pi.as_slice().unwrap());
            });

            qp_poly.change_representation(Representation::Evaluation);
            poly_parts_qp.push(qp_poly);
        }

        // perform key switching
        let mut c0_out = &poly_parts_qp[0] * &self.c0s[0];
        let mut c1_out = &poly_parts_qp[0] * &self.c1s[0];

        izip!(poly_parts_qp.iter(), self.c0s.iter(), self.c1s.iter())
            .skip(1)
            .for_each(|(p, c0i, c1i)| {
                c0_out += &(p * c0i);
                c1_out += &(p * c1i);
            });

        //TODO: switch results from QP to Q
    }

    pub fn generate_c1(
        count: usize,
        qp_ctx: &Arc<PolyContext>,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
    ) -> Vec<Poly> {
        let mut rng = ChaCha8Rng::from_seed(seed);
        (0..count)
            .into_iter()
            .map(|_| Poly::random(qp_ctx, &Representation::Evaluation, &mut rng))
            .collect_vec()
    }

    pub fn generate_c0<R: CryptoRng + CryptoRngCore>(
        c1s: &[Poly],
        g: &[BigUint],
        // ksk_ctx: &Arc<PolyContext>,
        poly: &Poly,
        sk: &SecretKey,
        rng: &mut R,
    ) -> Vec<Poly> {
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

                // An alternate to this will to be extend poly from Q to QP and calculate c0 = g * poly + e - (c1*sk). However there are two drawbacks to this:
                // 1. This will require pre-computation for switching Q to P and then extending Q to QP
                // 2. Notice that calculating P part will be useless since it will be multiplied by `g` afterwards. `g` is of form `P * (Q/Qj)^-1_Qj * (Q/Qj)` and will vanish
                // over pi.

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
                    let mut skqi = modq.reduce_vec_i64(&sk.coefficients);
                    nttq.forward(&mut skqi);

                    // [g * poly]_qi
                    c0qi.as_slice_mut()
                        .unwrap()
                        .copy_from_slice(vqi.as_slice().unwrap());
                    let g_u64 = (g_part % modq.modulus()).to_u64().unwrap();
                    modq.scalar_mul_vec(c0qi.as_slice_mut().unwrap(), g_u64);

                    // [g * poly]_qi + [e]_qi
                    modq.add_vec(c0qi.as_slice_mut().unwrap(), eqi.as_slice().unwrap());

                    // [c1s * sk]_qi
                    modq.mul_vec(&mut skqi, c1qi.as_slice().unwrap());

                    // [g * poly]_qi + [e]_qi - [c1s * sk]_qi
                    modq.sub_vec(c0qi.as_slice_mut().unwrap(), &skqi);
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

                    let mut skpi = modpi.reduce_vec_i64(&sk.coefficients);
                    nttpi.forward(&mut skpi);
                    modpi.mul_vec(&mut skpi, c1pi.as_slice().unwrap());

                    modpi.sub_vec(c0pi.as_slice_mut().unwrap(), &skpi);
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
    use crate::BfvParameters;
    use num_bigint::BigUint;
    use rand::thread_rng;

    #[test]
    fn key_switching_works() {
        let bfv_params = Arc::new(BfvParameters::new(
            &[60, 60, 60, 60, 60, 60, 60],
            65537,
            1 << 8,
        ));
        let ct_ctx = bfv_params.ciphertext_poly_contexts[0].clone();
        let ksk_ctx = ct_ctx.clone();

        let mut rng = thread_rng();

        let sk = SecretKey::random(&bfv_params, &mut rng);

        let poly = Poly::random(&ksk_ctx, &Representation::Evaluation, &mut rng);
        let ksk = BVKeySwitchingKey::new(&poly, &sk, &ct_ctx, &mut rng);

        let mut other_poly = Poly::random(&ct_ctx, &Representation::Coefficient, &mut rng);
        let cs = ksk.switch(&other_poly);

        let mut sk_poly =
            Poly::try_convert_from_i64(&sk.coefficients, &ksk_ctx, &Representation::Coefficient);
        sk_poly.change_representation(Representation::Evaluation);
        let mut res = &cs[0] + &(&cs[1] * &sk_poly);

        // expected
        other_poly.change_representation(Representation::Evaluation);
        other_poly *= &poly;

        res -= &other_poly;
        res.change_representation(Representation::Coefficient);

        izip!(Vec::<BigUint>::from(&res).iter(),).for_each(|v| {
            let diff_bits = std::cmp::min(v.bits(), (ksk_ctx.modulus() - v).bits());
            assert!(diff_bits <= 70);
        });
    }
}
