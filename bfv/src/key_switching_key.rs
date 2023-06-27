use crate::modulus::Modulus;
use crate::{mod_inverse_biguint, mod_inverse_biguint_u64};
use crate::{secret_key::SecretKey, Poly, PolyContext, Representation};
use crypto_bigint::rand_core::CryptoRngCore;
use itertools::{izip, Itertools};
use ndarray::{azip, s, Array1, Array2, Array3, Axis, IntoNdProducer};
use num_bigint::{BigUint, ToBigInt};
use num_traits::{FromPrimitive, One, ToPrimitive};
use rand::{CryptoRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use traits::Ntt;
struct BVKeySwitchingKey {
    c0s: Box<[Poly]>,
    c1s: Box<[Poly]>,
    seed: <ChaCha8Rng as SeedableRng>::Seed,
}

impl BVKeySwitchingKey {
    pub fn new<R: CryptoRng + CryptoRngCore>(
        poly: &Poly,
        sk: &SecretKey,
        ksk_ctx: &PolyContext<'_>,
        rng: &mut R,
    ) -> BVKeySwitchingKey {
        // check that ciphertext context has more than on moduli, otherwise key switching does not makes sense
        debug_assert!(ksk_ctx.moduli_count > 1);

        // c1s
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);
        let c1s = Self::generate_c1(ksk_ctx, seed);
        let c0s = Self::generate_c0(ksk_ctx, poly, &c1s, sk, rng);

        BVKeySwitchingKey {
            c0s: c0s.into_boxed_slice(),
            c1s: c1s.into_boxed_slice(),
            seed,
        }
    }

    pub fn switch(&self, poly: &Poly, ksk_ctx: &PolyContext<'_>) -> (Poly, Poly) {
        // TODO: check that poly matches ksk_ctx
        // And ksk_ctx matches the key
        debug_assert!(poly.representation == Representation::Coefficient);

        let mut p = ksk_ctx.try_convert_from_u64(
            poly.coefficients.slice(s![0, ..]).as_slice().unwrap(),
            Representation::Coefficient,
        );
        ksk_ctx.change_representation(&mut p, Representation::Evaluation);
        let mut c1_out = ksk_ctx.mul(&self.c1s[0], &p);
        ksk_ctx.mul_assign(&mut p, &self.c0s[0]);
        let mut c0_out = p;

        izip!(
            self.c0s.iter(),
            self.c1s.iter(),
            poly.coefficients.outer_iter()
        )
        .skip(1)
        .for_each(|(c0, c1, rests)| {
            let mut p = ksk_ctx
                .try_convert_from_u64(rests.as_slice().unwrap(), Representation::Coefficient);
            ksk_ctx.change_representation(&mut p, Representation::Evaluation);

            ksk_ctx.add_assign(&mut c1_out, &ksk_ctx.mul(c1, &p));
            ksk_ctx.mul_assign(&mut p, &c0);
            ksk_ctx.add_assign(&mut c0_out, &p);
        });

        (c0_out, c1_out)
    }

    pub fn generate_c1(
        ksk_ctx: &PolyContext<'_>,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
    ) -> Vec<Poly> {
        let mut rng = ChaCha8Rng::from_seed(seed);
        (0..ksk_ctx.moduli_count)
            .into_iter()
            .map(|_| ksk_ctx.random(Representation::Evaluation, &mut rng))
            .collect_vec()
    }

    pub fn generate_c0<R: CryptoRng + CryptoRngCore>(
        ksk_ctx: &PolyContext<'_>,
        poly: &Poly,
        c1s: &[Poly],
        sk: &SecretKey,
        rng: &mut R,
    ) -> Vec<Poly> {
        debug_assert!(poly.representation == Representation::Evaluation);

        let mut sk =
            ksk_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        ksk_ctx.change_representation(&mut sk, Representation::Evaluation);

        // gi = (q/qi) * [(q/qi)^-1]_qi
        let big_q = ksk_ctx.big_q();
        let g = ksk_ctx
            .iter_moduli_ops()
            .map(|modqi| {
                let qi = modqi.modulus();
                let qi_hat = &big_q / qi;
                &qi_hat * mod_inverse_biguint_u64(&qi_hat, qi)
            })
            .collect_vec();

        izip!(g.into_iter(), c1s.iter())
            .map(|(g, c1)| {
                let mut g = ksk_ctx.try_convert_from_biguint(
                    vec![g.clone(); ksk_ctx.degree].as_slice(),
                    Representation::Evaluation,
                );
                // m = gi*poly
                ksk_ctx.mul_assign(&mut g, &poly);

                let mut e = ksk_ctx.random_gaussian(Representation::Coefficient, 10, rng);
                ksk_ctx.change_representation(&mut e, Representation::Evaluation);
                // m + e
                ksk_ctx.add_assign(&mut e, &g);

                // -a*sk
                let c1_sk = ksk_ctx.mul(&c1, &sk);

                // m + e - a*sk
                ksk_ctx.sub_assign(&mut e, &c1_sk);

                e
            })
            .collect_vec()
    }
}

pub struct HybridKeySwitchingKey {
    // ksk_ctx is q_ctx
    seed: <ChaCha8Rng as SeedableRng>::Seed,
    dnum: usize,
    alpha: usize,
    aux_bits: usize,

    // approx_switch_crt_basis //
    qj_hat_inv_modqj_parts: Vec<Vec<u64>>,
    qj_moduli_ops_parts: Vec<Vec<Modulus>>,
    qj_hat_modqpj_parts: Vec<Array2<u64>>,
    qpj_moduli_ops_parts: Vec<Vec<Modulus>>,

    // approx_mod_down //
    p_hat_inv_modp: Vec<u64>,
    p_hat_modq: Array2<u64>,
    p_inv_modq: Vec<u64>,

    c0s: Box<[Poly]>,
    c1s: Box<[Poly]>,
}

impl HybridKeySwitchingKey {
    /// Warning: Ciphertext context needs to be as same as KeySwitching Context. This is not
    /// a limitation of hybrid key switching, instead a limitation of the way key switching is
    /// implemented here.
    /// Let's say ciphertext ctx = Q' and ksk ctx = Q. The extended ctx should be QP. To speed things
    /// up during `key_switch` operation, we assume Q == Q' because we extend poly from Qj to Q[..i*alpha] + Q[(i+1)*alpha..] + P.
    pub fn new<R: CryptoRng + CryptoRngCore>(
        poly: &Poly,
        sk: &SecretKey,
        ksk_ctx: &PolyContext<'_>,
        specialp_ctx: &PolyContext<'_>,
        qp_ctx: &PolyContext<'_>,
        alpha: usize,
        aux_bits: usize,
        rng: &mut R,
    ) -> HybridKeySwitchingKey {
        let mut qj = vec![];
        ksk_ctx
            .iter_moduli_ops()
            .chunks(alpha)
            .into_iter()
            .for_each(|modqi_parts| {
                // Qj
                let mut qji = BigUint::one();
                modqi_parts.into_iter().for_each(|modqi| {
                    qji *= modqi.modulus();
                });
                qj.push(qji);
            });

        // This is just for precaution. Ideally alpha*aux_bits must be greater than maxbits. But
        // having alpha*auxbits - maxbits greater than a few bits costs performance, since you end
        // up having an additional special prime for nothing. In our case, alpha and aux_bites are fixed
        // to 3 and 60. Hence, assuming that you have alpha consecutive ciphertext primes of size 40-60
        // (usually the case for lower levels) it must be the case that maxbits is around 120.
        let mut maxbits = qj[0].bits();
        qj.iter().skip(1).for_each(|q| {
            maxbits = std::cmp::max(maxbits, q.bits());
        });
        let ideal_special_primes_count = (maxbits as f64 / aux_bits as f64).ceil() as usize;
        assert!(ideal_special_primes_count == alpha);

        // P is special prime
        let p = specialp_ctx.big_q();

        // Calculating g values + Pre-comp for switching Qj to QP.
        let q = ksk_ctx.big_q();
        // g = P * Q/Qj * [(Q/Qj)^-1]_Qj
        let mut g = vec![];
        // we are forced to use Vec insted of `ndarray` because each of the
        // 2d vectors cannot be translated to a proper array in row major form.
        let mut qj_hat_inv_modqj_parts = vec![];
        let mut qj_hat_modqpj_parts = vec![];
        let mut qpj_moduli_ops_parts = vec![];
        let mut qj_moduli_ops_parts = vec![];

        let parts = (ksk_ctx.moduli_count as f64 / alpha as f64).ceil() as usize;

        ksk_ctx
            .iter_moduli_ops()
            .chunks(alpha)
            .into_iter()
            .enumerate()
            .for_each(|(chunk_index, modqj)| {
                let qj_moduli_ops = modqj.into_iter().map(|modqji| modqji.clone()).collect_vec();

                // Qj
                let mut qj = BigUint::one();
                qj_moduli_ops.iter().for_each(|modqi| qj *= modqi.modulus());

                // Q/Qj
                let qj_hat = &q / &qj;

                // [(Q/Qj)^-1]_Qj
                let qj_hat_inv_modqj = mod_inverse_biguint(&qj_hat, &qj);
                g.push(&p * qj_hat * qj_hat_inv_modqj);

                // precomp approx_switch_crt_basis
                let mut qj_hat_inv_modqj = vec![];
                qj_moduli_ops.iter().for_each(|modqji| {
                    let qji = modqji.modulus();
                    qj_hat_inv_modqj
                        .push(mod_inverse_biguint_u64(&(&qj / qji), qji).to_u64().unwrap());
                });
                qj_hat_inv_modqj_parts.push(qj_hat_inv_modqj);

                let mut qpj_moduli_ops = vec![];
                qpj_moduli_ops.extend_from_slice(&ksk_ctx.moduli_ops()[..alpha * chunk_index]);
                qpj_moduli_ops.extend_from_slice(
                    &ksk_ctx.moduli_ops()
                        [std::cmp::min(ksk_ctx.moduli_count, alpha * (chunk_index + 1))..],
                );
                qpj_moduli_ops.extend_from_slice(specialp_ctx.moduli_ops());

                let mut qj_hat_modqpj = vec![];
                qpj_moduli_ops.iter().for_each(|modqpji| {
                    qj_moduli_ops.iter().for_each(|modqji| {
                        qj_hat_modqpj.push(
                            ((&qj / modqji.modulus()) % modqpji.modulus())
                                .to_u64()
                                .unwrap(),
                        )
                    })
                });
                let qj_hat_modqpj = Array2::<u64>::from_shape_vec(
                    (qpj_moduli_ops.len(), qj_moduli_ops.len()),
                    qj_hat_modqpj,
                )
                .unwrap();

                qj_hat_modqpj_parts.push(qj_hat_modqpj);
                qpj_moduli_ops_parts.push(qpj_moduli_ops);
                qj_moduli_ops_parts.push(qj_moduli_ops);
            });

        // Precompute for P to Q (used for approx_switch_crt_basis in approx_mod_down)
        let mut p_hat_inv_modp = vec![];
        let mut p_hat_modq = vec![];
        specialp_ctx.iter_moduli_ops().for_each(|(modpi)| {
            let pi = modpi.modulus();
            let pi_hat = &p / pi;
            p_hat_inv_modp.push(mod_inverse_biguint_u64(&pi_hat, pi).to_u64().unwrap());
        });
        ksk_ctx.iter_moduli_ops().for_each(|modqj| {
            specialp_ctx.iter_moduli_ops().for_each(|(modpi)| {
                p_hat_modq.push(((&p / modpi.modulus()) % modqj.modulus()).to_u64().unwrap());
            });
        });
        let p_hat_modq = Array2::from_shape_vec((ksk_ctx.moduli_count, alpha), p_hat_modq).unwrap();
        let mut p_inv_modq = vec![];
        // Precompute for dividing values in basis Q by P (approx_mod_down)
        ksk_ctx.iter_moduli_ops().for_each(|modqi| {
            p_inv_modq.push(
                mod_inverse_biguint_u64(&p, modqi.modulus())
                    .to_u64()
                    .unwrap(),
            );
        });

        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);
        let c1s = Self::generate_c1(parts, &qp_ctx, seed);
        let c0s = Self::generate_c0(&qp_ctx, &c1s, &g, &poly, &sk, rng);

        HybridKeySwitchingKey {
            seed,
            dnum: parts,
            alpha,
            aux_bits,

            // approx_switch_crt_basis //
            qj_hat_inv_modqj_parts,
            qj_moduli_ops_parts,
            qj_hat_modqpj_parts,
            qpj_moduli_ops_parts,

            // approx_mod_down //
            p_hat_inv_modp,
            p_hat_modq,
            p_inv_modq,

            c0s: c0s.into_boxed_slice(),
            c1s: c1s.into_boxed_slice(),
        }
    }

    pub fn switch(
        &self,
        poly: &Poly,
        qp_ctx: &PolyContext<'_>,
        ksk_ctx: &PolyContext<'_>,
        specialp_ctx: &PolyContext<'_>,
    ) -> (Poly, Poly) {
        // TODO: check poly context
        debug_assert!(poly.representation == Representation::Coefficient);

        let alpha = self.alpha;

        // divide poly into parts and switch them from Qj to QP and key switch
        let mut c0_out = Poly::placeholder();
        let mut c1_out = Poly::placeholder();
        for i in 0..self.dnum {
            let mut qp_poly = qp_ctx.zero(Representation::Coefficient);

            let qj_coefficients = {
                if (i + 1) == self.dnum {
                    poly.coefficients.slice(s![(i * alpha).., ..])
                } else {
                    poly.coefficients
                        .slice(s![(i * alpha)..((i + 1) * alpha), ..])
                }
            };
            let mut parts_count = qj_coefficients.shape()[0];

            let mut p_whole_coefficients = PolyContext::approx_switch_crt_basis(
                &qj_coefficients,
                &self.qj_moduli_ops_parts[i],
                qp_ctx.degree,
                &self.qj_hat_inv_modqj_parts[i],
                &self.qj_hat_modqpj_parts[i],
                &self.qpj_moduli_ops_parts[i],
            );

            // ..p_start
            izip!(
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
            .for_each(|(mut qpi, pi)| {
                qpi.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(pi.as_slice().unwrap());
            });

            // p_start..p_start+qj
            izip!(
                qp_poly
                    .coefficients
                    .slice_mut(s![(i * alpha)..(i * alpha + parts_count), ..])
                    .outer_iter_mut()
                    .into_producer(),
                qj_coefficients.outer_iter().into_producer()
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
                    .slice_mut(s![(i * alpha + parts_count).., ..])
                    .outer_iter_mut()
                    .into_producer(),
                p_whole_coefficients
                    .slice(s![i * alpha.., ..])
                    .outer_iter()
                    .into_producer()
            )
            .for_each(|(mut qpi, pi)| {
                qpi.as_slice_mut()
                    .unwrap()
                    .copy_from_slice(pi.as_slice().unwrap());
            });

            qp_ctx.change_representation(&mut qp_poly, Representation::Evaluation);

            if c1_out.representation == Representation::Unknown {
                c1_out = qp_ctx.mul(&qp_poly, &self.c1s[i]);
                qp_ctx.mul_assign(&mut qp_poly, &self.c0s[i]);
                c0_out = qp_poly;
            } else {
                qp_ctx.add_assign(&mut c1_out, &qp_ctx.mul(&qp_poly, &self.c1s[i]));
                qp_ctx.mul_assign(&mut qp_poly, &self.c0s[i]);
                qp_ctx.add_assign(&mut c0_out, &qp_poly);
            }
        }

        // switch results from QP to Q
        qp_ctx.approx_mod_down(
            &mut c0_out,
            &ksk_ctx,
            &specialp_ctx,
            &self.p_hat_inv_modp,
            &self.p_hat_modq,
            &self.p_inv_modq,
        );

        qp_ctx.approx_mod_down(
            &mut c1_out,
            &ksk_ctx,
            &specialp_ctx,
            &self.p_hat_inv_modp,
            &self.p_hat_modq,
            &self.p_inv_modq,
        );

        (c0_out, c1_out)
    }

    fn generate_c1(
        count: usize,
        qp_ctx: &PolyContext<'_>,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
    ) -> Vec<Poly> {
        let mut rng = ChaCha8Rng::from_seed(seed);
        (0..count)
            .into_iter()
            .map(|_| qp_ctx.random(Representation::Evaluation, &mut rng))
            .collect_vec()
    }

    fn generate_c0<R: CryptoRng + CryptoRngCore>(
        qp_ctx: &PolyContext<'_>,
        c1s: &[Poly],
        g: &[BigUint],
        poly: &Poly,
        sk: &SecretKey,
        rng: &mut R,
    ) -> Vec<Poly> {
        //TODO: check poly is of correct context
        debug_assert!(poly.representation == Representation::Evaluation);

        // We run into problem here that makes using API for `PolynomialContext` as usual not desirable. We need to calculate
        // [c0]_QP = [g]_QP * [poly]_QP + [e]_QP - [c1]_QP * [sk]_QP.
        // To calcualte this we have everything in QP basis except `poly`, which we need to extend from Q to QP.
        // But extending poly from Q to QP will be wasteful because it is then multiplied with g and [g]_P is 0 (since P|g).
        // Moreover, this will require having additional pre-computes for switching Q to P. Hence, we prefer the method implemented
        // below. It basically does the same thing but processes Q and P modulus separately.
        let c0s = izip!(c1s.iter(), g)
            .map(|(c1, g_part)| {
                let mut c0 = qp_ctx.zero(Representation::Evaluation);
                let mut e = qp_ctx.random_gaussian(Representation::Coefficient, 10, rng);
                qp_ctx.change_representation(&mut e, Representation::Evaluation);

                // Q
                // g = P * Qj_hat * Qj_hat_inv_modQj
                // [c0]_qi = [g * poly]_qi + [e]_qi - [c1s * sk]_qi
                izip!(
                    qp_ctx.moduli_ops.0.iter(),
                    qp_ctx.ntt_ops.0.iter(),
                    poly.coefficients.outer_iter(),
                    c0.coefficients.outer_iter_mut(),
                    c1.coefficients.outer_iter(),
                    e.coefficients.outer_iter(),
                )
                .for_each(|(modqi, nttqi, xi, mut c0qi, c1qi, eqi)| {
                    let mut skqi = modqi.reduce_vec_i64_small(&sk.coefficients);
                    nttqi.forward(&mut skqi);

                    // [g * poly]_qi
                    c0qi.as_slice_mut()
                        .unwrap()
                        .copy_from_slice(xi.as_slice().unwrap());
                    let g_u64 = (g_part % modqi.modulus()).to_u64().unwrap();
                    modqi.scalar_mul_mod_fast_vec(c0qi.as_slice_mut().unwrap(), g_u64);

                    // [g * poly]_qi + [e]_qi
                    modqi.add_mod_fast_vec(c0qi.as_slice_mut().unwrap(), eqi.as_slice().unwrap());

                    // [c1s * sk]_qi
                    modqi.mul_mod_fast_vec(&mut skqi, c1qi.as_slice().unwrap());

                    // [g * poly]_qi + [e]_qi - [c1s * sk]_qi
                    modqi.sub_mod_fast_vec(c0qi.as_slice_mut().unwrap(), &skqi);
                });

                // P
                // [c0]_pi = [e]_pi - [c1s * sk]_pi
                // (`g` vanishes over pi)
                let to_skip = qp_ctx.moduli_ops.0.len();
                izip!(
                    qp_ctx.moduli_ops.1.iter(),
                    qp_ctx.ntt_ops.1.iter(),
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
    use crate::parameters::{BfvParameters, PolyType};
    use num_bigint::BigUint;
    use rand::thread_rng;

    #[test]
    fn key_switching_works() {
        let params = BfvParameters::default(6, 1 << 4);
        let ksk_ctx = params.poly_ctx(&PolyType::Q, 0);

        let mut rng = thread_rng();

        let sk = SecretKey::random(params.degree, &mut rng);

        let poly = ksk_ctx.random(Representation::Evaluation, &mut rng);
        let ksk = BVKeySwitchingKey::new(&poly, &sk, &ksk_ctx, &mut rng);

        let mut other_poly = ksk_ctx.random(Representation::Coefficient, &mut rng);

        let now = std::time::Instant::now();
        let cs = ksk.switch(&other_poly, &ksk_ctx);
        println!("Time elapsed: {:?}", now.elapsed());

        let mut sk_poly =
            ksk_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        ksk_ctx.change_representation(&mut sk_poly, Representation::Evaluation);
        let mut res = ksk_ctx.add(&cs.0, &ksk_ctx.mul(&cs.1, &sk_poly));

        // expected
        ksk_ctx.change_representation(&mut other_poly, Representation::Evaluation);
        let expected_poly = ksk_ctx.mul(&other_poly, &poly);

        let mut diff = ksk_ctx.sub(&res, &expected_poly);
        ksk_ctx.change_representation(&mut diff, Representation::Coefficient);

        izip!(ksk_ctx.try_convert_to_biguint(&diff)).for_each(|v| {
            let diff_bits = std::cmp::min(v.bits(), (ksk_ctx.big_q() - v).bits());
            assert!(diff_bits <= 70);
        });
    }

    #[test]
    fn hybrid_key_switching() {
        let params = BfvParameters::default(5, 1 << 6);
        let ksk_ctx = params.poly_ctx(&PolyType::Q, 0);
        let specialp_ctx = params.poly_ctx(&PolyType::SpecialP, 0);
        let qp_ctx = params.poly_ctx(&PolyType::QP, 0);

        let mut rng = thread_rng();

        let sk = SecretKey::random(params.degree, &mut rng);

        let poly = ksk_ctx.random(Representation::Evaluation, &mut rng);
        let ksk = HybridKeySwitchingKey::new(
            &poly,
            &sk,
            &ksk_ctx,
            &specialp_ctx,
            &qp_ctx,
            params.alpha,
            params.aux_bits,
            &mut rng,
        );

        let mut other_poly = ksk_ctx.random(Representation::Coefficient, &mut rng);
        let now = std::time::Instant::now();
        let cs = ksk.switch(&other_poly, &qp_ctx, &ksk_ctx, &specialp_ctx);
        println!("Time elapsed: {:?}", now.elapsed());

        let mut sk_poly =
            ksk_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        ksk_ctx.change_representation(&mut sk_poly, Representation::Evaluation);
        let mut res = ksk_ctx.add(&cs.0, &ksk_ctx.mul(&cs.1, &sk_poly));

        // expected
        ksk_ctx.change_representation(&mut other_poly, Representation::Evaluation);
        let expected_poly = ksk_ctx.mul(&other_poly, &poly);

        let mut diff = ksk_ctx.sub(&res, &expected_poly);
        ksk_ctx.change_representation(&mut diff, Representation::Coefficient);

        izip!(ksk_ctx.try_convert_to_biguint(&diff)).for_each(|v| {
            let diff_bits = std::cmp::min(v.bits(), (ksk_ctx.big_q() - v).bits());
            dbg!(&diff_bits);
        });
    }
}
