use crate::{
    BfvParameters, Ciphertext, Encoding, GaloisKey, HybridKeySwitchingKey, Plaintext, Poly,
    PolyType, PublicKey, RelinearizationKey, Representation, SecretKey, Substitution,
};
use itertools::{izip, Itertools};
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

type CRS = [u8; 32];

pub struct CollectivePublicKeyGenerator {}

impl CollectivePublicKeyGenerator {
    pub fn generate_share<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        sk: &SecretKey,
        crs: CRS,
        rng: &mut R,
    ) -> Poly {
        let qr = params.poly_ctx(&PolyType::Qr, 0);

        // sample common reference polynomial for cpk `c1`
        let mut crs_prng = ChaCha8Rng::from_seed(crs);
        let c1 = qr.random(Representation::Evaluation, &mut crs_prng);

        let mut s = qr.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        qr.change_representation(&mut s, Representation::Evaluation);

        // s_i*c1 + e_i
        qr.mul_assign(&mut s, &c1);
        let mut e = qr.random_gaussian(Representation::Coefficient, params.variance, rng);
        qr.change_representation(&mut e, Representation::Evaluation);
        qr.add_assign(&mut s, &e);

        s
    }

    pub fn aggregate_shares_and_finalise(
        params: &BfvParameters,
        shares: &[Poly],
        crs: CRS,
    ) -> PublicKey {
        let qr = params.poly_ctx(&PolyType::Qr, 0);

        let mut sum = shares[0].clone();
        shares
            .iter()
            .skip(1)
            .for_each(|share_i| qr.add_assign(&mut sum, &share_i));

        // generate c1 (TODO: is this necessary?)
        let mut crs_prng = ChaCha8Rng::from_seed(crs);
        let mut c1 = qr.random(Representation::Evaluation, &mut crs_prng);
        qr.neg_assign(&mut c1);

        PublicKey {
            c0: sum,
            c1,
            seed: crs,
        }
    }
}
pub struct CollectiveDecryption();

impl CollectiveDecryption {
    pub fn generate_share<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        ct: &Ciphertext,
        sk: &SecretKey,
        rng: &mut R,
    ) -> Poly {
        assert!(
            ct.c.len() == 2,
            "For collective decryption, ciphertext must be of degree 1"
        );

        let q_ctx = params.poly_ctx(&PolyType::Q, ct.level());

        // s_i * c_1 + e_1
        let mut s = q_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        q_ctx.change_representation(&mut s, Representation::Evaluation);
        if ct.c[1].representation() == &Representation::Evaluation {
            q_ctx.mul_assign(&mut s, &ct.c[1]);
        } else {
            let mut c1 = ct.c[1].clone();
            q_ctx.change_representation(&mut c1, Representation::Evaluation);
            q_ctx.mul_assign(&mut s, &c1);
        }
        q_ctx.change_representation(&mut s, Representation::Coefficient);

        // TODO: increase smudging noise
        // s_i * c_1 + e_1
        let e = q_ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
        q_ctx.add_assign(&mut s, &e);

        s
    }

    pub fn aggregate_share_and_decrypt(
        params: &BfvParameters,
        ct: &Ciphertext,
        shares: &[Poly],
    ) -> Plaintext {
        let q_ctx = params.poly_ctx(&PolyType::Q, ct.level());

        let mut sum_of_shares = shares[0].clone();
        shares.iter().skip(1).for_each(|share_i| {
            q_ctx.add_assign(&mut sum_of_shares, share_i);
        });

        // \sum s_i * c1
        let mut delta_m = sum_of_shares;

        // \delta * m + E = c0 + \sum s_i * c1
        if ct.c[0].representation() == &Representation::Coefficient {
            q_ctx.add_assign(&mut delta_m, &ct.c[0]);
        } else {
            let mut c0 = ct.c[0].clone();
            q_ctx.change_representation(&mut c0, Representation::Coefficient);
            q_ctx.add_assign(&mut delta_m, &c0);
        }

        // [round((t \Delta m)/Ql)]_t
        q_ctx.change_representation(&mut delta_m, Representation::Coefficient);
        let m = q_ctx.scale_and_round_decryption(
            &delta_m,
            &params.plaintext_modulus_op,
            params.max_bit_size_by2,
            &params.t_ql_hat_inv_modql_divql_modt[ct.level],
            &params.t_bql_hat_inv_modql_divql_modt[ct.level],
            &params.t_ql_hat_inv_modql_divql_frac[ct.level],
            &params.t_bql_hat_inv_modql_divql_frac[ct.level],
        );

        Plaintext {
            m,
            encoding: None,
            mul_poly: None,
            add_sub_poly: None,
        }
    }
}

pub struct CollectiveRlkGeneratorState(SecretKey);

pub struct CollectiveRlkGenerator();

impl CollectiveRlkGenerator {
    pub fn init_state<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        rng: &mut R,
    ) -> CollectiveRlkGeneratorState {
        CollectiveRlkGeneratorState(SecretKey::random_with_params(&params, rng))
    }

    /// Generates public input vector `a` of size `dnum` polynomials with `crs` as seed.
    /// `crs` is common reference string shared among all parties
    fn generate_public_inputs(params: &BfvParameters, crs: CRS, level: usize) -> Vec<Poly> {
        let ksk_params = params.hybrid_key_switching_params_at_level(level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        let mut crs_prng = ChaCha8Rng::from_seed(crs);
        (0..ksk_params.dnum)
            .into_iter()
            .map(|_| qp_ctx.random(Representation::Evaluation, &mut crs_prng))
            .collect_vec()
    }

    pub fn generate_share_1<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        sk: &SecretKey,
        state: &CollectiveRlkGeneratorState,
        crs: CRS,
        level: usize,
        rng: &mut R,
    ) -> (Vec<Poly>, Vec<Poly>) {
        let a_values = CollectiveRlkGenerator::generate_public_inputs(params, crs, level);

        let ksk_params = params.hybrid_key_switching_params_at_level(level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        // ephemeral secret
        let u_i = &state.0;

        // `HybridKeySwitchingKey::generate_c0` expects `poly` to be multiplied with in key switching to have context of ciphertext polynomial at level
        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let mut s_i_poly =
            q_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        q_ctx.change_representation(&mut s_i_poly, Representation::Evaluation);

        // [h_{0,i}]
        let h0s = HybridKeySwitchingKey::generate_c0(
            &qp_ctx,
            &a_values,
            &ksk_params.g,
            &s_i_poly,
            u_i,
            params.variance,
            rng,
        );

        // [h_{1,i}]
        // [s_i]_QP (TODO: Since QP is union of Q and P, we are unecessarily reducing s_i coefficients by prime q_0, ... q_{l-1} twice)
        let mut s_i_poly =
            qp_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        qp_ctx.change_representation(&mut s_i_poly, Representation::Evaluation);
        // s_i * a + e
        let h1s = a_values
            .iter()
            .map(|a| {
                let mut tmp = qp_ctx.mul(&s_i_poly, a);
                let mut e =
                    qp_ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
                qp_ctx.change_representation(&mut e, Representation::Evaluation);
                qp_ctx.add_assign(&mut tmp, &e);
                tmp
            })
            .collect_vec();

        (h0s, h1s)
    }

    pub fn aggregate_shares_1(
        params: &BfvParameters,
        h0s: &[Vec<Poly>],
        h1s: &[Vec<Poly>],
        level: usize,
    ) -> (Vec<Poly>, Vec<Poly>) {
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        let mut h0_agg = h0s[0].to_owned();
        h0s.iter().skip(1).for_each(|shares_i| {
            izip!(h0_agg.iter_mut(), shares_i.iter()).for_each(|(s0, s1)| {
                qp_ctx.add_assign(s0, s1);
            });
        });

        let mut h1_agg = h1s[0].to_owned();
        h1s.iter().skip(1).for_each(|shares_i| {
            izip!(h1_agg.iter_mut(), shares_i.iter()).for_each(|(s0, s1)| {
                qp_ctx.add_assign(s0, s1);
            });
        });

        (h0_agg, h1_agg)
    }

    pub fn generate_share_2<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        sk: &SecretKey,
        h0_agg: &[Poly],
        h1_agg: &[Poly],
        state: &CollectiveRlkGeneratorState,
        level: usize,
        rng: &mut R,
    ) -> (Vec<Poly>, Vec<Poly>) {
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        let mut s_i =
            qp_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        qp_ctx.change_representation(&mut s_i, Representation::Evaluation);

        let mut u_i =
            qp_ctx.try_convert_from_i64_small(&state.0.coefficients, Representation::Coefficient);
        qp_ctx.change_representation(&mut u_i, Representation::Evaluation);

        let (h0_dash, h1_dash): (Vec<Poly>, Vec<Poly>) = izip!(h0_agg.iter(), h1_agg.iter())
            .map(|(h0, h1)| {
                // h1'_i = (u_i - s_i) * \sum h1_i + e_2
                let mut h1_dash_i = qp_ctx.sub(&u_i, &s_i);
                qp_ctx.mul_assign(&mut h1_dash_i, &h1);
                let mut e_2 =
                    qp_ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
                qp_ctx.change_representation(&mut e_2, Representation::Evaluation);
                qp_ctx.add_assign(&mut h1_dash_i, &e_2);

                // h0'_i = s_i * \sum h0_i + e_1
                let mut h0_dash_i = qp_ctx.mul(&s_i, h0);
                let mut e_1 =
                    qp_ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
                qp_ctx.change_representation(&mut e_1, Representation::Evaluation);
                qp_ctx.add_assign(&mut h0_dash_i, &e_1);

                (h0_dash_i, h1_dash_i)
            })
            .unzip();

        (h0_dash, h1_dash)
    }

    pub fn aggregate_shares_2(
        params: &BfvParameters,
        h0_dash_shares: &[Vec<Poly>],
        h1_dash_shares: &[Vec<Poly>],
        h1_agg: Vec<Poly>,
        level: usize,
    ) -> RelinearizationKey {
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        // h0'+h1'
        let mut h0_dash_h1_dash_agg = h0_dash_shares[0].to_owned();
        h0_dash_shares.iter().skip(1).for_each(|shares_i| {
            izip!(h0_dash_h1_dash_agg.iter_mut(), shares_i.iter()).for_each(|(s0, s1)| {
                qp_ctx.add_assign(s0, s1);
            });
        });
        h1_dash_shares.iter().for_each(|shares_i| {
            izip!(h0_dash_h1_dash_agg.iter_mut(), shares_i.iter()).for_each(|(s0, s1)| {
                qp_ctx.add_assign(s0, s1);
            });
        });

        RelinearizationKey {
            ksk: HybridKeySwitchingKey {
                seed: None,
                c0s: h0_dash_h1_dash_agg.into_boxed_slice(),
                c1s: h1_agg.into_boxed_slice(),
            },
            level,
        }
    }
}

#[derive(Clone)]
pub struct CollectiveGaloisKeyGeneratorState {
    c1s: Vec<Poly>,
    substitution: Substitution,
    crs: CRS,
}

pub struct CollectiveGaloisKeyGenerator();
impl CollectiveGaloisKeyGenerator {
    pub fn init_state(
        params: &BfvParameters,
        exponent: usize,
        crs: CRS,
        level: usize,
    ) -> CollectiveGaloisKeyGeneratorState {
        let ksk_params = params.hybrid_key_switching_params_at_level(level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        let c1s = HybridKeySwitchingKey::generate_c1(ksk_params.dnum, &qp_ctx, crs);

        // substitution map
        let substitution = Substitution::new(exponent, params.degree);

        CollectiveGaloisKeyGeneratorState {
            c1s,
            substitution,
            crs,
        }
    }

    pub fn generate_share_1<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        sk: &SecretKey,
        state: &CollectiveGaloisKeyGeneratorState,
        level: usize,
        rng: &mut R,
    ) -> Vec<Poly> {
        let ksk_params = params.hybrid_key_switching_params_at_level(level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        // Substitute secret key
        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let mut sk_poly =
            q_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        q_ctx.change_representation(&mut sk_poly, Representation::Evaluation);
        let sk_poly = q_ctx.substitute(&sk_poly, &state.substitution);

        let c0s = HybridKeySwitchingKey::generate_c0(
            &qp_ctx,
            &state.c1s,
            &ksk_params.g,
            &sk_poly,
            sk,
            params.variance,
            rng,
        );

        c0s
    }

    pub fn aggregate_shares_1_and_finalise(
        params: &BfvParameters,
        share1_c0s: &[Vec<Poly>],
        state: CollectiveGaloisKeyGeneratorState,
        level: usize,
    ) -> GaloisKey {
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);
        let mut c0s_agg = share1_c0s[0].clone();
        share1_c0s.iter().skip(1).for_each(|share_i_c0s| {
            izip!(c0s_agg.iter_mut(), share_i_c0s.iter()).for_each(|(c0_sum, c0)| {
                qp_ctx.add_assign(c0_sum, c0);
            })
        });

        let ksk_key = HybridKeySwitchingKey {
            seed: Some(state.crs),
            c0s: c0s_agg.into_boxed_slice(),
            c1s: state.c1s.into_boxed_slice(),
        };

        GaloisKey {
            substitution: state.substitution,
            ksk_key,
            level,
        }
    }
}

pub struct MHEDebugger {}

impl MHEDebugger {
    pub unsafe fn measure_noise(
        parties: &[SecretKey],
        params: &BfvParameters,
        ct: &Ciphertext,
    ) -> u64 {
        let q_ctx = params.poly_ctx(&PolyType::Q, ct.level());

        // Calculate ideal secret key
        // s_{ideal} = \sum s_i
        let mut s_ideal = q_ctx.zero(Representation::Evaluation);
        parties.iter().for_each(|party_i| {
            let mut sk = q_ctx
                .try_convert_from_i64_small(&party_i.coefficients, Representation::Coefficient);
            q_ctx.change_representation(&mut sk, Representation::Evaluation);
            q_ctx.add_assign(&mut s_ideal, &sk);
        });

        // Decrypt ct using ideal secret key
        // delta_m = c_0
        let mut delta_m = ct.c[0].clone();
        q_ctx.change_representation(&mut delta_m, Representation::Evaluation);
        let s_ideal_clone = s_ideal.clone();
        for i in 1..ct.c.len() {
            // delta_m += c_i * (s_i)^i
            if ct.c[i].representation() == &Representation::Evaluation {
                let tmp = q_ctx.mul(&ct.c[i], &s_ideal);
                q_ctx.add_assign(&mut delta_m, &tmp);
            } else {
                let mut tmp = ct.c[i].clone();
                q_ctx.change_representation(&mut tmp, Representation::Evaluation);
                q_ctx.mul_assign(&mut tmp, &s_ideal);
                q_ctx.add_assign(&mut delta_m, &tmp);
            }
            q_ctx.mul_assign(&mut s_ideal, &s_ideal_clone);
        }

        q_ctx.change_representation(&mut delta_m, Representation::Coefficient);
        let m = q_ctx.scale_and_round_decryption(
            &delta_m,
            &params.plaintext_modulus_op,
            params.max_bit_size_by2,
            &params.t_ql_hat_inv_modql_divql_modt[ct.level],
            &params.t_bql_hat_inv_modql_divql_modt[ct.level],
            &params.t_ql_hat_inv_modql_divql_frac[ct.level],
            &params.t_bql_hat_inv_modql_divql_frac[ct.level],
        );
        // \delta * m (without error)
        let scaled_m = Plaintext::scale_m(
            &m,
            &params,
            &Encoding::simd(ct.level(), crate::PolyCache::None),
            Representation::Coefficient,
        );

        // \delta * m + E - \delta * m = E
        let diff_poly = q_ctx.sub(&delta_m, &scaled_m);
        let mut noise = 0u64;
        q_ctx
            .try_convert_to_biguint(&diff_poly)
            .iter()
            .for_each(|v| {
                noise = std::cmp::max(noise, std::cmp::min(v.bits(), (q_ctx.big_q() - v).bits()))
            });
        noise
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::thread_rng;

    use crate::{rot_to_galois_element, Encoding, EvaluationKey, Evaluator};

    use super::*;

    pub struct PartySecret {
        secret: SecretKey,
    }

    fn setup_parties(params: &BfvParameters, n: usize) -> Vec<PartySecret> {
        let mut rng = thread_rng();
        (0..n)
            .into_iter()
            .map(|_| {
                let sk = SecretKey::random_with_params(params, &mut rng);
                PartySecret { secret: sk }
            })
            .collect_vec()
    }

    fn gen_crs() -> CRS {
        let mut rng = thread_rng();
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        seed
    }

    fn gen_collective_public_key(
        params: &BfvParameters,
        parties: &[PartySecret],
        crs: CRS,
    ) -> PublicKey {
        let mut rng = thread_rng();
        let shares = parties
            .iter()
            .map(|party_i| {
                CollectivePublicKeyGenerator::generate_share(
                    &params,
                    &party_i.secret,
                    crs,
                    &mut rng,
                )
            })
            .collect_vec();

        CollectivePublicKeyGenerator::aggregate_shares_and_finalise(&params, &shares, crs)
    }

    fn collective_decryption(
        params: &BfvParameters,
        parties: &[PartySecret],
        ct: &Ciphertext,
    ) -> Vec<u64> {
        let mut rng = thread_rng();
        let shares = parties
            .iter()
            .map(|party_i| {
                CollectiveDecryption::generate_share(&params, &ct, &party_i.secret, &mut rng)
            })
            .collect_vec();

        CollectiveDecryption::aggregate_share_and_decrypt(&params, &ct, &shares)
            .decode(Encoding::default(), &params)
    }

    #[test]
    fn multi_party_encryption_decryption_works() {
        let no_of_parties = 10;
        let params = BfvParameters::default(10, 1 << 6);

        let parties = setup_parties(&params, no_of_parties);

        // Generate collective public key
        let crs = gen_crs();
        let public_key = gen_collective_public_key(&params, &parties, crs);

        // Encrypt message
        let mut rng = thread_rng();
        let m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let pt = Plaintext::encode(&m, &params, Encoding::default());
        let ct = public_key.encrypt(&params, &pt, &mut rng);

        unsafe {
            let secrets = parties.iter().map(|s| s.secret.clone()).collect_vec();
            dbg!(MHEDebugger::measure_noise(&secrets, &params, &ct));
        }

        // Distributed decryption
        let m_back = collective_decryption(&params, &parties, &ct);
        assert_eq!(m, m_back);
    }

    #[test]
    fn collective_rlk_key_generation_works() {
        let no_of_parties = 10;
        let params = BfvParameters::default(3, 1 << 15);
        let level = 0;
        let parties = setup_parties(&params, no_of_parties);

        // Generate RLK //

        // initialise state
        let mut rng = thread_rng();
        let collective_rlk_state = parties
            .iter()
            .map(|party_i| CollectiveRlkGenerator::init_state(&params, &mut rng))
            .collect_vec();

        // Generate and collect h0s and h1s
        let crs = gen_crs();
        let mut h0s = vec![];
        let mut h1s = vec![];
        izip!(parties.iter(), collective_rlk_state.iter()).for_each(|(party_i, state_i)| {
            let (h0_i, h1_i) = CollectiveRlkGenerator::generate_share_1(
                &params,
                &party_i.secret,
                state_i,
                crs,
                level,
                &mut rng,
            );
            h0s.push(h0_i);
            h1s.push(h1_i);
        });

        // aggregate h0s and h1s
        let (h0s_agg, h1s_agg) = CollectiveRlkGenerator::aggregate_shares_1(
            &params,
            h0s.as_slice(),
            h1s.as_slice(),
            level,
        );

        // Generate and collect h'0s h'1s
        let mut h_dash_0s = vec![];
        let mut h_dash_1s = vec![];
        izip!(parties.iter(), collective_rlk_state.iter()).for_each(|(party_i, state_i)| {
            let (h_dash_0_i, h_dash_1_i) = CollectiveRlkGenerator::generate_share_2(
                &params,
                &party_i.secret,
                &h0s_agg,
                &h1s_agg,
                state_i,
                level,
                &mut rng,
            );
            h_dash_0s.push(h_dash_0_i);
            h_dash_1s.push(h_dash_1_i);
        });

        // aggregate h'0s and h'1s
        let rlk = CollectiveRlkGenerator::aggregate_shares_2(
            &params, &h_dash_0s, &h_dash_1s, h1s_agg, level,
        );

        // Generate public key //
        let crs = gen_crs();
        let public_key = gen_collective_public_key(&params, &parties, crs);

        // Encryt two plaintexts
        let mut rng = thread_rng();
        let m0 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let m1 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let pt0 = Plaintext::encode(&m0, &params, Encoding::default());
        let pt1 = Plaintext::encode(&m1, &params, Encoding::default());
        let ct0 = public_key.encrypt(&params, &pt0, &mut rng);
        let ct1 = public_key.encrypt(&params, &pt1, &mut rng);

        // multiply ciphertexts
        let evaluation_key = EvaluationKey::new_raw(&[0], vec![rlk], &[], &[], vec![]);
        let evaluator = Evaluator::new(params);
        let ct0c1 = evaluator.mul(&ct0, &ct1);
        let ct_out = evaluator.relinearize(&ct0c1, &evaluation_key);

        unsafe {
            let secrets = parties.iter().map(|s| s.secret.clone()).collect_vec();
            dbg!(MHEDebugger::measure_noise(
                &secrets,
                evaluator.params(),
                &ct0c1
            ));
            dbg!(MHEDebugger::measure_noise(
                &secrets,
                evaluator.params(),
                &ct_out
            ));
            dbg!(MHEDebugger::measure_noise(
                &secrets,
                evaluator.params(),
                &ct0
            ));
            dbg!(MHEDebugger::measure_noise(
                &secrets,
                evaluator.params(),
                &ct1
            ));
        }

        // decrypt ct_out
        let m0m1 = collective_decryption(evaluator.params(), &parties, &ct_out);
        let mut m0m1_expected = m0;
        evaluator
            .params()
            .plaintext_modulus_op
            .mul_mod_fast_vec(&mut m0m1_expected, &m1);

        assert_eq!(m0m1, m0m1_expected);
    }

    #[test]
    fn collective_galois_key_rotation_works() {
        let no_of_parties = 10;
        let params = BfvParameters::default(6, 1 << 15);
        let level = 0;
        let parties = setup_parties(&params, no_of_parties);

        // Generate Galois key //

        // Rotate left by 1
        let exponent = rot_to_galois_element(1, params.degree);

        let crs = gen_crs();

        // initialise state
        let mut rng = thread_rng();
        let collective_galois_state = parties
            .iter()
            .map(|party_i| CollectiveGaloisKeyGenerator::init_state(&params, exponent, crs, level))
            .collect_vec();

        // Generate and collect h0s and h1s
        let share1_c0s = izip!(parties.iter(), collective_galois_state.iter())
            .map(|(party_i, state_i)| {
                CollectiveGaloisKeyGenerator::generate_share_1(
                    &params,
                    &party_i.secret,
                    state_i,
                    level,
                    &mut rng,
                )
            })
            .collect_vec();

        // aggregate share1_c0s and finalise
        let state = collective_galois_state[0].clone();
        let galois_key = CollectiveGaloisKeyGenerator::aggregate_shares_1_and_finalise(
            &params,
            &share1_c0s,
            state,
            level,
        );

        // Generate public key //
        let crs = gen_crs();
        let public_key = gen_collective_public_key(&params, &parties, crs);

        // Encryt a message
        let mut rng = thread_rng();
        let mut m0 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let pt0 = Plaintext::encode(&m0, &params, Encoding::default());
        let ct0 = public_key.encrypt(&params, &pt0, &mut rng);

        // multiply ciphertexts
        let evaluation_key = EvaluationKey::new_raw(&[], vec![], &[level], &[1], vec![galois_key]);
        let evaluator = Evaluator::new(params);
        let ct0_rotated = evaluator.rotate(&ct0, 1, &evaluation_key);

        unsafe {
            let secrets = parties.iter().map(|s| s.secret.clone()).collect_vec();

            dbg!(MHEDebugger::measure_noise(
                &secrets,
                evaluator.params(),
                &ct0_rotated
            ));
            dbg!(MHEDebugger::measure_noise(
                &secrets,
                evaluator.params(),
                &ct0
            ));
        }

        // decrypt ct_out
        let m0_rotated = collective_decryption(evaluator.params(), &parties, &ct0_rotated);

        let len = m0.len();
        let (m0_first, m0_last) = m0.split_at_mut(len / 2);
        m0_first.rotate_left(1);
        m0_last.rotate_left(1);

        assert_eq!(m0, m0_rotated);
    }
}
