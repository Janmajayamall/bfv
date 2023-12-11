use crate::{
    BfvParameters, Ciphertext, Encoding, HybridKeySwitchingKey, Plaintext, Poly, PolyType,
    PublicKey, Representation, SecretKey,
};
use itertools::{izip, Itertools};
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

struct CRP();

type CRS = [u8; 32];

struct CollectivePublicKeyGenerator {}

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

struct CollectiveDecryption();

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

struct CollectiveRlkGeneratorState(SecretKey);

struct CollectiveRlkGenerator();

impl CollectiveRlkGenerator {
    // Generates public input vector `a` of size `dnum` values with `crs` as seed.
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
        crs: CRS,
        level: usize,
        rng: &mut R,
    ) -> (Vec<Poly>, Vec<Poly>, CollectiveRlkGeneratorState) {
        let a_values = CollectiveRlkGenerator::generate_public_inputs(params, crs, level);

        let ksk_params = params.hybrid_key_switching_params_at_level(level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        // ephemeral secret
        let u_i = SecretKey::random_with_params(&params, rng);

        // `HybridKeySwitchingKey::generate_c0` expects `poly` to be multiplied with in key switching to have context of ciphertext polynomial at level
        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let mut s_i_poly =
            q_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        q_ctx.change_representation(&mut s_i_poly, Representation::Evaluation);

        // h_{0,i}
        let h0s = HybridKeySwitchingKey::generate_c0(
            &qp_ctx,
            &a_values,
            &ksk_params.g,
            &s_i_poly,
            &u_i,
            params.variance,
            rng,
        );

        // h_{1,i}
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

        (h0s, h1s, CollectiveRlkGeneratorState(u_i))
    }

    pub fn aggregate_share_1() {
        todo!()
    }

    pub fn generate_share_2<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        sk: &SecretKey,
        h0_agg: &Poly,
        h1_agg: &Poly,
        state: &CollectiveRlkGeneratorState,
        level: usize,
        rng: &mut R,
    ) -> (Poly, Poly) {
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        let mut s_i =
            qp_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        qp_ctx.change_representation(&mut s_i, Representation::Evaluation);

        // h1'_i = (u_i - s_i) * \sum h1_i + e_2
        let mut u_i =
            qp_ctx.try_convert_from_i64_small(&state.0.coefficients, Representation::Coefficient);
        qp_ctx.change_representation(&mut u_i, Representation::Evaluation);
        qp_ctx.sub_assign(&mut u_i, &s_i);
        let mut h1_dash_i = u_i;
        qp_ctx.mul_assign(&mut h1_dash_i, &h1_agg);
        let mut e_2 = qp_ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
        qp_ctx.change_representation(&mut e_2, Representation::Evaluation);
        qp_ctx.add_assign(&mut h1_dash_i, &e_2);

        // h0'_i = s_i * \sum h0_i + e_1
        qp_ctx.mul_assign(&mut s_i, h0_agg);
        let mut h0_dash_i = s_i;
        let mut e_1 = qp_ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
        qp_ctx.change_representation(&mut e_1, Representation::Evaluation);
        qp_ctx.add_assign(&mut h0_dash_i, &e_1);

        (h0_dash_i, h1_dash_i)
    }

    pub fn aggrgegate_share_2_and_finalise() {}
}

struct MHE {}

struct Party {
    secret: SecretKey,
}

struct MHEDebugger {}

impl MHEDebugger {
    pub unsafe fn measure_noise(parties: &[Party], params: &BfvParameters, ct: &Ciphertext) -> u64 {
        let q_ctx = params.poly_ctx(&PolyType::Q, ct.level());

        // Calculate ideal secret key
        // s_{ideal} = \sum s_i
        let mut s_ideal = q_ctx.zero(Representation::Evaluation);
        parties.iter().for_each(|party_i| {
            let mut sk = q_ctx.try_convert_from_i64_small(
                &party_i.secret.coefficients,
                Representation::Coefficient,
            );
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

    use crate::Encoding;

    use super::*;

    fn setup_parties(params: &BfvParameters, n: usize) -> Vec<Party> {
        let mut rng = thread_rng();
        (0..n)
            .into_iter()
            .map(|_| {
                let sk = SecretKey::random_with_params(params, &mut rng);
                Party { secret: sk }
            })
            .collect_vec()
    }

    fn gen_crs() -> CRS {
        let mut rng = thread_rng();
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        seed
    }

    #[test]
    fn multi_party_encryption_decryption_works() {
        let no_of_parties = 10;
        let params = BfvParameters::default(10, 1 << 6);

        let parties = setup_parties(&params, no_of_parties);

        // Generate collective public key
        let crs = gen_crs();
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

        let public_key =
            CollectivePublicKeyGenerator::aggregate_shares_and_finalise(&params, &shares, crs);

        // Encrypt message
        let m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let pt = Plaintext::encode(&m, &params, Encoding::default());
        let ct = public_key.encrypt(&params, &pt, &mut rng);

        unsafe {
            dbg!(MHEDebugger::measure_noise(&parties, &params, &ct));
        }

        // Distributed decryption
        let shares = parties
            .iter()
            .map(|party_i| {
                CollectiveDecryption::generate_share(&params, &ct, &party_i.secret, &mut rng)
            })
            .collect_vec();
        let m_back: Vec<u64> =
            CollectiveDecryption::aggregate_share_and_decrypt(&params, &ct, &shares)
                .decode(Encoding::default(), &params);
        assert_eq!(m, m_back);
    }
}
