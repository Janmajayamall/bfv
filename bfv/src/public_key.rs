use crate::{BfvParameters, Ciphertext, Plaintext, Poly, PolyType, Representation, SecretKey};
use rand::{CryptoRng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

struct PublicKey {
    c0: Poly,
    c1: Poly,
    seed: [u8; 32],
}

impl PublicKey {
    /// Generates c1 poly in basis Qr in Evaluation representation
    pub fn gen_c1_with_seed(params: &BfvParameters, seed: &[u8; 32]) -> Poly {
        let mut prng = ChaCha8Rng::from_seed(seed.clone());
        let qr_ctx = params.poly_ctx(&PolyType::Qr, 0);
        qr_ctx.random(Representation::Evaluation, &mut prng)
    }

    pub fn new<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        sk: &SecretKey,
        rng: &mut R,
    ) -> PublicKey {
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);

        let qr_ctx = params.poly_ctx(&PolyType::Qr, 0);
        let mut s =
            qr_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        qr_ctx.change_representation(&mut s, Representation::Evaluation);

        let mut c1 = PublicKey::gen_c1_with_seed(params, &seed);

        // c1 * s
        qr_ctx.mul_assign(&mut s, &c1);
        let mut c1_s = s;

        // c1*s + e
        let mut e = qr_ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
        qr_ctx.change_representation(&mut e, Representation::Evaluation);
        qr_ctx.add_assign(&mut c1_s, &e);

        // -c1
        qr_ctx.neg_assign(&mut c1);

        PublicKey { c0: c1_s, c1, seed }
    }

    pub fn encrypt<R: CryptoRng + RngCore>(
        &self,
        params: &BfvParameters,
        pt: &Plaintext,
        rng: &mut R,
    ) -> Ciphertext {
        assert!(pt.encoding.is_some(), "Plaintext encoding is missing!");
        assert!(
            pt.encoding.as_ref().unwrap().level == 0,
            "Pke plaintext must be at level 0"
        );

        let pke_params = params.pke_parameters.as_ref().expect("Pke is disabled");

        // [Qr*m]_t
        let mut m = pt.m.clone();
        params
            .plaintext_modulus_op
            .scalar_mul_mod_fast_vec(&mut m, pke_params.qr_mod_t);

        // [[Qr*m]_t * (-t)^{-1}]_Qr
        let qr_ctx = params.poly_ctx(&PolyType::Qr, 0);
        let mut m_poly = qr_ctx.try_convert_from_u64(&m, Representation::Coefficient);
        qr_ctx.change_representation(&mut m_poly, Representation::Evaluation);
        qr_ctx.mul_assign(&mut m_poly, &pke_params.neg_t_inv_modqr);

        // Generate zero encryption in basis Qr
        // Sample ephemeral secret u
        let u = SecretKey::random_with_params(&params, rng);
        let mut u_poly =
            qr_ctx.try_convert_from_i64_small(&u.coefficients, Representation::Coefficient);
        qr_ctx.change_representation(&mut u_poly, Representation::Evaluation);

        // u * pk0 + e0
        let mut u_pk0 = qr_ctx.mul(&u_poly, &self.c0);
        let mut e0 = qr_ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
        qr_ctx.change_representation(&mut e0, Representation::Evaluation);
        qr_ctx.add_assign(&mut u_pk0, &e0);

        // u * pk1 + e1
        qr_ctx.mul_assign(&mut u_poly, &self.c1);
        let mut e1 = qr_ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
        qr_ctx.change_representation(&mut e1, Representation::Evaluation);
        qr_ctx.add_assign(&mut u_poly, &e1);

        // Zero ciphertext in basis Qr
        let mut c0 = u_pk0;
        let mut c1 = u_poly;

        // Encrypt message m by adding m to zero ciphertext
        qr_ctx.add_assign(&mut c0, &m_poly);

        // Scale and round ciphertext polynomials by 1/r and change basis to Q to reduce error
        qr_ctx.change_representation(&mut c0, Representation::Coefficient);
        qr_ctx.change_representation(&mut c1, Representation::Coefficient);

        qr_ctx.mod_down_next(&mut c0, &pke_params.r_inv_modq);
        qr_ctx.mod_down_next(&mut c1, &pke_params.r_inv_modq);

        Ciphertext {
            c: vec![c0, c1],
            poly_type: PolyType::Q,
            seed: None,
            level: 0,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::Encoding;
    use rand::thread_rng;

    #[test]
    fn pke_works() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(5, 1 << 6);

        let sk = SecretKey::random_with_params(&params, &mut rng);
        let pk = PublicKey::new(&params, &sk, &mut rng);

        let m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let pt = Plaintext::encode(&m, &params, Encoding::default());

        let ct = pk.encrypt(&params, &pt, &mut rng);

        let m_back: Vec<u64> = sk
            .decrypt(&ct, &params)
            .decode(Encoding::default(), &params);

        assert_eq!(m, m_back);
    }
}
