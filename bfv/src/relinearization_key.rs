use crate::{
    BfvParameters, Ciphertext, HybridKeySwitchingKey, Poly, PolyContext, PolyType, Representation,
    SecretKey,
};
use rand::{CryptoRng, RngCore};
use std::sync::Arc;
use traits::Ntt;

pub struct RelinearizationKey {
    ksk: HybridKeySwitchingKey,
    level: usize,
}

impl RelinearizationKey {
    pub fn new<T: Ntt, R: CryptoRng + RngCore>(
        params: &BfvParameters<T>,
        sk: &SecretKey,
        level: usize,
        rng: &mut R,
    ) -> RelinearizationKey {
        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);
        let specialp_ctx = params.poly_ctx(&PolyType::SpecialP, level);

        let mut sk_poly =
            q_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        q_ctx.change_representation(&mut sk_poly, Representation::Evaluation);

        // sk^2
        let sk_sq = q_ctx.mul(&sk_poly, &sk_poly);

        // Key switching key
        let ksk = HybridKeySwitchingKey::new(
            &sk_sq,
            sk,
            &q_ctx,
            &specialp_ctx,
            &qp_ctx,
            params.alpha,
            params.aux_bits,
            rng,
        );

        RelinearizationKey { ksk, level }
    }

    pub fn relinearize<T: Ntt>(&self, ct: &Ciphertext, params: &BfvParameters<T>) -> Ciphertext {
        assert!(ct.c.len() == 3); // otherwise invalid relinerization
        assert!(ct.c[0].representation == Representation::Coefficient);
        assert!(ct.level == self.level);

        let level = ct.level;
        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);
        let specialp_ctx = params.poly_ctx(&PolyType::SpecialP, level);

        let (mut cs0, mut cs1) = self.ksk.switch(&ct.c[3], &qp_ctx, &q_ctx, &specialp_ctx);
        q_ctx.change_representation(&mut cs0, Representation::Coefficient);
        q_ctx.change_representation(&mut cs1, Representation::Coefficient);

        q_ctx.add_assign(&mut cs0, &ct.c[0]);
        q_ctx.add_assign(&mut cs1, &ct.c[1]);

        Ciphertext {
            c: vec![cs0, cs1],
            poly_type: PolyType::Q,
            level: ct.level,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{Encoding, Plaintext};

    use super::*;

    // #[test]
    // fn relinerization_works() {
    //     let params = Arc::new(BfvParameters::default(3, 8));

    //     let mut rng = thread_rng();
    //     let sk = SecretKey::random(&params, &mut rng);

    //     let m = params
    //         .plaintext_modulus_op
    //         .random_vec(params.polynomial_degree, &mut rng);
    //     let pt = Plaintext::encode(&m, &params, Encoding::simd(0));

    //     let ct = sk.encrypt(&pt, &mut rng);
    //     let ct2 = sk.encrypt(&pt, &mut rng);

    //     let ct3 = ct.multiply1(&ct2);

    //     // rlk key
    //     let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

    //     // relinearize
    //     let ct3_rl = rlk.relinearize(&ct3);

    //     // decrypt and check equivalence!
    //     let res_m = sk.decrypt(&ct3_rl).decode(Encoding::simd(0));
    //     let mut m_clone = m.clone();
    //     params
    //         .plaintext_modulus_op
    //         .mul_mod_fast_vec(&mut m_clone, &m);
    //     assert_eq!(m_clone, res_m);
    // }
}
