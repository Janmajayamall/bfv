use crate::{
    BfvParameters, Ciphertext, HybridKeySwitchingKey, Modulus, Poly, PolyContext, PolyType,
    Representation, SecretKey, Substitution,
};

use rand::{CryptoRng, RngCore};
use traits::Ntt;

pub struct GaloisKey {
    substitution: Substitution,
    ksk_key: HybridKeySwitchingKey,
    level: usize,
}

impl GaloisKey {
    pub fn new<T: Ntt, R: CryptoRng + RngCore>(
        exponent: usize,
        params: &BfvParameters<T>,
        level: usize,
        sk: &SecretKey,
        rng: &mut R,
    ) -> GaloisKey {
        let substitution = Substitution::new(exponent, params.degree);

        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);
        let specialp_ctx = params.poly_ctx(&PolyType::SpecialP, level);

        // Substitute secret key
        let mut sk_poly =
            q_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        q_ctx.change_representation(&mut sk_poly, Representation::Evaluation);

        let sk_poly = q_ctx.substitute(&sk_poly, &substitution);

        // Generate key switching key for substituted secret key
        let ksk_key = HybridKeySwitchingKey::new(
            &sk_poly,
            &sk,
            &q_ctx,
            &specialp_ctx,
            &qp_ctx,
            params.alpha,
            params.aux_bits,
            rng,
        );

        GaloisKey {
            substitution,
            ksk_key,
            level,
        }
    }

    pub fn rotate<T: Ntt>(&self, ct: &Ciphertext, params: &BfvParameters<T>) -> Ciphertext {
        assert!(ct.c.len() == 2);
        assert!(ct.level == self.level);
        assert!(ct.poly_type == PolyType::Q);

        let level = self.level;
        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);
        let specialp_ctx = params.poly_ctx(&PolyType::SpecialP, level);

        // Key switch c1
        let mut c1 = q_ctx.substitute(&ct.c[1], &self.substitution);
        if c1.representation == Representation::Evaluation {
            q_ctx.change_representation(&mut c1, Representation::Coefficient);
        }

        let (mut cs0, mut cs1) = self.ksk_key.switch(&c1, &qp_ctx, &q_ctx, &specialp_ctx);

        // Key switch returns polynomial in Evaluation form
        if ct.c[0].representation != cs0.representation {
            q_ctx.change_representation(&mut cs0, ct.c[0].representation.clone());
            q_ctx.change_representation(&mut cs1, ct.c[0].representation.clone());
        }

        q_ctx.add_assign(&mut cs0, &q_ctx.substitute(&ct.c[0], &self.substitution));

        Ciphertext {
            c: vec![cs0, cs1],
            poly_type: PolyType::Q,
            level,
        }
    }
}
