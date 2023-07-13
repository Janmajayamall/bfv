use crate::{
    proto,
    traits::{TryFromWithParameters, TryFromWithPolyContext},
    BfvParameters, Ciphertext, HybridKeySwitchingKey, Modulus, Poly, PolyContext, PolyType,
    Representation, SecretKey, Substitution,
};
use rand::{CryptoRng, RngCore};

#[derive(Debug, PartialEq)]
pub struct GaloisKey {
    pub(crate) substitution: Substitution,
    pub(crate) ksk_key: HybridKeySwitchingKey,
    pub(crate) level: usize,
}

impl GaloisKey {
    pub fn new<R: CryptoRng + RngCore>(
        exponent: usize,
        params: &BfvParameters,
        level: usize,
        sk: &SecretKey,
        rng: &mut R,
    ) -> GaloisKey {
        let substitution = Substitution::new(exponent, params.degree);

        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        // Substitute secret key
        let mut sk_poly =
            q_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        q_ctx.change_representation(&mut sk_poly, Representation::Evaluation);

        let sk_poly = q_ctx.substitute(&sk_poly, &substitution);

        // Generate key switching key for substituted secret key
        let ksk_key = HybridKeySwitchingKey::new(
            params.hybrid_key_switching_params_at_level(level),
            &sk_poly,
            &sk,
            &qp_ctx,
            rng,
        );

        GaloisKey {
            substitution,
            ksk_key,
            level,
        }
    }

    pub fn rotate(&self, ct: &Ciphertext, params: &BfvParameters) -> Ciphertext {
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

        let (mut cs0, mut cs1) = self.ksk_key.switch(
            params.hybrid_key_switching_params_at_level(level),
            &c1,
            &qp_ctx,
            &q_ctx,
            &specialp_ctx,
        );

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
            seed: None,
        }
    }
}

impl TryFromWithParameters for proto::GaloisKey {
    type Parameters = BfvParameters;
    type Value = GaloisKey;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let ctx = parameters.poly_ctx(&PolyType::QP, value.level);
        let ksk = Some(proto::HybridKeySwitchingKey::try_from_with_context(
            &value.ksk_key,
            &ctx,
        ));
        proto::GaloisKey {
            exponent: value.substitution.exponent as u32,
            ksk,
            level: value.level as u32,
        }
    }
}

impl TryFromWithParameters for GaloisKey {
    type Value = proto::GaloisKey;
    type Parameters = BfvParameters;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let substitution = Substitution::new(value.exponent as usize, parameters.degree);
        let level = value.level as usize;

        let ctx = parameters.poly_ctx(&PolyType::QP, level);
        let ksk = HybridKeySwitchingKey::try_from_with_context(&value.ksk.as_ref().unwrap(), &ctx);
        GaloisKey {
            substitution,
            ksk_key: ksk,
            level,
        }
    }
}
