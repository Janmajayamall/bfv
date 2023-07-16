use crate::{
    proto,
    traits::{TryFromWithParameters, TryFromWithPolyContext},
    BfvParameters, Ciphertext, HybridKeySwitchingKey, PolyType, Representation, SecretKey,
};
use rand::{CryptoRng, RngCore};

#[derive(PartialEq, Debug)]
pub struct RelinearizationKey {
    pub(crate) ksk: HybridKeySwitchingKey,
    pub(crate) level: usize,
}

impl RelinearizationKey {
    pub fn new<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        sk: &SecretKey,
        level: usize,
        rng: &mut R,
    ) -> RelinearizationKey {
        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);

        let mut sk_poly =
            q_ctx.try_convert_from_i64_small(&sk.coefficients, Representation::Coefficient);
        q_ctx.change_representation(&mut sk_poly, Representation::Evaluation);

        // sk^2
        let sk_sq = q_ctx.mul(&sk_poly, &sk_poly);

        // Key switching key
        let ksk = HybridKeySwitchingKey::new(
            params.hybrid_key_switching_params_at_level(level),
            &sk_sq,
            sk,
            &qp_ctx,
            params.variance,
            rng,
        );

        RelinearizationKey { ksk, level }
    }

    pub fn relinearize(&self, ct: &Ciphertext, params: &BfvParameters) -> Ciphertext {
        assert!(ct.c.len() == 3); // otherwise invalid relinerization
        assert!(ct.c[0].representation == Representation::Coefficient);
        assert!(ct.level == self.level);

        let level = ct.level;
        let q_ctx = params.poly_ctx(&PolyType::Q, level);
        let qp_ctx = params.poly_ctx(&PolyType::QP, level);
        let specialp_ctx = params.poly_ctx(&PolyType::SpecialP, level);

        let (mut cs0, mut cs1) = self.ksk.switch(
            params.hybrid_key_switching_params_at_level(self.level),
            &ct.c[2],
            &qp_ctx,
            &q_ctx,
            &specialp_ctx,
        );
        q_ctx.change_representation(&mut cs0, Representation::Coefficient);
        q_ctx.change_representation(&mut cs1, Representation::Coefficient);

        q_ctx.add_assign(&mut cs0, &ct.c[0]);
        q_ctx.add_assign(&mut cs1, &ct.c[1]);

        Ciphertext {
            c: vec![cs0, cs1],
            poly_type: PolyType::Q,
            level: ct.level,
            seed: None,
        }
    }
}

impl TryFromWithParameters for proto::RelinearizationKey {
    type Parameters = BfvParameters;
    type Value = RelinearizationKey;
    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let level = value.level;
        let ctx = parameters.poly_ctx(&PolyType::QP, level);

        // message types default to optional in proto3. For more info check this
        // answer https://github.com/tokio-rs/prost/discussions/679 and the one linked in it.
        // This is enforced by proto3, not something prost does.
        let ksk = Some(proto::HybridKeySwitchingKey::try_from_with_context(
            &value.ksk, &ctx,
        ));

        proto::RelinearizationKey {
            ksk,
            level: level as u32,
        }
    }
}

impl TryFromWithParameters for RelinearizationKey {
    type Parameters = BfvParameters;
    type Value = proto::RelinearizationKey;
    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let level = value.level as usize;
        let ctx = parameters.poly_ctx(&PolyType::QP, level);
        let ksk = HybridKeySwitchingKey::try_from_with_context(
            value.ksk.as_ref().expect("Rlk missing"),
            &ctx,
        );

        RelinearizationKey { ksk, level }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;

    #[test]
    fn serialize_and_deserialize_rlk() {
        let params = BfvParameters::default(6, 1 << 4);

        let mut rng = thread_rng();
        let sk = SecretKey::random(params.degree, &mut rng);

        let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

        let rlk_proto = proto::RelinearizationKey::try_from_with_parameters(&rlk, &params);
        let rlk_back = RelinearizationKey::try_from_with_parameters(&rlk_proto, &params);

        assert_eq!(rlk, rlk_back);
    }
}
