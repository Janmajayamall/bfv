use std::{collections::HashMap, default};

use crate::{
    convert_bytes_to_ternary, convert_from_bytes, convert_ternary_to_bytes, convert_to_bytes,
    BfvParameters, Ciphertext, CollectiveDecryption, CollectiveDecryptionShare,
    CollectivePublicKeyShare, CollectiveRlkAggShare1, CollectiveRlkAggTrimmedShare1,
    CollectiveRlkShare1, CollectiveRlkShare2, EvaluationKey, GaloisKey, HybridKeySwitchingKey,
    Poly, PolyContext, PolyType, PublicKey, RelinearizationKey, Representation, SecretKey,
    Substitution,
};
use itertools::{izip, Itertools};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use traits::{TryFromWithLevelledParameters, TryFromWithParameters, TryFromWithPolyContext};

// include protos
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/my.bfv.rs"));
}

// Poly //
impl<'a> TryFromWithPolyContext<'a> for Poly {
    type Value = proto::Poly;
    type PolyContext = crate::PolyContext<'a>;

    fn try_from_with_context(poly: &Self::Value, poly_ctx: &'a Self::PolyContext) -> Self {
        let coefficients = izip!(poly.coefficients.iter(), poly_ctx.iter_moduli_ops())
            .flat_map(|(xi, modqi)| {
                let values = convert_from_bytes(xi, modqi.modulus());
                assert!(values.len() == poly_ctx.degree());
                values
            })
            .collect_vec();
        let coefficients =
            Array2::from_shape_vec((poly_ctx.moduli_count(), poly_ctx.degree()), coefficients)
                .unwrap();

        Poly {
            coefficients,
            representation: Representation::Coefficient,
        }
    }
}
impl<'a> TryFromWithPolyContext<'a> for proto::Poly {
    type Value = Poly;
    type PolyContext = crate::PolyContext<'a>;

    fn try_from_with_context(poly: &Self::Value, poly_ctx: &'a Self::PolyContext) -> Self {
        assert!(poly.representation == Representation::Coefficient);

        let bytes = izip!(poly.coefficients.outer_iter(), poly_ctx.iter_moduli_ops())
            .map(|(xi, modqi)| convert_to_bytes(xi.as_slice().unwrap(), modqi.modulus()))
            .collect_vec();

        proto::Poly {
            coefficients: bytes,
        }
    }
}

// SecretKey //
impl TryFromWithParameters for proto::SecretKey {
    type Value = SecretKey;
    type Parameters = BfvParameters;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let bytes = convert_ternary_to_bytes(&value.coefficients);
        proto::SecretKey {
            coefficients: bytes,
        }
    }
}

impl TryFromWithParameters for SecretKey {
    type Parameters = BfvParameters;
    type Value = proto::SecretKey;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let coefficients =
            convert_bytes_to_ternary(&value.coefficients, parameters.degree).into_boxed_slice();

        SecretKey { coefficients }
    }
}

// Ciphertext //
impl TryFromWithParameters for proto::Ciphertext {
    type Value = Ciphertext;
    type Parameters = BfvParameters;
    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        assert!(value.poly_type() == PolyType::Q);
        let poly_ctx = parameters.poly_ctx(&value.poly_type, value.level);

        let slice = {
            if value.seed.is_none() {
                // if seed is not present, then we need all ciphertext polynomials
                value.c.len()
            } else {
                // if seed is present, then the ciphertext can be assumed to be fresh ciphertext with
                // polynomial degree of <= 2 where the second polynomial is seeded. Thus we only need to
                // serialize the first polynomial
                assert!(value.c.len() <= 2);
                1
            }
        };

        let c = value.c[..slice]
            .iter()
            .map(|p| {
                // Avoid converting polynomial to `Coefficient` representation to allow
                // assert in `Poly` to fail. This avoids adding hidden NTTs.
                proto::Poly::try_from_with_context(p, &poly_ctx)
            })
            .collect_vec();

        let seed = value.seed.as_ref().and_then(|s| Some(s.to_vec()));

        proto::Ciphertext {
            c,
            level: value.level as u32,
            seed,
        }
    }
}
impl TryFromWithParameters for Ciphertext {
    type Value = proto::Ciphertext;
    type Parameters = BfvParameters;
    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let level = value.level as usize;
        let poly_ctx = parameters.poly_ctx(&PolyType::Q, level);

        let mut c = value
            .c
            .iter()
            .map(|p_proto| Poly::try_from_with_context(p_proto, &poly_ctx))
            .collect_vec();

        let seed = value.seed.as_ref().and_then(|s| {
            let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
            seed.copy_from_slice(&s);
            Some(seed)
        });

        if seed.is_some() {
            assert!(c.len() < 2);

            let a = poly_ctx.random_with_seed(seed.unwrap());
            c.push(a);
        }

        Ciphertext {
            c,
            poly_type: PolyType::Q,
            level,
            seed,
        }
    }
}

// Hybrid Key Switching Key //
impl<'a> TryFromWithPolyContext<'a> for proto::HybridKeySwitchingKey {
    type PolyContext = PolyContext<'a>;
    type Value = HybridKeySwitchingKey;
    fn try_from_with_context(value: &Self::Value, poly_ctx: &'a Self::PolyContext) -> Self {
        let c0s = value
            .c0s
            .iter()
            .map(|p| {
                // Since neither c0s nor c1s change representation to `Coefficient` form it is safe to assume
                // that c0s polynomials will be `Evaluation` form
                let mut p = p.clone();
                poly_ctx.change_representation(&mut p, Representation::Coefficient);
                proto::Poly::try_from_with_context(&p, &poly_ctx)
            })
            .collect_vec();

        let c1s = {
            if value.seed.is_none() {
                value
                    .c1s
                    .iter()
                    .map(|p| {
                        let mut p = p.clone();
                        poly_ctx.change_representation(&mut p, Representation::Coefficient);
                        proto::Poly::try_from_with_context(&p, &poly_ctx)
                    })
                    .collect_vec()
            } else {
                vec![]
            }
        };

        let seed = value.seed.and_then(|s| Some(s.to_vec()));

        proto::HybridKeySwitchingKey { c0s, c1s, seed }
    }
}

impl<'a> TryFromWithPolyContext<'a> for HybridKeySwitchingKey {
    type PolyContext = PolyContext<'a>;
    type Value = proto::HybridKeySwitchingKey;
    fn try_from_with_context(value: &Self::Value, poly_ctx: &'a Self::PolyContext) -> Self {
        let c0s = value
            .c0s
            .iter()
            .map(|p| {
                // c0s and c1s are only needed in `Evaluation` form so it safe to convert them
                // from `Coefficient` (default form for serialization) to `Evaluation`.
                let mut p = Poly::try_from_with_context(p, &poly_ctx);
                poly_ctx.change_representation(&mut p, Representation::Evaluation);
                p
            })
            .collect_vec();

        let (c1s, seed) = {
            if value.seed.is_none() {
                assert!(value.c1s.len() == value.c0s.len());
                let c = value
                    .c1s
                    .iter()
                    .map(|p| {
                        let mut p = Poly::try_from_with_context(p, &poly_ctx);
                        poly_ctx.change_representation(&mut p, Representation::Evaluation);
                        p
                    })
                    .collect_vec();
                (c, None)
            } else {
                let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
                seed.copy_from_slice(value.seed());
                let c = HybridKeySwitchingKey::generate_c1(c0s.len(), poly_ctx, seed);
                (c, Some(seed))
            }
        };

        HybridKeySwitchingKey {
            seed,
            c0s: c0s.into_boxed_slice(),
            c1s: c1s.into_boxed_slice(),
        }
    }
}

// Galois Key //
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

// Relinerization Key //
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

// Evaluation Key //
impl TryFromWithParameters for proto::EvaluationKey {
    type Parameters = BfvParameters;
    type Value = EvaluationKey;
    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        // since HashMap iterates over values in arbitrary order seralisation of same `EvaluationKey`
        // twice can produce different `proto::EvaluationKey`s.
        let rlks = value
            .rlks
            .iter()
            .map(|(i, k)| proto::RelinearizationKey::try_from_with_parameters(&k, parameters))
            .collect_vec();
        let mut rot_indices = vec![];
        let rtgs = value
            .rtgs
            .iter()
            .map(|(i, k)| {
                rot_indices.push(i.0 as i32);
                proto::GaloisKey::try_from_with_parameters(&k, parameters)
            })
            .collect_vec();

        proto::EvaluationKey {
            rlks,
            rtgs,
            rot_indices,
        }
    }
}

impl TryFromWithParameters for EvaluationKey {
    type Parameters = BfvParameters;
    type Value = proto::EvaluationKey;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let mut rlks = HashMap::new();
        value.rlks.iter().for_each(|v| {
            let v = RelinearizationKey::try_from_with_parameters(v, parameters);
            rlks.insert(v.level, v);
        });

        let mut rtgs = HashMap::new();
        value
            .rtgs
            .iter()
            .zip(value.rot_indices.iter())
            .for_each(|(gk, rot_index)| {
                let v = GaloisKey::try_from_with_parameters(gk, parameters);
                rtgs.insert((*rot_index as isize, v.level), v);
            });

        EvaluationKey { rlks, rtgs }
    }
}

// Public Key //
impl TryFromWithParameters for proto::PublicKey {
    type Parameters = BfvParameters;
    type Value = PublicKey;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let qr_ctx = parameters.poly_ctx(&PolyType::Qr, 0);

        let mut c0 = value.c0.clone();
        qr_ctx.change_representation(&mut c0, Representation::Coefficient);
        let c0_seriliazed = proto::Poly::try_from_with_context(&c0, &qr_ctx);

        proto::PublicKey {
            c0: Some(c0_seriliazed),
            seed: value.seed.to_vec(),
        }
    }
}

impl TryFromWithParameters for PublicKey {
    type Parameters = BfvParameters;
    type Value = proto::PublicKey;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let qr_ctx = parameters.poly_ctx(&PolyType::Qr, 0);

        let mut c0 = Poly::try_from_with_context(value.c0.as_ref().unwrap(), &qr_ctx);
        qr_ctx.change_representation(&mut c0, Representation::Evaluation);

        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        seed.copy_from_slice(&value.seed);

        let mut prng = ChaCha8Rng::from_seed(seed);
        let mut c1 = qr_ctx.random(Representation::Evaluation, &mut prng);
        qr_ctx.neg_assign(&mut c1);

        PublicKey { c0, c1, seed }
    }
}

// Multi party //

// CollectivePublicKeyShare //
impl TryFromWithParameters for proto::CollectivePublicKeyShare {
    type Parameters = BfvParameters;
    type Value = CollectivePublicKeyShare;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let qr = parameters.poly_ctx(&PolyType::Qr, 0);
        let share = proto::Poly::try_from_with_context(&value.0, &qr);

        proto::CollectivePublicKeyShare { share: Some(share) }
    }
}

impl TryFromWithParameters for CollectivePublicKeyShare {
    type Parameters = BfvParameters;
    type Value = proto::CollectivePublicKeyShare;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let qr = parameters.poly_ctx(&PolyType::Qr, 0);
        let poly = Poly::try_from_with_context(&value.share.as_ref().unwrap(), &qr);

        CollectivePublicKeyShare(poly)
    }
}

// CollectiveDecryptionShare //
impl TryFromWithLevelledParameters for proto::CollectiveDecryptionShare {
    type Parameters = BfvParameters;
    type Value = CollectiveDecryptionShare;

    fn try_from_with_levelled_parameters(
        value: &Self::Value,
        parameters: &Self::Parameters,
        level: usize,
    ) -> Self {
        let q = parameters.poly_ctx(&PolyType::Q, level);
        let share = proto::Poly::try_from_with_context(&value.0, &q);
        proto::CollectiveDecryptionShare { share: Some(share) }
    }
}

impl TryFromWithLevelledParameters for CollectiveDecryptionShare {
    type Parameters = BfvParameters;
    type Value = proto::CollectiveDecryptionShare;

    fn try_from_with_levelled_parameters(
        value: &Self::Value,
        parameters: &Self::Parameters,
        level: usize,
    ) -> Self {
        let q = parameters.poly_ctx(&PolyType::Q, level);
        let poly = Poly::try_from_with_context(value.share.as_ref().unwrap(), &q);

        CollectiveDecryptionShare(poly)
    }
}

// Relinerization procedure //
// CollectiveRlkShare1 //
impl TryFromWithParameters for proto::CollectiveRlkShare1 {
    type Parameters = BfvParameters;
    type Value = CollectiveRlkShare1;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let qp_ctx = parameters.poly_ctx(&PolyType::QP, 0);
        let polys = value
            .0
            .iter()
            .flat_map(|p_vec| {
                p_vec
                    .iter()
                    .map(|p| proto::Poly::try_from_with_context(p, &qp_ctx))
                    .collect_vec()
            })
            .collect_vec();

        proto::CollectiveRlkShare1 { shares: polys }
    }
}

impl TryFromWithParameters for CollectiveRlkShare1 {
    type Parameters = BfvParameters;
    type Value = proto::CollectiveRlkShare1;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let qp_ctx = parameters.poly_ctx(&PolyType::QP, 0);

        let dnum = parameters.dnum.unwrap();
        assert!(
            value.shares.len() == 2 * dnum,
            "Not enough shares in CollectiveRlkShare1"
        );

        let h0s = value.shares[..dnum]
            .iter()
            .map(|p| Poly::try_from_with_context(p, &qp_ctx))
            .collect_vec();

        let h1s = value.shares[dnum..]
            .iter()
            .map(|p| Poly::try_from_with_context(p, &qp_ctx))
            .collect_vec();

        CollectiveRlkShare1([h0s, h1s])
    }
}

// CollectiveRlkShare2 //
impl TryFromWithParameters for proto::CollectiveRlkShare2 {
    type Parameters = BfvParameters;
    type Value = CollectiveRlkShare2;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        assert!(
            value.0.len() == parameters.dnum.unwrap(),
            "Polys in collective share 2 are do not equal decomposition count (i.e. dnum)"
        );

        let qp_ctx = parameters.poly_ctx(&PolyType::QP, 0);
        let shares = value
            .0
            .iter()
            .map(|p| proto::Poly::try_from_with_context(p, &qp_ctx))
            .collect_vec();

        proto::CollectiveRlkShare2 { shares }
    }
}
impl TryFromWithParameters for CollectiveRlkShare2 {
    type Parameters = BfvParameters;
    type Value = proto::CollectiveRlkShare2;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        assert!(
            value.shares.len() == parameters.dnum.unwrap(),
            "Polys in collective share 2 are do not equal decomposition count (i.e. dnum)"
        );

        let qp_ctx = parameters.poly_ctx(&PolyType::QP, 0);
        let polys = value
            .shares
            .iter()
            .map(|p| Poly::try_from_with_context(p, &qp_ctx))
            .collect_vec();

        CollectiveRlkShare2(polys)
    }
}

// CollectiveRlkAggTrimmedShare1 //
impl TryFromWithParameters for proto::CollectiveRlkAggTrimmedShare1 {
    type Parameters = BfvParameters;
    type Value = CollectiveRlkAggTrimmedShare1;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        assert!(
            value.0.len() == parameters.dnum.unwrap(),
            "Polys do not equal decomposition count (i.e. dnum)"
        );

        let qp_ctx = parameters.poly_ctx(&PolyType::QP, 0);
        let shares = value
            .0
            .iter()
            .map(|p| {
                let mut p = p.clone();
                qp_ctx.change_representation(&mut p, Representation::Coefficient);
                proto::Poly::try_from_with_context(&p, &qp_ctx)
            })
            .collect_vec();

        proto::CollectiveRlkAggTrimmedShare1 { shares }
    }
}

impl TryFromWithParameters for CollectiveRlkAggTrimmedShare1 {
    type Parameters = BfvParameters;
    type Value = proto::CollectiveRlkAggTrimmedShare1;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        assert!(
            value.shares.len() == parameters.dnum.unwrap(),
            "Polys do not equal decomposition count (i.e. dnum)"
        );

        let qp_ctx = parameters.poly_ctx(&PolyType::QP, 0);
        let polys = value
            .shares
            .iter()
            .map(|p| {
                let mut p = Poly::try_from_with_context(p, &qp_ctx);
                qp_ctx.change_representation(&mut p, Representation::Evaluation);
                p
            })
            .collect_vec();

        CollectiveRlkAggTrimmedShare1(polys)
    }
}

mod tests {
    use super::*;
    use crate::{
        CollectivePublicKeyGenerator, CollectiveRlkGenerator, Encoding, Evaluator, Plaintext,
        SecretKey, CRS,
    };
    use prost::Message;
    use rand::{thread_rng, RngCore};

    #[test]
    fn serialize_and_deserialize_secret_key() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(5, 1 << 4);

        let sk = SecretKey::random_with_params(&params, &mut rng);

        let sk_proto = proto::SecretKey::try_from_with_parameters(&sk, &params);
        let sk_back = SecretKey::try_from_with_parameters(&sk_proto, &params);

        assert_eq!(sk, sk_back);
    }

    #[test]
    fn serialize_and_deserialize_ciphertexts() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(5, 1 << 4);

        let sk = SecretKey::random(params.degree, params.hw, &mut rng);
        let mut m0 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let evaluator = Evaluator::new(params);
        let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
        let mut ct0 = evaluator.encrypt(&sk, &pt0, &mut rng);
        ct0.seed = None;

        let ct_proto = proto::Ciphertext::try_from_with_parameters(&ct0, evaluator.params());
        let ct_back = Ciphertext::try_from_with_parameters(&ct_proto, evaluator.params());

        assert_eq!(ct0, ct_back);
    }

    #[test]
    fn serialize_and_deserialize_poly() {
        let params = BfvParameters::default(3, 1 << 15);
        let ctx = params.poly_ctx(&crate::PolyType::Q, 0);

        let mut rng = thread_rng();
        let poly = ctx.random(Representation::Coefficient, &mut rng);
        let proto = proto::Poly::try_from_with_context(&poly, &ctx);
        let bytes = proto.encode_to_vec();
        dbg!(bytes.len());
        let poly_back = Poly::try_from_with_context(&proto, &ctx);

        assert_eq!(poly, poly_back);
    }

    #[test]
    fn serialize_and_deserialize_hybrid_ksk() {
        let params = BfvParameters::default(15, 1 << 8);
        let qp_ctx = params.poly_ctx(&PolyType::QP, 0);
        let ksk_ctx = params.poly_ctx(&PolyType::Q, 0);

        let mut rng = thread_rng();
        let poly = ksk_ctx.random(Representation::Evaluation, &mut rng);
        let sk = SecretKey::random(params.degree, params.hw, &mut rng);
        let mut ksk = HybridKeySwitchingKey::new(
            params.hybrid_key_switching_params_at_level(0),
            &poly,
            &sk,
            &qp_ctx,
            params.variance,
            &mut rng,
        );

        let ksk_proto = proto::HybridKeySwitchingKey::try_from_with_context(&ksk, &qp_ctx);
        dbg!(ksk_proto.encode_to_vec().len());
        let ksk_back = HybridKeySwitchingKey::try_from_with_context(&ksk_proto, &qp_ctx);

        assert_eq!(ksk, ksk_back);
    }

    #[test]
    fn serialize_and_deserialize_rlk() {
        let params = BfvParameters::default(6, 1 << 4);

        let mut rng = thread_rng();
        let sk = SecretKey::random(params.degree, params.hw, &mut rng);

        let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

        let rlk_proto = proto::RelinearizationKey::try_from_with_parameters(&rlk, &params);
        let rlk_back = RelinearizationKey::try_from_with_parameters(&rlk_proto, &params);

        assert_eq!(rlk, rlk_back);
    }

    #[test]
    fn serialize_and_deserialize_ek() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 8);

        let sk = SecretKey::random(params.degree, params.hw, &mut rng);

        let ek = EvaluationKey::new(
            &params,
            &sk,
            &[0, 1, 2, 3, 4, 5],
            &[0, 1, 2, 3, 4, 5],
            &[1, 2, 3, -1, -2, -3],
            &mut rng,
        );

        let ek_proto = proto::EvaluationKey::try_from_with_parameters(&ek, &params);
        let ek_back = EvaluationKey::try_from_with_parameters(&ek_proto, &params);

        assert_eq!(ek, ek_back);
    }

    #[test]
    fn serialize_and_deserialize_public_key() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(2, 1 << 4);

        let sk = SecretKey::random(params.degree, params.hw, &mut rng);
        let pk = PublicKey::new(&params, &sk, &mut rng);

        let pk_proto = proto::PublicKey::try_from_with_parameters(&pk, &params);
        let pk_back = PublicKey::try_from_with_parameters(&pk_proto, &params);

        assert_eq!(pk, pk_back);
    }

    #[test]
    fn serialize_and_deserialize_multi_party_bfv() {
        pub struct PartySecret {
            secret: SecretKey,
        }

        fn gen_crs() -> CRS {
            let mut rng = thread_rng();
            let mut seed = [0u8; 32];
            rng.fill_bytes(&mut seed);
            seed
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

        let no_of_parties = 10;
        let params = BfvParameters::default(3, 1 << 8);
        let level = 0;
        let parties = setup_parties(&params, no_of_parties);

        // Generate collective public key //
        let mut rng = thread_rng();
        let crs_cpk = gen_crs();
        let shares = parties
            .iter()
            .map(|party_i| {
                CollectivePublicKeyGenerator::generate_share(
                    &params,
                    &party_i.secret,
                    crs_cpk,
                    &mut rng,
                )
            })
            .collect_vec();
        let collective_public_key =
            CollectivePublicKeyGenerator::aggregate_shares_and_finalise(&params, &shares, crs_cpk);

        // test serialize/deserialize collective public key
        {
            let serialized =
                proto::PublicKey::try_from_with_parameters(&collective_public_key, &params);
            let deserialized = PublicKey::try_from_with_parameters(&serialized, &params);

            assert_eq!(&collective_public_key, &deserialized);
        }

        // Generate RLK //

        // initialise state
        let mut rng = thread_rng();
        let collective_rlk_state = parties
            .iter()
            .map(|party_i| CollectiveRlkGenerator::init_state(&params, &mut rng))
            .collect_vec();

        // Generate and collect h0s and h1s
        let crs_rlk = gen_crs();
        let mut collective_rlk_share1 = izip!(parties.iter(), collective_rlk_state.iter())
            .map(|(party_i, state_i)| {
                CollectiveRlkGenerator::generate_share_1(
                    &params,
                    &party_i.secret,
                    state_i,
                    crs_rlk,
                    level,
                    &mut rng,
                )
            })
            .collect_vec();

        // test deserilisation/serilisation of collective_rlk_share1
        collective_rlk_share1.iter().for_each(|c| {
            // serialize
            let serialized = proto::CollectiveRlkShare1::try_from_with_parameters(c, &params);

            // deserialize
            let deserialized = CollectiveRlkShare1::try_from_with_parameters(&serialized, &params);

            assert_eq!(c, &deserialized);
        });

        // aggregate h0s and h1s
        let collective_rlk_agg_share1 =
            CollectiveRlkGenerator::aggregate_shares_1(&params, &collective_rlk_share1, level);

        // Generate and collect h'0s h'1s
        let collective_rlk_share2 = izip!(parties.iter(), collective_rlk_state.iter())
            .map(|(party_i, state_i)| {
                CollectiveRlkGenerator::generate_share_2(
                    &params,
                    &party_i.secret,
                    &collective_rlk_agg_share1,
                    state_i,
                    level,
                    &mut rng,
                )
            })
            .collect_vec();

        // test deserilisation/serilisation of collective_rlk_share2
        collective_rlk_share2.iter().for_each(|c| {
            // serialize
            let serialized = proto::CollectiveRlkShare2::try_from_with_parameters(c, &params);

            // deserialize
            let deserialized = CollectiveRlkShare2::try_from_with_parameters(&serialized, &params);

            assert_eq!(c, &deserialized);
        });

        // trim collective rlk aggregated share 1
        let collective_rlk_agg_trimmed_share1 = collective_rlk_agg_share1.trim();

        // test deserilisation/serilisation of collective_rlk_agg_trimmed_share1
        {
            let serialized = proto::CollectiveRlkAggTrimmedShare1::try_from_with_parameters(
                &collective_rlk_agg_trimmed_share1,
                &params,
            );
            let deserialized =
                CollectiveRlkAggTrimmedShare1::try_from_with_parameters(&serialized, &params);

            assert_eq!(&collective_rlk_agg_trimmed_share1, &deserialized);
        }

        // aggregate h'0s and h'1s
        let rlk = CollectiveRlkGenerator::aggregate_shares_2(
            &params,
            &collective_rlk_share2,
            collective_rlk_agg_trimmed_share1,
            level,
        );

        // test deserilisation/serilisation of rilinearization key
        {
            let serialized = proto::RelinearizationKey::try_from_with_parameters(&rlk, &params);
            let deserialized = RelinearizationKey::try_from_with_parameters(&serialized, &params);

            assert_eq!(&rlk, &deserialized);
        }

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
        let ct0 = collective_public_key.encrypt(&params, &pt0, &mut rng);
        let ct1 = collective_public_key.encrypt(&params, &pt1, &mut rng);

        // multiply ciphertexts
        let evaluation_key = EvaluationKey::new_raw(&[0], vec![rlk], &[], &[], vec![]);
        let evaluator = Evaluator::new(params);
        let ct0c1 = evaluator.mul(&ct0, &ct1);
        let ct_out = evaluator.relinearize(&ct0c1, &evaluation_key);

        // Collective decryption
        let collective_decryption_shares = parties
            .iter()
            .map(|party_i| {
                CollectiveDecryption::generate_share(
                    evaluator.params(),
                    &ct_out,
                    &party_i.secret,
                    &mut rng,
                )
            })
            .collect_vec();

        {
            collective_decryption_shares.iter().for_each(|share| {
                let serialized =
                    proto::CollectiveDecryptionShare::try_from_with_levelled_parameters(
                        share,
                        evaluator.params(),
                        0,
                    );
                let deserialized = CollectiveDecryptionShare::try_from_with_levelled_parameters(
                    &serialized,
                    evaluator.params(),
                    level,
                );

                assert_eq!(share, &deserialized);
            });
        }

        let m0m1: Vec<u64> = CollectiveDecryption::aggregate_share_and_decrypt(
            evaluator.params(),
            &ct_out,
            &collective_decryption_shares,
        )
        .decode(Encoding::default(), evaluator.params());
        let mut m0m1_expected = m0;
        evaluator
            .params()
            .plaintext_modulus_op
            .mul_mod_fast_vec(&mut m0m1_expected, &m1);

        assert_eq!(m0m1, m0m1_expected);
    }
}
