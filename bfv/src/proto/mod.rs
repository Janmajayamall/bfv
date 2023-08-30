use std::collections::HashMap;

use crate::{
    convert_bytes_to_ternary, convert_from_bytes, convert_ternary_to_bytes, convert_to_bytes,
    BfvParameters, Ciphertext, EvaluationKey, GaloisKey, HybridKeySwitchingKey, Poly, PolyContext,
    PolyType, RelinearizationKey, Representation, SecretKey, Substitution,
};
use itertools::{izip, Itertools};
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use traits::{TryFromWithParameters, TryFromWithPolyContext};

// include protos
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/_.rs"));
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
                // serialise the first polynomial
                assert!(value.c.len() <= 2);
                1
            }
        };

        let c = value.c[..slice]
            .iter()
            .map(|p| {
                // Avoid converting polynomial to `Coefficient` representation to allow
                // assert in `Poly` to fail. This also avoids adding silent NTTs of which
                // user of the API isn't aware.
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
                // `generate_c1` returns c1s in `Coefficient` representation. Convert them to `Evaluation` representation.
                let mut c = HybridKeySwitchingKey::generate_c1(c0s.len(), poly_ctx, seed);
                c.iter_mut().for_each(|p| {
                    poly_ctx.change_representation(p, Representation::Evaluation);
                });
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

mod tests {
    use super::*;
    use crate::{Encoding, Evaluator, SecretKey};
    use prost::Message;
    use rand::thread_rng;

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
}
