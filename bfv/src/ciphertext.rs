use itertools::Itertools;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{
    proto,
    traits::{TryFromWithParameters, TryFromWithPolyContext},
    BfvParameters, Poly, PolyType,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Ciphertext {
    pub(crate) c: Vec<Poly>,
    pub(crate) poly_type: PolyType,
    pub(crate) seed: Option<<ChaCha8Rng as SeedableRng>::Seed>,
    pub(crate) level: usize,
}

impl Ciphertext {
    pub fn new(c: Vec<Poly>, poly_type: PolyType, level: usize) -> Ciphertext {
        Ciphertext {
            c,
            poly_type,
            level,
            seed: None,
        }
    }

    pub fn placeholder() -> Ciphertext {
        Ciphertext {
            c: vec![],
            poly_type: PolyType::Q,
            level: 0,
            seed: None,
        }
    }

    pub fn c_ref_mut(&mut self) -> &mut [Poly] {
        &mut self.c
    }

    pub fn c_ref(&self) -> &[Poly] {
        &self.c
    }

    pub fn poly_type(&self) -> PolyType {
        self.poly_type.clone()
    }

    pub fn level(&self) -> usize {
        self.level
    }
}

impl TryFromWithParameters for proto::Ciphertext {
    type Value = Ciphertext;
    type Parameters = BfvParameters;
    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        assert!(value.poly_type() == PolyType::Q);
        let poly_ctx = parameters.poly_ctx(&value.poly_type, value.level);
        let c = value
            .c
            .iter()
            .map(|p| proto::Poly::try_from_with_context(p, &poly_ctx))
            .collect_vec();

        proto::Ciphertext {
            c,
            level: value.level as u32,
        }
    }
}

impl TryFromWithParameters for Ciphertext {
    type Value = proto::Ciphertext;
    type Parameters = BfvParameters;
    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let level = value.level as usize;
        let poly_ctx = parameters.poly_ctx(&PolyType::Q, level);
        let c = value
            .c
            .iter()
            .map(|p_proto| Poly::try_from_with_context(p_proto, &poly_ctx))
            .collect_vec();

        Ciphertext {
            c,
            poly_type: PolyType::Q,
            level,
        }
    }
}

mod tests {
    use crate::{Encoding, Evaluator, SecretKey};

    use super::*;
    use prost::Message;
    use rand::thread_rng;

    #[test]
    fn serialize_and_deserialize_ciphertexts() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(5, 1 << 3);

        let sk = SecretKey::random(params.degree, &mut rng);
        let mut m0 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let evaluator = Evaluator::new(params);
        let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
        let ct0 = evaluator.encrypt(&sk, &pt0, &mut rng);

        let ct_proto = proto::Ciphertext::try_from_with_parameters(&ct0, evaluator.params());

        let bytes = ct_proto.encode_to_vec();

        let ct_back = Ciphertext::try_from_with_parameters(&ct_proto, evaluator.params());

        assert_eq!(ct0, ct_back);
    }
}
