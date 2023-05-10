use crate::modulus::Modulus;
use crate::parameters::BfvParameters;
use crate::poly::{Poly, Representation};
use std::sync::Arc;

#[derive(PartialEq, Clone)]
pub enum EncodingType {
    Simd,
    Poly,
}

#[derive(Clone)]
pub struct Encoding {
    pub(crate) encoding_type: EncodingType,
    pub(crate) level: usize,
}

impl Encoding {
    pub fn simd(level: usize) -> Encoding {
        Encoding {
            encoding_type: EncodingType::Simd,
            level,
        }
    }
}

pub struct Plaintext {
    pub(crate) m: Vec<u64>,
    pub(crate) params: Arc<BfvParameters>,
    pub(crate) encoding: Option<Encoding>,
}

impl Plaintext {
    pub fn new() {}

    /// Encodes a given message `m` to plaintext using given `encoding`
    ///
    /// Panics if `m` values length is greater than polynomial degree
    pub fn encode(m: &[u64], params: &Arc<BfvParameters>, encoding: Encoding) -> Plaintext {
        assert!(m.len() <= params.polynomial_degree);

        let mut m1 = vec![0u64; params.polynomial_degree];
        let mut m = m.to_vec();

        params.plaintext_modulus_op.reduce_vec(&mut m);
        m.iter().enumerate().for_each(|(i, v)| {
            if encoding.encoding_type == EncodingType::Simd {
                m1[params.matrix_reps_index_map[i]] = *v;
            } else {
                m1[i] = *v;
            }
        });

        if encoding.encoding_type == EncodingType::Simd {
            params.plaintext_ntt_op.backward(&mut m1);
        }

        Plaintext {
            m: m1,
            params: params.clone(),
            encoding: Some(encoding),
        }
    }

    pub fn decode(&self, encoding: Encoding) -> Vec<u64> {
        assert!(self.encoding.is_none());

        let mut m1 = self.m.clone();
        if encoding.encoding_type == EncodingType::Simd {
            self.params.plaintext_ntt_op.forward(&mut m1);
        }

        let mut m = vec![0u64; self.params.polynomial_degree];
        for i in (0..self.params.polynomial_degree) {
            if encoding.encoding_type == EncodingType::Simd {
                m[i] = m1[self.params.matrix_reps_index_map[i]];
            } else {
                m[i] = m1[i];
            }
        }

        m
    }

    /// Returns message polynomial `m` scaled by Q/t
    ///
    /// Panics if encoding is not specified
    pub fn to_poly(&self) -> Poly {
        match &self.encoding {
            Some(encoding) => {
                let modt = Modulus::new(self.params.plaintext_modulus);
                let mut m = self.m.clone();
                modt.scalar_mul_mod_fast_vec(&mut m, self.params.ql_modt[encoding.level]);

                let mut m_poly = Poly::try_convert_from_u64(
                    &m,
                    &self.params.ciphertext_poly_contexts[encoding.level],
                    &Representation::Coefficient,
                );
                m_poly.change_representation(Representation::Evaluation);

                m_poly *= &self.params.neg_t_inv_modql[encoding.level];
                m_poly
            }
            None => {
                panic!("Encoding not specified!");
            }
        }
    }
}
