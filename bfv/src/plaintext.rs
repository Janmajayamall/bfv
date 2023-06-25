use traits::Ntt;

use crate::modulus::Modulus;
use crate::parameters::{BfvParameters, PolyType};
use crate::poly::{Poly, Representation};
use crate::Ciphertext;
use std::ops::SubAssign;
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

impl Default for Encoding {
    fn default() -> Self {
        Encoding {
            encoding_type: EncodingType::Simd,
            level: 0,
        }
    }
}

#[derive(Clone)]
pub struct Plaintext {
    pub(crate) m: Vec<u64>,
    pub(crate) encoding: Option<Encoding>,
    pub(crate) poly_ntt: Option<Poly>,
}

impl Plaintext {
    /// Encodes a given message `m` to plaintext using given `encoding`
    ///
    /// Panics if `m` values length is greater than polynomial degree
    pub fn encode<T: Ntt>(m: &[u64], params: &BfvParameters<T>, encoding: Encoding) -> Plaintext {
        assert!(m.len() <= params.degree);

        let mut m1 = vec![0u64; params.degree];
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

        // convert m to polynomial with poly context at specific level
        let ctx = params.poly_ctx(&crate::PolyType::Q, encoding.level);
        let mut poly_ntt = ctx.try_convert_from_u64(&m1, Representation::Coefficient);
        ctx.change_representation(&mut poly_ntt, Representation::Evaluation);

        Plaintext {
            m: m1,
            encoding: Some(encoding),
            poly_ntt: Some(poly_ntt),
        }
    }

    pub fn decode<T: Ntt>(&self, encoding: Encoding, params: &BfvParameters<T>) -> Vec<u64> {
        assert!(self.encoding.is_none());

        let mut m1 = self.m.clone();
        if encoding.encoding_type == EncodingType::Simd {
            params.plaintext_ntt_op.forward(&mut m1);
        }

        let mut m = vec![0u64; params.degree];
        for i in (0..params.degree) {
            if encoding.encoding_type == EncodingType::Simd {
                m[i] = m1[params.matrix_reps_index_map[i]];
            } else {
                m[i] = m1[i];
            }
        }

        m
    }

    /// Returns message polynomial `m` scaled by Q/t
    ///
    /// Panics if encoding is not specified
    pub fn to_poly<T: Ntt>(&self, params: &BfvParameters<T>) -> Poly {
        match &self.encoding {
            Some(encoding) => {
                let modt = &params.plaintext_modulus_op;

                let mut m = self.m.clone();
                modt.scalar_mul_mod_fast_vec(&mut m, params.ql_modt[encoding.level]);

                let ctx = params.poly_ctx(&PolyType::Q, encoding.level);
                let mut m_poly = ctx.try_convert_from_u64(&m, Representation::Coefficient);

                // An alternate method to this will be to store [-t_inv]_Q
                // and perform scalar multiplication [-t_inv]_Q with `m_poly`
                // in coefficient form.
                // We prefer this because `m_poly` needs to change representation
                // to `Evaluation` anyways.
                ctx.change_representation(&mut m_poly, Representation::Evaluation);
                ctx.mul_assign(&mut m_poly, &params.neg_t_inv_modql[encoding.level]);
                m_poly
            }
            None => {
                panic!("Encoding not specified!");
            }
        }
    }

    pub fn poly_ntt_ref(&self) -> &Poly {
        self.poly_ntt.as_ref().expect("Missing poly ntt")
    }
}

#[cfg(test)]
mod tests {}
