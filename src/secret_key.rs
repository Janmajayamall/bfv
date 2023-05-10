use crate::ciphertext::Ciphertext;
use crate::parameters::BfvParameters;
use crate::plaintext::{Encoding, Plaintext};
use crate::poly::{Poly, Representation};
use itertools::Itertools;
use num_bigint::BigUint;
use rand::distributions::{Distribution, Uniform};
use rand::{CryptoRng, RngCore};
use std::sync::Arc;

pub struct SecretKey {
    pub(crate) coefficients: Box<[i64]>,
    pub(crate) params: Arc<BfvParameters>,
}

impl SecretKey {
    /// Generates a random secret key
    pub fn random<R: CryptoRng + RngCore>(params: &Arc<BfvParameters>, rng: &mut R) -> SecretKey {
        let coefficients = Uniform::new(-1, 2)
            .sample_iter(rng)
            .take(params.polynomial_degree)
            .collect_vec()
            .into_boxed_slice();

        SecretKey {
            coefficients,
            params: params.clone(),
        }
    }

    /// Creates a new secret key with given coefficients.
    ///
    /// Panics if coefficients length does not match with degree of given bfv parameter
    ///
    /// Panics if each value in coefficients does not belong to ternary distribution (ie {-1,0,1}).
    pub fn new(coefficients: Vec<i64>, params: &Arc<BfvParameters>) -> SecretKey {
        assert!(coefficients.len() == params.polynomial_degree);
        coefficients.iter().for_each(|c| {
            assert!(-1 <= *c && 1 >= *c);
        });

        SecretKey {
            coefficients: coefficients.into_boxed_slice(),
            params: params.clone(),
        }
    }

    /// Returns secret key polynomial for polynomial context at given level in Evaluation form
    fn to_poly(&self, level: usize) -> Poly {
        let context = self.params.ciphertext_poly_contexts[level].clone();
        let mut p = Poly::try_convert_from_i64_small(
            &self.coefficients,
            &context,
            &Representation::Coefficient,
        );
        p.change_representation(Representation::Evaluation);
        p
    }

    /// Encrypts given plaintext with the secret key
    pub fn encrypt<R: CryptoRng + RngCore>(&self, pt: &Plaintext, rng: &mut R) -> Ciphertext {
        debug_assert!(pt.params == self.params);

        if pt.encoding.is_none() {
            panic!("Plaintext encoding missing!");
        }
        let encoding = pt.encoding.clone().unwrap();

        let mut sk_poly = self.to_poly(encoding.level);

        let m = pt.to_poly();
        let a = Poly::random(
            &self.params.ciphertext_poly_contexts[encoding.level],
            &Representation::Evaluation,
            rng,
        );
        sk_poly *= &a;
        let mut e = Poly::random_gaussian(
            &self.params.ciphertext_poly_contexts[encoding.level],
            &Representation::Coefficient,
            10,
            rng,
        );
        e.change_representation(Representation::Evaluation);
        e -= &sk_poly;
        e += &m;

        Ciphertext {
            c: vec![e, a],
            params: self.params.clone(),
            level: encoding.level,
        }
    }

    pub fn decrypt(&self, ct: &Ciphertext) -> Plaintext {
        // Panic on empty ciphertext
        assert!(ct.c.len() != 0);

        debug_assert!(ct.params == self.params);

        let mut m = ct.c[0].clone();
        debug_assert!(m.representation == Representation::Evaluation);
        let mut s = self.to_poly(ct.level);
        let mut s_carry = s.clone();
        for i in 1..ct.c.len() {
            m += &(&s_carry * &ct.c[i]);
            s_carry *= &s;
        }

        m.change_representation(Representation::Coefficient);
        let m = m.scale_and_round_decryption(
            &self.params.plaintext_modulus_op,
            self.params.max_bit_size_by2,
            &self.params.t_qlhat_inv_modql_divql_modt[ct.level],
            &self.params.t_bqlhat_inv_modql_divql_modt[ct.level],
            &self.params.t_qlhat_inv_modql_divql_frac[ct.level],
            &self.params.t_bqlhat_inv_modql_divql_frac[ct.level],
        );

        Plaintext {
            m,
            params: self.params.clone(),
            encoding: None,
        }
    }

    pub fn measure_noise<R: CryptoRng + RngCore>(&self, ct: &Ciphertext, rng: &mut R) -> u64 {
        // TODO: replace default simd with encoding used for ciphertext. This will require
        // adding encoding info to ciphertext
        let m = self.decrypt(ct).decode(Encoding::simd(ct.level));
        let m = Plaintext::encode(&m, &ct.params, Encoding::simd(ct.level)).to_poly();

        let mut m2 = ct.c[0].clone();
        let s = self.to_poly(ct.level);
        let mut s_carry = s.clone();
        for i in 1..ct.c.len() {
            m2 += &(&s_carry * &ct.c[i]);
            s_carry *= &s;
        }

        m2 -= &m;
        m2.change_representation(Representation::Coefficient);

        let mut noise = 0u64;
        Vec::<BigUint>::from(&m2).iter().for_each(|v| {
            noise = std::cmp::max(
                noise,
                std::cmp::min(v.bits(), (ct.c[0].context.modulus() - v).bits()),
            )
        });
        noise
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{
        distributions::{Distribution, Uniform},
        thread_rng, Rng,
    };

    #[test]
    fn test_encryption_decryption() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::new(&[60], 1153, 8));
        let sk = SecretKey::random(&params, &mut rng);

        let m = rng
            .clone()
            .sample_iter(Uniform::new(0, params.plaintext_modulus))
            .take(params.polynomial_degree)
            .collect_vec();
        let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
        let ct = sk.encrypt(&pt, &mut rng);

        dbg!(sk.measure_noise(&ct, &mut rng));

        let pt2 = sk.decrypt(&ct);
        let m2 = pt2.decode(Encoding::simd(0));
        assert_eq!(m, m2);
    }
}
