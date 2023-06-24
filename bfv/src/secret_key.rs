use crate::ciphertext::Ciphertext;
use crate::parameters::{BfvParameters, PolyType};
use crate::plaintext::{Encoding, Plaintext};
use crate::poly::{Poly, Representation};
use crate::PolyContext;
use itertools::Itertools;
use num_bigint::BigUint;
use rand::distributions::{Distribution, Uniform};
use rand::{CryptoRng, RngCore};
use std::sync::Arc;
use traits::Ntt;

pub struct SecretKey {
    pub(crate) coefficients: Box<[i64]>,
}

impl SecretKey {
    /// Generates a random secret key
    pub fn random<R: CryptoRng + RngCore>(degree: usize, rng: &mut R) -> SecretKey {
        let coefficients = Uniform::new(-1, 2)
            .sample_iter(rng)
            .take(degree)
            .collect_vec()
            .into_boxed_slice();

        SecretKey { coefficients }
    }

    /// Creates a new secret key with given coefficients.
    ///
    /// Panics if coefficients length does not match with degree of given bfv parameter
    ///
    /// Panics if each value in coefficients does not belong to ternary distribution (ie {-1,0,1}).
    pub fn new(coefficients: Vec<i64>, degree: usize) -> SecretKey {
        assert!(coefficients.len() == degree);
        coefficients.iter().for_each(|c| {
            assert!(-1 <= *c && 1 >= *c);
        });

        SecretKey {
            coefficients: coefficients.into_boxed_slice(),
        }
    }

    /// Returns secret key polynomial for polynomial context at given level in Evaluation form
    fn to_poly<T: Ntt>(&self, ctx: &PolyContext<'_, T>) -> Poly {
        let mut p = ctx.try_convert_from_i64_small(&self.coefficients, Representation::Coefficient);
        ctx.change_representation(&mut p, Representation::Evaluation);
        p
    }

    /// Encrypts given plaintext with the secret key
    pub fn encrypt<T: Ntt, R: CryptoRng + RngCore>(
        &self,
        params: &BfvParameters<T>,
        pt: &Plaintext,
        rng: &mut R,
    ) -> Ciphertext {
        if pt.encoding.is_none() {
            panic!("Plaintext encoding missing!");
        }
        let encoding = pt.encoding.clone().unwrap();

        let ctx = params.poly_ctx(&PolyType::Q, encoding.level);
        let mut sk_poly = self.to_poly(&ctx);

        let m = pt.to_poly(params);
        let mut a = ctx.random(Representation::Evaluation, rng);

        // sk*a
        ctx.mul_assign(&mut sk_poly, &a);

        let mut e = ctx.random_gaussian(Representation::Coefficient, 10, rng);
        ctx.change_representation(&mut e, Representation::Evaluation);

        // e + m
        ctx.add_assign(&mut e, &m);
        // e + m - sk*s
        ctx.sub_assign(&mut e, &sk_poly);

        ctx.change_representation(&mut e, Representation::Coefficient);
        ctx.change_representation(&mut a, Representation::Coefficient);

        Ciphertext {
            c: vec![e, a],
            poly_type: PolyType::Q,
            level: encoding.level,
        }
    }

    pub fn decrypt<T: Ntt>(&self, ct: &Ciphertext, params: &BfvParameters<T>) -> Plaintext {
        // Panic on empty ciphertext
        assert!(ct.c.len() != 0);
        assert!(ct.poly_type == PolyType::Q);

        let ctx = params.poly_ctx(&ct.poly_type, ct.level);

        let mut m = ct.c[0].clone();
        ctx.change_representation(&mut m, Representation::Evaluation);

        let s = self.to_poly(&ctx);
        let mut s_carry = s.clone();
        for i in 1..ct.c.len() {
            if ct.c[i].representation == Representation::Evaluation {
                ctx.add_assign(&mut m, &ctx.mul(&s_carry, &ct.c[i]));
            } else {
                let mut tmp = ct.c[i].clone();
                ctx.change_representation(&mut tmp, Representation::Evaluation);
                ctx.mul_assign(&mut tmp, &s_carry);
                ctx.add_assign(&mut m, &tmp);
            }
            ctx.mul_assign(&mut s_carry, &s);
        }

        ctx.change_representation(&mut m, Representation::Coefficient);
        let m = ctx.scale_and_round_decryption(
            &m,
            &params.plaintext_modulus_op,
            params.max_bit_size_by2,
            &params.t_ql_hat_inv_modql_divql_modt[ct.level],
            &params.t_bql_hat_inv_modql_divql_modt[ct.level],
            &params.t_ql_hat_inv_modql_divql_frac[ct.level],
            &params.t_bql_hat_inv_modql_divql_frac[ct.level],
        );
        Plaintext {
            m,
            encoding: None,
            poly_ntt: None,
        }
    }

    pub fn measure_noise<T: Ntt, R: CryptoRng + RngCore>(
        &self,
        ct: &Ciphertext,
        params: &BfvParameters<T>,
        rng: &mut R,
    ) -> u64 {
        // TODO: replace default simd with encoding used for ciphertext. This will require
        // adding encoding info to ciphertext
        let m = self
            .decrypt(ct, params)
            .decode(Encoding::simd(ct.level), params);
        let scaled_m = Plaintext::encode(&m, &params, Encoding::simd(ct.level)).to_poly(&params);

        let ctx = params.poly_ctx(&ct.poly_type, ct.level);

        let mut m = ct.c[0].clone();
        ctx.change_representation(&mut m, Representation::Evaluation);
        let s = self.to_poly(&ctx);
        let mut s_carry = s.clone();
        for i in 1..ct.c.len() {
            if ct.c[i].representation == Representation::Evaluation {
                ctx.add_assign(&mut m, &ctx.mul(&s_carry, &ct.c[i]));
            } else {
                let mut tmp = ct.c[i].clone();
                ctx.change_representation(&mut tmp, Representation::Evaluation);
                ctx.mul_assign(&mut tmp, &s_carry);
                ctx.add_assign(&mut m, &tmp);
            }
            ctx.mul_assign(&mut s_carry, &s);
        }

        ctx.sub_assign(&mut m, &scaled_m);
        ctx.change_representation(&mut m, Representation::Coefficient);

        let mut noise = 0u64;
        ctx.try_convert_to_biguint(&m).iter().for_each(|v| {
            noise = std::cmp::max(noise, std::cmp::min(v.bits(), (ctx.big_q() - v).bits()))
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
        let params = BfvParameters::default(1, 1 << 4);
        let sk = SecretKey::random(params.degree, &mut rng);

        let m = rng
            .clone()
            .sample_iter(Uniform::new(0, params.plaintext_modulus))
            .take(params.degree)
            .collect_vec();
        let pt = Plaintext::encode(&m, &params, Encoding::simd(0));
        let ct = sk.encrypt(&params, &pt, &mut rng);

        dbg!(sk.measure_noise(&ct, &params, &mut rng));

        let pt2 = sk.decrypt(&ct, &params);
        let m2 = pt2.decode(Encoding::simd(0), &params);
        assert_eq!(m, m2);
    }
}
