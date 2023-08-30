use crate::plaintext::{Encoding, Plaintext};
use crate::{BfvParameters, Ciphertext, PolyCache, PolyType};
use crate::{Poly, PolyContext, Representation};
use itertools::Itertools;
use rand::distributions::{Distribution, Uniform};
use rand::{CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone, PartialEq, Debug)]
pub struct SecretKey {
    pub(crate) coefficients: Box<[i64]>,
}

impl SecretKey {
    /// Generates a random secret key with fixed hamming weight `hw`.
    ///
    /// The code is adapted from [Lattigo](https://github.com/tuneinsight/lattigo)
    pub fn random<R: CryptoRng + RngCore>(degree: usize, hw: usize, rng: &mut R) -> SecretKey {
        let mut sk = vec![0i64; degree];

        // Think of indices vec as a set from which we sample `hw` indices to either set 1 or -1.
        let mut indices = (0..degree).into_iter().collect_vec();

        // We need `hw` random bits.
        let mut random_bytes = vec![0u8; (hw as f64 / 8.0).ceil() as usize];
        rng.fill_bytes(&mut random_bytes);

        let mut byte_index = 0;
        let mut random_bit_pos = 0;

        for i in 0..hw {
            // sample random index in range [0, indices.len())
            let sampled_index = rng.gen_range(0..degree - i);

            match random_bytes[byte_index] & 1 {
                0 => sk[indices[sampled_index]] = 1,
                1 => sk[indices[sampled_index]] = -1,
                _ => {
                    panic!("Impossible!")
                }
            }

            random_bytes[byte_index] >>= 1;
            random_bit_pos += 1;
            // bits at `byte_index` position have been consumed. Move to the next.
            if random_bit_pos == 8 {
                byte_index += 1;
                random_bit_pos = 0;
            }

            // removed the sampled index from `indices` set.
            indices[sampled_index] = *indices.last().unwrap();
            indices.truncate(indices.len() - 1);
        }

        SecretKey {
            coefficients: sk.into_boxed_slice(),
        }
    }

    /// Convenience wrapper around `SecretKey::random` if `BfvParameters` happens to be already
    /// initialised
    pub fn random_with_params<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        rng: &mut R,
    ) -> SecretKey {
        SecretKey::random(params.degree, params.hw, rng)
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
    fn to_poly(&self, ctx: &PolyContext<'_>) -> Poly {
        let mut p = ctx.try_convert_from_i64_small(&self.coefficients, Representation::Coefficient);
        ctx.change_representation(&mut p, Representation::Evaluation);
        p
    }

    /// Encrypts given plaintext with the secret key
    pub fn encrypt<R: CryptoRng + RngCore>(
        &self,
        params: &BfvParameters,
        pt: &Plaintext,
        rng: &mut R,
    ) -> Ciphertext {
        if pt.encoding.is_none() {
            panic!("Plaintext encoding missing!");
        }
        let encoding = pt.encoding.clone().unwrap();

        let ctx = params.poly_ctx(&PolyType::Q, encoding.level);
        let mut sk_poly = self.to_poly(&ctx);

        let m = pt.scale_plaintext(params, Representation::Evaluation);

        // seed `a`
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);
        // Since `Ciphertext` defaults to `Coefficient` representation and `a` is
        // part of `Ciphertext`, we clone and change clone's representation to `Evaluation`
        // in order to avoid another Ntt operation later.
        let mut a = ctx.random_with_seed(seed);
        let mut a_eval = a.clone();
        ctx.change_representation(&mut a_eval, Representation::Evaluation);

        // sk*a
        ctx.mul_assign(&mut sk_poly, &a_eval);

        let mut e = ctx.random_gaussian(Representation::Coefficient, params.variance, rng);
        ctx.change_representation(&mut e, Representation::Evaluation);

        // e + m
        ctx.add_assign(&mut e, &m);
        // e + m - sk*s
        ctx.sub_assign(&mut e, &sk_poly);

        ctx.change_representation(&mut e, Representation::Coefficient);

        Ciphertext {
            c: vec![e, a],
            poly_type: PolyType::Q,
            level: encoding.level,
            seed: Some(seed),
        }
    }

    pub fn decrypt(&self, ct: &Ciphertext, params: &BfvParameters) -> Plaintext {
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
            mul_poly: None,
            add_sub_poly: None,
        }
    }

    pub fn measure_noise(&self, ct: &Ciphertext, params: &BfvParameters) -> u64 {
        // TODO: replace default simd with encoding used for ciphertext. This will require
        // adding encoding info to ciphertext
        let m = self.decrypt(ct, params).decode(Encoding::default(), params);
        let scaled_m = Plaintext::encode(&m, &params, Encoding::simd(ct.level(), PolyCache::None))
            .scale_plaintext(&params, Representation::Evaluation);

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
        let sk = SecretKey::random(params.degree, params.hw, &mut rng);

        let m = rng
            .clone()
            .sample_iter(Uniform::new(0, params.plaintext_modulus))
            .take(params.degree)
            .collect_vec();
        let pt = Plaintext::encode(&m, &params, Encoding::default());
        let ct = sk.encrypt(&params, &pt, &mut rng);

        dbg!(sk.measure_noise(&ct, &params));

        let pt2 = sk.decrypt(&ct, &params);
        let m2 = pt2.decode(Encoding::default(), &params);
        assert_eq!(m, m2);
    }

    #[test]
    fn test_hamming_weight() {
        let mut rng = thread_rng();
        let sk = SecretKey::random(32768, 16384, &mut rng);

        let mut ones = 0;
        let mut nones = 0;
        let mut zeros = 0;

        sk.coefficients.iter().for_each(|c| match *c {
            0 => zeros += 1,
            1 => ones += 1,
            -1 => nones += 1,
            _ => {}
        });

        println!("ones: {ones}, nones: {nones}, zeros: {zeros}")
    }
}
