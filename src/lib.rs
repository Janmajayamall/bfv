use fhe_math::zq::Modulus;
use itertools::{izip, Itertools};
use nb_theory::generate_prime;
use ndarray::{Array2, MathCell};
use num_bigint::BigUint;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::ToPrimitive;
use poly::{Poly, PolyContext, Representation};
use rand::{distributions::Uniform, prelude::Distribution, CryptoRng, RngCore};
use std::sync::Arc;

mod nb_theory;
mod poly;

/// Stores all the pre-computation
/// values.
///
/// 1. Poly Contexts of all levels
/// 2. pre-computations at all level
/// 3.
#[derive(PartialEq)]
struct BfvParameters {
    ciphertext_moduli: Vec<u64>,
    extension_moduli: Vec<u64>,
    ciphertext_moduli_sizes: Vec<usize>,
    pub ciphertext_poly_contexts: Vec<Arc<PolyContext>>,
    pub extension_poly_contexts: Vec<Arc<PolyContext>>,

    pub plaintext_modulus: u64,
    pub plaintext_modulus_op: Modulus,
    pub polynomial_degree: usize,

    // Encryption
    pub ql_modt: Vec<u64>,
    pub neg_t_inv_modql: Vec<Poly>,
    pub matrix_reps_index_map: Vec<usize>,

    // Decryption
    pub t_qlhat_inv_modql_divql_modt: Vec<Vec<u64>>,
    pub t_bqlhat_inv_modql_divql_modt: Vec<Vec<u64>>,
    pub t_qlhat_inv_modql_divql_frac: Vec<Vec<f64>>,
    pub t_bqlhat_inv_modql_divql_frac: Vec<Vec<f64>>,
    pub max_bit_size_by2: usize,

    // Fast conversion P over Q
    pub neg_pql_hat_inv_modql: Vec<Vec<u64>>,
    pub ql_inv_modp: Vec<Array2<u64>>,

    // Switch CRT basis Q to P //
    pub ql_hat_modp: Vec<Array2<u64>>,
    pub ql_hat_inv_modql: Vec<Vec<u64>>,
    pub ql_inv: Vec<Vec<f64>>,
    pub alphal_modp: Vec<Array2<u64>>,
}

impl BfvParameters {
    /// Noise of fresh ciphertext
    pub fn v_norm(sigma: f64, n: usize) -> f64 {
        let alpha: f64 = 36.0;

        // Bound of secret key. We set it to 1 since secret key coefficients are sampled from ternary distribution
        let bound_key = 1.0;

        // Bound of error. Error is sampled from gaussian distribution
        let bound_error = alpha.sqrt() * sigma;

        // expansion factor delta
        let delta = 2.0 * (n as f64).sqrt();

        (bound_error * (1.0 + 2.0 * delta * bound_key)).log2()
    }

    /// creates new bfv parameteres with necessary values
    pub fn new(
        ciphertext_moduli_sizes: &[usize],
        plaintext_modulus: u64,
        polynomial_degree: usize,
    ) -> BfvParameters {
        // generate primes
        let mut ciphertext_moduli = vec![];
        ciphertext_moduli_sizes.iter().for_each(|size| {
            let mut upper_bound = 1u64 << size;
            loop {
                if let Some(prime) =
                    generate_prime(*size, 2 * polynomial_degree as u64, upper_bound)
                {
                    if !ciphertext_moduli.contains(&prime) {
                        ciphertext_moduli.push(prime);
                        break;
                    } else {
                        upper_bound = prime;
                    }
                } else {
                    // not enough primes
                    panic!("Not enough primes!");
                }
            }
        });

        // generate extension modulus P
        let mut extension_moduli = vec![];
        ciphertext_moduli_sizes.iter().for_each(|size| {
            let mut upper_bound = 1u64 << size;
            loop {
                if let Some(prime) =
                    generate_prime(*size, 2 * polynomial_degree as u64, upper_bound)
                {
                    if !ciphertext_moduli.contains(&prime) && !extension_moduli.contains(&prime) {
                        extension_moduli.push(prime);
                        break;
                    } else {
                        upper_bound = prime;
                    }
                } else {
                    // not enough primes
                    panic!("Not enough primes!");
                }
            }
        });

        // create contexts for all levels
        let moduli_count = ciphertext_moduli.len();
        let mut poly_contexts = vec![];
        let mut extension_poly_contexts = vec![];
        for i in 0..moduli_count {
            let moduli_at_level = ciphertext_moduli[..moduli_count - i].to_vec();
            let extension_moduli_at_level = extension_moduli[..moduli_count - i].to_vec();
            poly_contexts.push(Arc::new(PolyContext::new(
                moduli_at_level.as_slice(),
                polynomial_degree,
            )));
            extension_poly_contexts.push(Arc::new(PolyContext::new(
                extension_moduli_at_level.as_slice(),
                polynomial_degree,
            )));
        }

        // ENCRYPTION //
        let mut ql_modt = vec![];
        let mut neg_t_inv_modql = vec![];
        poly_contexts.iter().for_each(|poly_context| {
            let q = poly_context.modulus();
            let q_dig = poly_context.modulus_dig();

            // [Q * t]_t
            ql_modt.push((q % plaintext_modulus).to_u64().unwrap());

            // [(-t)^-1]_Q
            let neg_t_inv_modq = BigUint::from_bytes_le(
                &(&q_dig - plaintext_modulus)
                    .mod_inverse(q_dig)
                    .unwrap()
                    .to_biguint()
                    .unwrap()
                    .to_bytes_le(),
            );
            let mut neg_t_inv_modq = Poly::try_convert_from_biguint(
                &[neg_t_inv_modq],
                poly_context,
                &Representation::Coefficient,
            );
            neg_t_inv_modq.change_representation(Representation::Evaluation);
            neg_t_inv_modql.push(neg_t_inv_modq);
        });

        // DECRYPTION //
        // Pre computation for decryption
        let b = ciphertext_moduli_sizes.iter().max().unwrap() / 2;
        let mut t_qlhat_inv_modql_divql_modt = vec![];
        let mut t_bqlhat_inv_modql_divql_modt = vec![];
        let mut t_qlhat_inv_modql_divql_frac = vec![];
        let mut t_bqlhat_inv_modql_divql_frac = vec![];
        poly_contexts.iter().for_each(|poly_context| {
            let ql = poly_context.modulus();
            let ql_dig = poly_context.modulus_dig();

            let mut rationals = vec![];
            let mut brationals = vec![];
            let mut fractionals = vec![];
            let mut bfractionals = vec![];

            poly_context.moduli.iter().for_each(|qi| {
                // [qihat_inv]_qi
                let qihat_inv = BigUint::from_bytes_le(
                    &(&ql_dig / qi)
                        .mod_inverse(BigUintDig::from(*qi))
                        .unwrap()
                        .to_biguint()
                        .unwrap()
                        .to_bytes_le(),
                );

                // [round((t * qihat_inv_modq) / qi)]_t
                let rational = (((&qihat_inv * plaintext_modulus) / qi) % plaintext_modulus)
                    .to_u64()
                    .unwrap();
                let brational = (((((&qihat_inv * (1u64 << b)) % qi) * plaintext_modulus) / qi)
                    % plaintext_modulus)
                    .to_u64()
                    .unwrap();

                // ((t * qihat_inv_modqi) % qi) / qi
                let fractional = ((&qihat_inv * plaintext_modulus) % qi).to_f64().unwrap()
                    / qi.to_f64().unwrap();
                let bfractional = ((((&qihat_inv * (1u64 << b)) % qi) * plaintext_modulus) % qi)
                    .to_f64()
                    .unwrap()
                    / qi.to_f64().unwrap();

                rationals.push(rational);
                brationals.push(brational);
                fractionals.push(fractional);
                bfractionals.push(bfractional);
            });

            t_qlhat_inv_modql_divql_modt.push(rationals);
            t_bqlhat_inv_modql_divql_modt.push(brationals);
            t_qlhat_inv_modql_divql_frac.push(fractionals);
            t_bqlhat_inv_modql_divql_frac.push(bfractionals)
        });

        // Fast Conv P Over Q //
        let mut neg_pql_hat_inv_modql = vec![];
        let mut ql_inv_modp = vec![];
        izip!(poly_contexts.iter(), extension_poly_contexts.iter()).for_each(
            |(q_context, p_context)| {
                let q = q_context.modulus();
                let q_dig = q_context.modulus_dig();
                let p = p_context.modulus();

                let mut neg_pq_hat_inv_modq = vec![];
                let mut qi_inv_modp = vec![];

                izip!(q_context.moduli.iter()).for_each(|qi| {
                    let q_hat_inv_modqi = BigUint::from_bytes_le(
                        &(&q_dig / qi)
                            .mod_inverse(BigUintDig::from(*qi))
                            .unwrap()
                            .to_biguint()
                            .unwrap()
                            .to_bytes_le(),
                    );
                    neg_pq_hat_inv_modq.push(
                        ((qi - ((&p * q_hat_inv_modqi) % qi)) % qi)
                            .to_u64()
                            .unwrap(),
                    );

                    p_context
                        .moduli_ops
                        .iter()
                        .for_each(|pi| qi_inv_modp.push(pi.inv(*qi % pi.modulus()).unwrap()));
                });

                neg_pql_hat_inv_modql.push(neg_pq_hat_inv_modq);
                ql_inv_modp.push(
                    Array2::from_shape_vec(
                        (q_context.moduli.len(), p_context.moduli.len()),
                        qi_inv_modp,
                    )
                    .unwrap(),
                );
            },
        );

        // Switch CRT basis Q to P //
        let mut ql_hat_modp = vec![];
        let mut ql_hat_inv_modql = vec![];
        let mut ql_inv = vec![];
        let mut alphal_modp = vec![];
        izip!(poly_contexts.iter(), extension_poly_contexts.iter()).for_each(
            |(q_context, p_context)| {
                let q = q_context.modulus();
                let q_dig = q_context.modulus_dig();

                let mut q_hat_inv_modq = vec![];
                let mut q_inv = vec![];
                q_context.moduli.iter().for_each(|qi| {
                    let qihat_inv = BigUint::from_bytes_le(
                        &(&q_dig / qi)
                            .mod_inverse(BigUintDig::from(*qi))
                            .unwrap()
                            .to_biguint()
                            .unwrap()
                            .to_bytes_le(),
                    )
                    .to_u64()
                    .unwrap();
                    q_hat_inv_modq.push(qihat_inv);
                    q_inv.push(1.0 / (*qi as f64));
                });

                let mut alpha_modp = vec![];
                for i in 0..(q_context.moduli.len() + 1) {
                    let u_q = &q * i;
                    p_context.moduli.iter().for_each(|pi| {
                        alpha_modp.push((&u_q % *pi).to_u64().unwrap());
                    });
                }

                let mut q_hat_modp = vec![];
                p_context.moduli.iter().for_each(|pi| {
                    q_context.moduli.iter().for_each(|qi| {
                        q_hat_modp.push(((&q / qi) % pi).to_u64().unwrap());
                    })
                });

                ql_hat_modp.push(
                    Array2::from_shape_vec(
                        (p_context.moduli.len(), q_context.moduli.len()),
                        q_hat_modp,
                    )
                    .unwrap(),
                );
                ql_hat_inv_modql.push(q_hat_inv_modq);
                ql_inv.push(q_inv);
                alphal_modp.push(
                    Array2::from_shape_vec(
                        (q_context.moduli.len() + 1, p_context.moduli.len()),
                        alpha_modp,
                    )
                    .unwrap(),
                )
            },
        );

        // To generate mapping for matrix representation index, we use: https://github.com/microsoft/SEAL/blob/82b07db635132e297282649e2ab5908999089ad2/native/src/seal/batchencoder.cpp
        let row = polynomial_degree >> 1;
        let m = polynomial_degree << 1;
        let gen = 3;
        let mut pos = 1;
        let mut matrix_reps_index_map = vec![0usize; polynomial_degree];
        for i in 0..row {
            let index1 = (pos - 1) >> 1;
            let index2 = (m - pos - 1) >> 1;
            matrix_reps_index_map[i] =
                index1.reverse_bits() >> (polynomial_degree.leading_zeros() + 1);
            matrix_reps_index_map[i | row] =
                index2.reverse_bits() >> (polynomial_degree.leading_zeros() + 1);
            pos *= gen;
            pos &= m - 1;
        }

        BfvParameters {
            ciphertext_moduli,
            extension_moduli,
            ciphertext_moduli_sizes: ciphertext_moduli_sizes.to_vec(),
            ciphertext_poly_contexts: poly_contexts,
            extension_poly_contexts,
            plaintext_modulus,
            plaintext_modulus_op: Modulus::new(plaintext_modulus).unwrap(),
            polynomial_degree,
            ql_modt,
            neg_t_inv_modql,
            t_qlhat_inv_modql_divql_modt,
            t_bqlhat_inv_modql_divql_modt,
            t_qlhat_inv_modql_divql_frac,
            t_bqlhat_inv_modql_divql_frac,
            neg_pql_hat_inv_modql,
            ql_inv_modp,
            ql_hat_modp,
            ql_hat_inv_modql,
            ql_inv,
            alphal_modp,
            max_bit_size_by2: b,
            matrix_reps_index_map,
        }
    }
}

struct Ciphertext {
    c: Vec<Poly>,
    params: Arc<BfvParameters>,
    level: usize,
}

struct SecretKey {
    coefficients: Box<[i64]>,
    params: Arc<BfvParameters>,
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
        assert!(coefficients.len() != params.polynomial_degree);
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
        let mut p =
            Poly::try_convert_from_i64(&self.coefficients, &context, &Representation::Coefficient);
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
        debug_assert!(ct.params == self.params);

        // Panic on empty ciphertext
        assert!(ct.c.len() != 0);

        let mut m = ct.c[0].clone();
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

#[derive(PartialEq, Clone)]
enum EncodingType {
    Simd,
    Poly,
}

#[derive(Clone)]
struct Encoding {
    pub encoding_type: EncodingType,
    pub level: usize,
}

impl Encoding {
    pub fn simd(level: usize) -> Encoding {
        Encoding {
            encoding_type: EncodingType::Simd,
            level,
        }
    }
}

struct Plaintext {
    m: Vec<u64>,
    params: Arc<BfvParameters>,
    encoding: Option<Encoding>,
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

        Plaintext {
            m: m1,
            params: params.clone(),
            encoding: Some(encoding),
        }
    }

    pub fn decode(&self, encoding: Encoding) -> Vec<u64> {
        assert!(self.encoding.is_none());

        let mut m = vec![0u64; self.params.polynomial_degree];
        for i in (0..self.params.polynomial_degree) {
            if encoding.encoding_type == EncodingType::Simd {
                m[i] = self.m[self.params.matrix_reps_index_map[i]];
            } else {
                m[i] = self.m[i];
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
                let modt = Modulus::new(self.params.plaintext_modulus).unwrap();
                let mut m = self.m.clone();
                modt.scalar_mul_vec(&mut m, self.params.ql_modt[encoding.level]);

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

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::izip;
    use num_traits::{identities::One, ToPrimitive, Zero};
    use rand::{thread_rng, Rng};

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

    #[test]
    fn trial() {
        dbg!(BfvParameters::v_norm(3.2, 8));
    }
}
