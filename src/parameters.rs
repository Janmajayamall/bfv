use crate::modulus::Modulus;
use crate::nb_theory::generate_primes_vec;
use crate::poly::{Poly, PolyContext, Representation};
use fhe_math::zq::{ntt::NttOperator, Modulus as ModulusOld};
use itertools::{izip, Itertools};
use ndarray::Array2;
use num_bigint::BigUint;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{Pow, ToPrimitive};
use std::sync::Arc;

#[derive(PartialEq, Clone, Debug)]
pub struct BfvParameters {
    pub ciphertext_moduli: Box<[u64]>,
    pub extension_moduli: Box<[u64]>,
    pub ciphertext_moduli_sizes: Box<[usize]>,
    pub ciphertext_poly_contexts: Box<[Arc<PolyContext>]>,
    pub extension_poly_contexts: Box<[Arc<PolyContext>]>,
    pub pq_poly_contexts: Box<[Arc<PolyContext>]>,

    pub plaintext_modulus: u64,
    pub plaintext_modulus_op: Modulus,
    pub plaintext_ntt_op: NttOperator,
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

    // Fast expand CRT basis Q to P to PQ
    // Fast conversion P over Q
    pub neg_pql_hat_inv_modql: Vec<Vec<u64>>,
    pub neg_pql_hat_inv_modql_shoup: Vec<Vec<u64>>,
    pub ql_inv_modp: Vec<Array2<u64>>,
    //  Switch CRT basis P to Q //
    pub pl_hat_modq: Vec<Array2<u64>>,
    pub pl_hat_inv_modpl: Vec<Vec<u64>>,
    pub pl_hat_inv_modpl_shoup: Vec<Vec<u64>>,
    pub pl_inv: Vec<Vec<f64>>,
    pub alphal_modq: Vec<Array2<u64>>,

    // Scale and Round //
    pub tql_p_hat_inv_modp_divp_modql: Vec<Array2<u64>>,
    pub tql_p_hat_inv_modp_divp_frac_hi: Vec<Vec<u64>>,
    pub tql_p_hat_inv_modp_divp_frac_lo: Vec<Vec<u64>>,

    // Switch CRT basis Q to P //
    pub ql_hat_modp: Vec<Array2<u64>>,
    pub ql_hat_inv_modql: Vec<Vec<u64>>,
    pub ql_hat_inv_modql_shoup: Vec<Vec<u64>>,
    pub ql_inv: Vec<Vec<f64>>,
    pub alphal_modp: Vec<Array2<u64>>,
    // Hybrid key switching

    // Mod Down //
    pub lastq_inv_modq: Vec<Vec<u64>>,
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
        let f = (2.0_f64).pow(60);
        dbg!(((delta * bound_error * f * 8.0) as f64 / 2.0).log2());

        (bound_error * (1.0 + 2.0 * delta * bound_key))
    }

    /// Returns noise in bits from BV key switching operation
    ///
    /// Formula for noise estimation taken from B.2.1 of https://eprint.iacr.org/2021/204.
    pub fn noise_ks(levels: usize, sigma: f64, n: usize, max_qsize: usize) -> usize {
        let alpha = 36_f64;
        let bound_error = alpha.sqrt() * sigma;

        let delta = 2.0 * (n as f64).sqrt();

        ((delta * bound_error * (2_f64).pow(max_qsize as f64) * (levels as f64 + 1_f64)) / 2.0)
            .log2() as usize
    }

    /// creates new bfv parameteres with necessary values
    pub fn new(
        ciphertext_moduli_sizes: &[usize],
        plaintext_modulus: u64,
        polynomial_degree: usize,
    ) -> BfvParameters {
        // generate primes
        let ciphertext_moduli =
            generate_primes_vec(ciphertext_moduli_sizes, polynomial_degree, &[]);

        // generate extension modulus P
        // P = Q
        let extension_moduli = generate_primes_vec(
            ciphertext_moduli_sizes,
            polynomial_degree,
            &ciphertext_moduli,
        );

        // create contexts for all levels
        let moduli_count = ciphertext_moduli.len();
        let mut poly_contexts = vec![];
        let mut extension_poly_contexts = vec![];
        let mut pq_poly_contexts = vec![];
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

            let pq_at_level = [extension_moduli_at_level, moduli_at_level].concat();
            pq_poly_contexts.push(Arc::new(PolyContext::new(
                pq_at_level.as_slice(),
                polynomial_degree,
            )));
        }

        // ENCRYPTION //
        let mut ql_modt = vec![];
        let mut neg_t_inv_modql = vec![];
        poly_contexts.iter().for_each(|poly_context| {
            let q = poly_context.modulus();
            let q_dig = poly_context.modulus_dig();

            // [Q]_t
            ql_modt.push((&q % plaintext_modulus).to_u64().unwrap());

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
            // dbg!(&neg_t_inv_modq.coefficients);
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

        // Fast expand CRT basis Q to P to PQ
        // 1. Fast Conv P Over Q //
        let mut neg_pql_hat_inv_modql = vec![];
        let mut neg_pql_hat_inv_modql_shoup = vec![];
        let mut ql_inv_modp = vec![];
        izip!(poly_contexts.iter(), extension_poly_contexts.iter()).for_each(
            |(q_context, p_context)| {
                let q = q_context.modulus();
                let q_dig = q_context.modulus_dig();
                let p = p_context.modulus();

                let mut neg_pq_hat_inv_modq = vec![];
                let mut neg_pq_hat_inv_modq_shoup = vec![];

                izip!(q_context.moduli_ops.iter()).for_each(|modqi| {
                    let qi = modqi.modulus();
                    let q_hat_inv_modqi = BigUint::from_bytes_le(
                        &(&q_dig / qi)
                            .mod_inverse(BigUintDig::from(qi))
                            .unwrap()
                            .to_biguint()
                            .unwrap()
                            .to_bytes_le(),
                    );
                    let tmp = ((qi - ((&p * q_hat_inv_modqi) % qi)) % qi)
                        .to_u64()
                        .unwrap();
                    neg_pq_hat_inv_modq.push(tmp);
                    neg_pq_hat_inv_modq_shoup.push(modqi.compute_shoup(tmp));
                });

                let mut q_inv_modp = vec![];
                p_context.moduli_ops.iter().for_each(|modpj| {
                    q_context.moduli.iter().for_each(|qi| {
                        q_inv_modp.push(modpj.inv(*qi % modpj.modulus()));
                    });
                });

                neg_pql_hat_inv_modql.push(neg_pq_hat_inv_modq);
                neg_pql_hat_inv_modql_shoup.push(neg_pq_hat_inv_modq_shoup);
                ql_inv_modp.push(
                    Array2::from_shape_vec(
                        (p_context.moduli.len(), q_context.moduli.len()),
                        q_inv_modp,
                    )
                    .unwrap(),
                );
            },
        );
        // 2. Switch CRT basis P to Q //
        let mut pl_hat_modq = vec![];
        let mut pl_hat_inv_modpl = vec![];
        let mut pl_hat_inv_modpl_shoup = vec![];
        let mut pl_inv = vec![];
        let mut alphal_modq = vec![];
        izip!(poly_contexts.iter(), extension_poly_contexts.iter()).for_each(
            |(q_context, p_context)| {
                let p = p_context.modulus();
                let p_dig = p_context.modulus_dig();

                let mut p_hat_inv_modp = vec![];
                let mut p_hat_inv_modp_shoup = vec![];
                let mut p_inv = vec![];
                p_context.moduli_ops.iter().for_each(|modpi| {
                    let pi = modpi.modulus();
                    let pihat_inv = BigUint::from_bytes_le(
                        &(&p_dig / pi)
                            .mod_inverse(BigUintDig::from(pi))
                            .unwrap()
                            .to_biguint()
                            .unwrap()
                            .to_bytes_le(),
                    )
                    .to_u64()
                    .unwrap();
                    p_hat_inv_modp.push(pihat_inv);
                    p_hat_inv_modp_shoup.push(modpi.compute_shoup(pihat_inv));
                    p_inv.push(1.0 / (pi as f64));
                });

                let mut alpha_modp = vec![];
                for i in 0..(p_context.moduli.len() + 1) {
                    let u_p = &p * i;
                    q_context.moduli.iter().for_each(|qi| {
                        alpha_modp.push((&u_p % *qi).to_u64().unwrap());
                    });
                }

                let mut p_hat_modq = vec![];
                q_context.moduli.iter().for_each(|qi| {
                    p_context.moduli.iter().for_each(|pi| {
                        p_hat_modq.push(((&p / pi) % qi).to_u64().unwrap());
                    })
                });

                pl_hat_modq.push(
                    Array2::from_shape_vec(
                        (q_context.moduli.len(), p_context.moduli.len()),
                        p_hat_modq,
                    )
                    .unwrap(),
                );
                pl_hat_inv_modpl.push(p_hat_inv_modp);
                pl_hat_inv_modpl_shoup.push(p_hat_inv_modp_shoup);
                pl_inv.push(p_inv);
                alphal_modq.push(
                    Array2::from_shape_vec(
                        (p_context.moduli.len() + 1, q_context.moduli.len()),
                        alpha_modp,
                    )
                    .unwrap(),
                )
            },
        );

        // Scale and Round //
        let mut tql_p_hat_inv_modp_divp_modql = vec![];
        let mut tql_p_hat_inv_modp_divp_frac_hi = vec![];
        let mut tql_p_hat_inv_modp_divp_frac_lo = vec![];
        izip!(
            poly_contexts.iter(),
            extension_poly_contexts.iter(),
            pq_poly_contexts.iter()
        )
        .for_each(|(q_context, p_context, pq_context)| {
            let pq = pq_context.modulus();
            let pq_dig = pq_context.modulus_dig();
            let q = q_context.modulus();
            let t = plaintext_modulus;

            let mut tq_p_hat_inv_modp_divp_modq = vec![];
            let mut tq_p_hat_inv_modp_divp_frac_hi = vec![];
            let mut tq_p_hat_inv_modp_divp_frac_lo = vec![];
            q_context.moduli.iter().for_each(|qi| {
                p_context.moduli.iter().for_each(|pi| {
                    let tq_p_hat_inv_modp = BigUint::from_bytes_le(
                        &(&pq_dig / pi)
                            .mod_inverse(BigUintDig::from(*pi))
                            .unwrap()
                            .to_biguint()
                            .unwrap()
                            .to_bytes_le(),
                    ) * t
                        * &q;
                    let rational = ((&tq_p_hat_inv_modp / *pi) % *qi).to_u64().unwrap();

                    // let mut frac = tq_p_hat_inv_modp % pi;
                    // frac <<= 127;
                    // frac /= *pi;
                    // let frac = frac.to_u128().unwrap();
                    // let frac_hi = (frac >> 64) as u64;
                    // let frac_lo = (frac - ((frac_hi as u128) << 64)) as u64;

                    tq_p_hat_inv_modp_divp_modq.push(rational);
                    // tq_p_hat_inv_modp_divp_frac_hi.push(frac_hi);
                    // tq_p_hat_inv_modp_divp_frac_lo.push(frac_lo);
                });

                let tq_qi_hat_inv_modqi = (BigUint::from_bytes_le(
                    &(&pq_dig / qi)
                        .mod_inverse(BigUintDig::from(*qi))
                        .unwrap()
                        .to_biguint()
                        .unwrap()
                        .to_bytes_le(),
                ) * t
                    * &q);
                tq_p_hat_inv_modp_divp_modq
                    .push(((tq_qi_hat_inv_modqi / *qi) % *qi).to_u64().unwrap());
            });

            p_context.moduli.iter().for_each(|pi| {
                let tq_p_hat_inv_modp = BigUint::from_bytes_le(
                    &(&pq_dig / pi)
                        .mod_inverse(BigUintDig::from(*pi))
                        .unwrap()
                        .to_biguint()
                        .unwrap()
                        .to_bytes_le(),
                ) * t
                    * &q;

                let mut frac = tq_p_hat_inv_modp % pi;
                frac <<= 127;
                frac /= *pi;
                let frac = frac.to_u128().unwrap();
                let frac_hi = (frac >> 64) as u64;
                let frac_lo = (frac - ((frac_hi as u128) << 64)) as u64;

                tq_p_hat_inv_modp_divp_frac_hi.push(frac_hi);
                tq_p_hat_inv_modp_divp_frac_lo.push(frac_lo);
            });

            tql_p_hat_inv_modp_divp_modql.push(
                Array2::from_shape_vec(
                    (q_context.moduli.len(), p_context.moduli.len() + 1),
                    tq_p_hat_inv_modp_divp_modq,
                )
                .unwrap(),
            );
            tql_p_hat_inv_modp_divp_frac_hi.push(tq_p_hat_inv_modp_divp_frac_hi);
            tql_p_hat_inv_modp_divp_frac_lo.push(tq_p_hat_inv_modp_divp_frac_lo);
        });

        // Switch CRT basis Q to P //
        let mut ql_hat_modp = vec![];
        let mut ql_hat_inv_modql = vec![];
        let mut ql_hat_inv_modql_shoup = vec![];
        let mut ql_inv = vec![];
        let mut alphal_modp = vec![];
        izip!(poly_contexts.iter(), extension_poly_contexts.iter()).for_each(
            |(q_context, p_context)| {
                let q = q_context.modulus();
                let q_dig = q_context.modulus_dig();

                let mut q_hat_inv_modq = vec![];
                let mut q_hat_inv_modq_shoup = vec![];
                let mut q_inv = vec![];
                q_context.moduli_ops.iter().for_each(|modqi| {
                    let qi = modqi.modulus();
                    let qihat_inv = BigUint::from_bytes_le(
                        &(&q_dig / qi)
                            .mod_inverse(BigUintDig::from(qi))
                            .unwrap()
                            .to_biguint()
                            .unwrap()
                            .to_bytes_le(),
                    )
                    .to_u64()
                    .unwrap();
                    q_hat_inv_modq.push(qihat_inv);
                    q_hat_inv_modq_shoup.push(modqi.compute_shoup(qihat_inv));
                    q_inv.push(1.0 / (qi as f64));
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
                ql_hat_inv_modql_shoup.push(q_hat_inv_modq_shoup);
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

        // Mod down next //
        let mut lastq_inv_modq = vec![];
        poly_contexts.iter().for_each(|ctx| {
            let lastq = ctx.moduli.last().unwrap();
            let tmp = ctx.moduli_ops[..ctx.moduli.len() - 1]
                .iter()
                .map(|modqi| modqi.inv(*lastq))
                .collect_vec();
            lastq_inv_modq.push(tmp);
        });

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

        let plaintext_modulus_op = Modulus::new(plaintext_modulus);

        // TODO: change ModulusOld with Modulus
        let plaintext_ntt_op = NttOperator::new(
            &ModulusOld::new(plaintext_modulus).unwrap(),
            polynomial_degree,
        )
        .unwrap();

        BfvParameters {
            ciphertext_moduli: ciphertext_moduli.into_boxed_slice(),
            extension_moduli: extension_moduli.into_boxed_slice(),
            ciphertext_moduli_sizes: ciphertext_moduli_sizes.to_vec().into_boxed_slice(),
            ciphertext_poly_contexts: poly_contexts.into_boxed_slice(),
            extension_poly_contexts: extension_poly_contexts.into_boxed_slice(),
            pq_poly_contexts: pq_poly_contexts.into_boxed_slice(),

            plaintext_modulus,
            plaintext_modulus_op,
            plaintext_ntt_op,
            polynomial_degree,

            // ENCRYPTION //
            ql_modt,
            neg_t_inv_modql,

            // DECRYPTION //
            t_qlhat_inv_modql_divql_modt,
            t_bqlhat_inv_modql_divql_modt,
            t_qlhat_inv_modql_divql_frac,
            t_bqlhat_inv_modql_divql_frac,

            // Fast expand CRT basis Q to P to PQ
            neg_pql_hat_inv_modql,
            neg_pql_hat_inv_modql_shoup,
            ql_inv_modp,
            pl_hat_modq,
            pl_hat_inv_modpl,
            pl_hat_inv_modpl_shoup,
            pl_inv,
            alphal_modq,

            // Scale and Round //
            tql_p_hat_inv_modp_divp_modql,
            tql_p_hat_inv_modp_divp_frac_hi,
            tql_p_hat_inv_modp_divp_frac_lo,

            // Switch CRT basis Q to P //
            ql_hat_modp,
            ql_hat_inv_modql,
            ql_hat_inv_modql_shoup,
            ql_inv,
            alphal_modp,
            max_bit_size_by2: b,
            matrix_reps_index_map,

            // Mod down next //
            lastq_inv_modq,
        }
    }

    pub fn default(moduli_count: usize, polynomial_degree: usize) -> Self {
        BfvParameters::new(&vec![60; moduli_count], 65537, polynomial_degree)
    }

    pub fn ciphertext_ctx_at_level(&self, level: usize) -> Arc<PolyContext> {
        self.ciphertext_poly_contexts[level].clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_params_size() {
        let params = BfvParameters::new(
            &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            65537,
            1 << 15,
        );
        dbg!(std::mem::size_of_val(&params));
    }
}
