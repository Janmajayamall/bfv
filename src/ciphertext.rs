use crate::parameters::BfvParameters;
use crate::poly::{Poly, Representation};
use itertools::Itertools;
use std::sync::Arc;

pub struct Ciphertext {
    pub(crate) c: Vec<Poly>,
    pub(crate) params: Arc<BfvParameters>,
    pub(crate) level: usize,
}

impl Ciphertext {
    pub fn multiply1(&self, rhs: &Ciphertext) -> Ciphertext {
        debug_assert!(self.params == rhs.params);
        debug_assert!(self.c.len() == 2);
        debug_assert!(rhs.c.len() == 2);

        assert!(self.level == rhs.level);

        let level = self.level;

        // let mut now = std::time::Instant::now();
        let mut c00 = self.c[0].expand_crt_basis(
            &self.params.pq_poly_contexts[level],
            &self.params.extension_poly_contexts[level],
            &self.params.ql_hat_modp[level],
            &self.params.ql_hat_inv_modql[level],
            &self.params.ql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.alphal_modp[level],
        );
        let mut c01 = self.c[1].expand_crt_basis(
            &self.params.pq_poly_contexts[level],
            &self.params.extension_poly_contexts[level],
            &self.params.ql_hat_modp[level],
            &self.params.ql_hat_inv_modql[level],
            &self.params.ql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.alphal_modp[level],
        );
        // println!("Extend1 {:?}", now.elapsed());

        // now = std::time::Instant::now();
        let mut c10 = rhs.c[0].fast_expand_crt_basis_p_over_q(
            &self.params.extension_poly_contexts[level],
            &self.params.pq_poly_contexts[level],
            &self.params.neg_pql_hat_inv_modql[level],
            &self.params.neg_pql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.ql_inv_modp[level],
            &self.params.pl_hat_modq[level],
            &self.params.pl_hat_inv_modpl[level],
            &self.params.pl_hat_inv_modpl_shoup[level],
            &self.params.pl_inv[level],
            &self.params.alphal_modq[level],
        );
        let mut c11 = rhs.c[1].fast_expand_crt_basis_p_over_q(
            &self.params.extension_poly_contexts[level],
            &self.params.pq_poly_contexts[level],
            &self.params.neg_pql_hat_inv_modql[level],
            &self.params.neg_pql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.ql_inv_modp[level],
            &self.params.pl_hat_modq[level],
            &self.params.pl_hat_inv_modpl[level],
            &self.params.pl_hat_inv_modpl_shoup[level],
            &self.params.pl_inv[level],
            &self.params.alphal_modq[level],
        );
        c10.change_representation(Representation::Evaluation);
        c11.change_representation(Representation::Evaluation);
        // println!("Extend2 {:?}", now.elapsed());

        // now = std::time::Instant::now();
        // tensor
        // c00 * c10
        let mut c_r0 = &c00 * &c10;

        // c00 * c11 + c01 * c10
        c00 *= &c11;
        c10 *= &c01;
        c00 += &c10;

        // c01 * c11
        c01 *= &c11;
        // println!("Tensor {:?}", now.elapsed());

        // Scale down
        // now = std::time::Instant::now();
        c_r0.change_representation(Representation::Coefficient);
        c00.change_representation(Representation::Coefficient);
        c01.change_representation(Representation::Coefficient);
        let mut c = vec![c_r0, c00, c01]
            .iter_mut()
            .map(|p| {
                p.scale_and_round(
                    &self.params.ciphertext_poly_contexts[level],
                    &self.params.extension_poly_contexts[level],
                    &self.params.ciphertext_poly_contexts[level],
                    &self.params.tql_p_hat_inv_modp_divp_modql[level],
                    &self.params.tql_p_hat_inv_modp_divp_frac_hi[level],
                    &self.params.tql_p_hat_inv_modp_divp_frac_lo[level],
                )
            })
            .collect_vec();
        c.iter_mut().for_each(|p| {
            p.change_representation(Representation::Evaluation);
        });
        // println!("Scale Down {:?}", now.elapsed());

        Ciphertext {
            c,
            params: self.params.clone(),
            level: self.level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        plaintext::{Encoding, EncodingType, Plaintext},
        secret_key::SecretKey,
    };
    use itertools::izip;
    use ndarray::Array2;
    use num_traits::{identities::One, ToPrimitive, Zero};
    use rand::{
        distributions::{Distribution, Uniform},
        thread_rng, Rng,
    };

    #[test]
    fn test_ciphertext_multiplication1() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global()
            .unwrap();

        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::new(
            &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            65537,
            1 << 15,
        ));
        let sk = SecretKey::random(&params, &mut rng);

        let mut m1 = rng
            .clone()
            .sample_iter(Uniform::new(0, params.plaintext_modulus))
            .take(params.polynomial_degree)
            .collect_vec();
        let m2 = rng
            .clone()
            .sample_iter(Uniform::new(0, params.plaintext_modulus))
            .take(params.polynomial_degree)
            .collect_vec();
        let pt1 = Plaintext::encode(&m1, &params, Encoding::simd(0));
        let pt2 = Plaintext::encode(&m2, &params, Encoding::simd(0));
        let ct1 = sk.encrypt(&pt1, &mut rng);
        let ct2 = sk.encrypt(&pt2, &mut rng);

        let now = std::time::Instant::now();
        let ct3 = ct1.multiply1(&ct2);
        println!("total time: {:?}", now.elapsed());

        dbg!(sk.measure_noise(&ct3, &mut rng));

        params.plaintext_modulus_op.mul_mod_fast_vec(&mut m1, &m2);

        let res = sk.decrypt(&ct3).decode(Encoding {
            encoding_type: EncodingType::Simd,
            level: 0,
        });
        assert_eq!(res, m1);
    }

    #[test]
    fn clone_perf() {
        let params = Arc::new(BfvParameters::new(
            &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            65537,
            1 << 15,
        ));

        let now = std::time::Instant::now();
        let a = Poly::zero(
            &params.ciphertext_poly_contexts[0],
            &Representation::Coefficient,
        );
        println!("time: {:?}", now.elapsed());

        let mut rng = thread_rng();
        let b = Poly::random(
            &params.ciphertext_poly_contexts[0],
            &Representation::Coefficient,
            &mut rng,
        );

        let now = std::time::Instant::now();
        let c = b.clone();
        println!("time: {:?}", now.elapsed());
    }
}
