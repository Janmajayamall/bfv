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
    pub fn multiply1(&mut self, rhs: &mut Ciphertext) -> Ciphertext {
        let f = self.params.ciphertext_moduli[2];

        debug_assert!(self.params == rhs.params);
        debug_assert!(self.c.len() == 2);
        debug_assert!(rhs.c.len() == 2);

        assert!(self.level == rhs.level);

        let level = self.level;

        let mut now = std::time::Instant::now();
        let mut c1 = self
            .c
            .iter_mut()
            .map(|p| {
                p.expand_crt_basis(
                    &self.params.pq_poly_contexts[level],
                    &self.params.extension_poly_contexts[level],
                    &self.params.ql_hat_modp[level],
                    &self.params.ql_hat_inv_modql[level],
                    &self.params.ql_hat_inv_modql_shoup[level],
                    &self.params.ql_inv[level],
                    &self.params.alphal_modp[level],
                )
            })
            .collect_vec();
        println!("Extend1 {:?}", now.elapsed());

        now = std::time::Instant::now();
        let mut c2 = rhs
            .c
            .iter_mut()
            .map(|p| {
                p.change_representation(Representation::Coefficient);
                let mut p = p.fast_expand_crt_basis_p_over_q(
                    &self.params.extension_poly_contexts[level],
                    &self.params.pq_poly_contexts[level],
                    &self.params.neg_pql_hat_inv_modql[level],
                    &self.params.neg_pql_hat_inv_modql_shoup[level],
                    &self.params.ql_inv_modp[level],
                    &self.params.pl_hat_modq[level],
                    &self.params.pl_hat_inv_modpl[level],
                    &self.params.pl_hat_inv_modpl_shoup[level],
                    &self.params.pl_inv[level],
                    &self.params.alphal_modq[level],
                );
                p.change_representation(Representation::Evaluation);
                p
            })
            .collect_vec();
        println!("Extend2 {:?}", now.elapsed());

        now = std::time::Instant::now();
        // tensor
        // c1_0 * c2_0
        let c_r0 = &c1[0] * &c2[0];

        // c1_0 * c2_1 + c1_1 * c2_0
        c1[0] *= &c2[1];
        c2[0] *= &c1[1];
        c1[0] += &c2[0];

        // c1_1 * c2_1
        c1[1] *= &c2[1];

        let mut c = vec![c_r0, c1[0].clone(), c1[1].clone()];
        println!("Tensor {:?}", now.elapsed());

        now = std::time::Instant::now();
        let c = c
            .iter_mut()
            .map(|p| {
                p.change_representation(Representation::Coefficient);
                let mut p = p.scale_and_round(
                    &self.params.ciphertext_poly_contexts[level],
                    &self.params.extension_poly_contexts[level],
                    &self.params.ciphertext_poly_contexts[level],
                    &self.params.tql_p_hat_inv_modp_divp_modql[level],
                    &self.params.tql_p_hat_inv_modp_divp_frac_hi[level],
                    &self.params.tql_p_hat_inv_modp_divp_frac_lo[level],
                );
                p.change_representation(Representation::Evaluation);
                p
            })
            .collect_vec();
        println!("Scale Down {:?}", now.elapsed());

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
    use num_traits::{identities::One, ToPrimitive, Zero};
    use rand::{
        distributions::{Distribution, Uniform},
        thread_rng, Rng,
    };

    #[test]
    fn test_ciphertext_multiplication1() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::new(
            &[60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60],
            65537,
            8,
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
        let mut ct1 = sk.encrypt(&pt1, &mut rng);
        let mut ct2 = sk.encrypt(&pt2, &mut rng);

        let now = std::time::Instant::now();
        let ct3 = ct1.multiply1(&mut ct2);
        println!("time: {:?}", now.elapsed());

        dbg!(sk.measure_noise(&ct3, &mut rng));

        params.plaintext_modulus_op.mul_mod_fast_vec(&mut m1, &m2);

        let res = sk.decrypt(&ct3).decode(Encoding {
            encoding_type: EncodingType::Simd,
            level: 0,
        });
        assert_eq!(res, m1);
    }
}
