use crate::relinearization_key::RelinearizationKey;
use crate::{BfvParameters, Ciphertext, EvaluationKey, PolyType};
use crate::{Encoding, GaloisKey, Plaintext, SecretKey};
use crate::{Poly, Representation};
use itertools::{izip, Itertools};
use num_bigint::{BigUint, RandBigInt};
use rand::{thread_rng, CryptoRng, Rng, RngCore};

pub struct Evaluator {
    pub(crate) params: BfvParameters,
}

impl Evaluator {
    pub fn new(params: BfvParameters) -> Evaluator {
        Evaluator { params }
    }

    pub fn params(&self) -> &BfvParameters {
        &self.params
    }

    pub fn ciphertext_change_representation(&self, c0: &mut Ciphertext, to: Representation) {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);
        c0.c.iter_mut().for_each(|p| {
            ctx.change_representation(p, to.clone());
        });
    }

    pub fn mul(&self, lhs: &Ciphertext, rhs: &Ciphertext) -> Ciphertext {
        let mut res = self.mul_lazy(lhs, rhs);
        self.scale_and_round(&mut res)
    }

    pub fn mul_lazy(&self, lhs: &Ciphertext, rhs: &Ciphertext) -> Ciphertext {
        assert!(lhs.c.len() == 2);
        assert!(rhs.c.len() == 2);
        #[cfg(debug_assertions)]
        {
            // We save 2 ntts if polynomial passed to `fast_expand_crt_basis_p_over_q` is in coefficient form. Hence
            // it is cheaper to pass ciphertexts in coefficient form. But if you are stuck with two ciphertext one in coefficient
            // and another in evaluation, pass the one in evaluation form as `self`. This way ciphertext in coefficient
            // form is passed to `fast_expand_crt_basis_p_over_q`  giving us same saving as if both ciphertexts were
            // in coefficient form.
            if (lhs.c[0].representation != rhs.c[0].representation)
                && (rhs.c[0].representation != Representation::Coefficient)
            {
                panic!("Different representation in multiply1 only allows when self is in `Evalaution`")
            }
        }
        assert!(lhs.level == rhs.level);
        assert!(lhs.poly_type == rhs.poly_type);
        assert!(lhs.poly_type == PolyType::Q);

        let level = lhs.level;
        let q_ctx = self.params.poly_ctx(&PolyType::Q, level);
        let p_ctx = self.params.poly_ctx(&PolyType::P, level);
        let pq_ctx = self.params.poly_ctx(&PolyType::PQ, level);

        // let mut now = std::time::Instant::now();
        let mut c00 = q_ctx.expand_crt_basis(
            &lhs.c[0],
            &pq_ctx,
            &p_ctx,
            &self.params.ql_hat_modpl[level],
            &self.params.ql_hat_inv_modql[level],
            &self.params.ql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.alphal_modpl[level],
        );
        let mut c01 = q_ctx.expand_crt_basis(
            &lhs.c[1],
            &pq_ctx,
            &p_ctx,
            &self.params.ql_hat_modpl[level],
            &self.params.ql_hat_inv_modql[level],
            &self.params.ql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.alphal_modpl[level],
        );
        // println!("Extend1 {:?}", now.elapsed());
        if c00.representation != Representation::Evaluation {
            pq_ctx.change_representation(&mut c00, Representation::Evaluation);
            pq_ctx.change_representation(&mut c01, Representation::Evaluation);
        }
        // println!("Extend1 (In Evaluation) {:?}", now.elapsed());

        // now = std::time::Instant::now();
        let mut c10 = q_ctx.fast_expand_crt_basis_p_over_q(
            &rhs.c[0],
            &p_ctx,
            &pq_ctx,
            &self.params.neg_pql_hat_inv_modql[level],
            &self.params.neg_pql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.ql_inv_modpl[level],
            &self.params.pl_hat_modql[level],
            &self.params.pl_hat_inv_modpl[level],
            &self.params.pl_hat_inv_modpl_shoup[level],
            &self.params.pl_inv[level],
            &self.params.alphal_modql[level],
        );
        let mut c11 = q_ctx.fast_expand_crt_basis_p_over_q(
            &rhs.c[1],
            &p_ctx,
            &pq_ctx,
            &self.params.neg_pql_hat_inv_modql[level],
            &self.params.neg_pql_hat_inv_modql_shoup[level],
            &self.params.ql_inv[level],
            &self.params.ql_inv_modpl[level],
            &self.params.pl_hat_modql[level],
            &self.params.pl_hat_inv_modpl[level],
            &self.params.pl_hat_inv_modpl_shoup[level],
            &self.params.pl_inv[level],
            &self.params.alphal_modql[level],
        );
        // println!("Extend2 {:?}", now.elapsed());
        pq_ctx.change_representation(&mut c10, Representation::Evaluation);
        pq_ctx.change_representation(&mut c11, Representation::Evaluation);
        // println!("Extend2 (In Evaluation) {:?}", now.elapsed());

        // now = std::time::Instant::now();
        // tensor
        // c00 * c10
        let c_r0 = pq_ctx.mul(&c00, &c10);

        // c00 * c11 + c01 * c10
        pq_ctx.mul_assign(&mut c00, &c11);
        pq_ctx.mul_assign(&mut c10, &c01);
        pq_ctx.add_assign(&mut c00, &c10);

        // c01 * c11
        pq_ctx.mul_assign(&mut c01, &c11);
        // println!("Tensor {:?}", now.elapsed());

        Ciphertext {
            c: vec![c_r0, c00, c01],
            poly_type: PolyType::PQ,
            level: level,
            seed: None,
        }
    }

    pub fn scale_and_round(&self, c0: &mut Ciphertext) -> Ciphertext {
        // debug_assert!(c0.c[0].representation == Representation::E)
        assert!(c0.poly_type == PolyType::PQ);
        let level = c0.level;
        let pq_ctx = self.params.poly_ctx(&PolyType::PQ, level);
        let q_ctx = self.params.poly_ctx(&PolyType::Q, level);
        let p_ctx = self.params.poly_ctx(&PolyType::P, level);

        let c =
            c0.c.iter_mut()
                .map(|pq_poly| {
                    pq_ctx.change_representation(pq_poly, Representation::Coefficient);
                    pq_ctx.scale_and_round(
                        pq_poly,
                        &q_ctx,
                        &p_ctx,
                        &q_ctx,
                        &self.params.tql_pl_hat_inv_modpl_divpl_modql[level],
                        &self.params.tql_pl_hat_inv_modpl_divpl_frachi[level],
                        &self.params.tql_pl_hat_inv_modpl_divpl_fraclo[level],
                    )
                })
                .collect_vec();

        Ciphertext {
            c,
            poly_type: PolyType::Q,
            level,
            seed: None,
        }
    }

    pub fn relinearize(&self, c0: &Ciphertext, ek: &EvaluationKey) -> Ciphertext {
        ek.rlks
            .get(&c0.level)
            .expect("Rlk missing!")
            .relinearize(&c0, &self.params)
    }

    pub fn rotate(&self, c0: &Ciphertext, rotate_by: isize, ek: &EvaluationKey) -> Ciphertext {
        ek.rtgs
            .get(&(rotate_by, c0.level))
            .expect(&format!("Rtg missing! :{rotate_by} {}", c0.level))
            .rotate(&c0, &self.params)
    }

    pub fn add_assign(&self, c0: &mut Ciphertext, c1: &Ciphertext) {
        // TODO: perform checks
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);

        izip!(c0.c.iter_mut(), c1.c.iter()).for_each(|(p0, p1)| {
            ctx.add_assign(p0, p1);
        });
        c0.seed = None;
    }

    pub fn add(&self, c0: &Ciphertext, c1: &Ciphertext) -> Ciphertext {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);

        let c = izip!(c0.c.iter(), c1.c.iter())
            .map(|(p0, p1)| ctx.add(p0, p1))
            .collect_vec();

        Ciphertext {
            c,
            poly_type: c0.poly_type.clone(),
            level: c0.level,
            seed: None,
        }
    }

    pub fn sub_assign(&self, c0: &mut Ciphertext, c1: &Ciphertext) {
        // TODO: perform checks
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);

        izip!(c0.c.iter_mut(), c1.c.iter()).for_each(|(p0, p1)| {
            ctx.sub_assign(p0, p1);
        });
        c0.seed = None;
    }

    pub fn sub(&self, c0: &Ciphertext, c1: &Ciphertext) -> Ciphertext {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);

        let c = izip!(c0.c.iter(), c1.c.iter())
            .map(|(p0, p1)| ctx.sub(p0, p1))
            .collect_vec();

        Ciphertext {
            c,
            poly_type: c0.poly_type.clone(),
            level: c0.level,
            seed: None,
        }
    }

    /// c0 += c1 * poly
    pub fn fma_poly(&self, c0: &mut Ciphertext, c1: &Ciphertext, poly: &Poly) {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);
        izip!(c0.c.iter_mut(), c1.c.iter()).for_each(|(p0, p1)| {
            ctx.add_assign(p0, &ctx.mul(p1, poly));
        });

        c0.seed = None;
    }

    pub fn mul_poly_assign(&self, c0: &mut Ciphertext, poly: &Poly) {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);
        c0.c.iter_mut().for_each(|p0| ctx.mul_assign(p0, poly));

        c0.seed = None;
    }

    pub fn mul_poly(&self, c0: &Ciphertext, poly: &Poly) -> Ciphertext {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);
        let c = c0.c.iter().map(|p0| ctx.mul(p0, poly)).collect_vec();
        Ciphertext {
            c,
            poly_type: c0.poly_type.clone(),
            level: c0.level,
            seed: None,
        }
    }

    /// c0 = poly - c0
    pub fn sub_ciphertext_from_poly_inplace(&self, c0: &mut Ciphertext, poly: &Poly) {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);
        ctx.sub_reversed_inplace(&mut c0.c[0], &poly);
        ctx.neg_assign(&mut c0.c[1]);

        c0.seed = None;
    }

    pub fn mod_down_next(&self, c0: &mut Ciphertext) {
        assert!(c0.poly_type == PolyType::Q);
        let level = c0.level;
        let ctx = self.params.poly_ctx(&c0.poly_type, level);
        c0.c.iter_mut().for_each(|p| {
            ctx.mod_down_next(p, &self.params.lastq_inv_modql[level]);
        });
        c0.level = level + 1;

        c0.seed = None;
    }

    pub fn mod_down_level(&self, c0: &mut Ciphertext, level: usize) {
        let start_level = c0.level;
        for _ in start_level..level {
            self.mod_down_next(c0);
        }
    }

    pub fn plaintext_encode(&self, m: &[u64], encoding: Encoding) -> Plaintext {
        Plaintext::encode(m, &self.params, encoding)
    }

    pub fn encrypt<R: RngCore + CryptoRng>(
        &self,
        sk: &SecretKey,
        pt: &Plaintext,
        rng: &mut R,
    ) -> Ciphertext {
        sk.encrypt(&self.params, pt, rng)
    }

    pub fn decrypt(&self, sk: &SecretKey, ct: &Ciphertext) -> Plaintext {
        sk.decrypt(ct, &self.params)
    }

    pub fn plaintext_decode(&self, pt: &Plaintext, encoding: Encoding) -> Vec<u64> {
        pt.decode(encoding, &self.params)
    }

    pub fn measure_noise(&self, sk: &SecretKey, ct: &Ciphertext) -> u64 {
        sk.measure_noise(ct, &self.params)
    }

    pub unsafe fn add_noise(&self, c0: &mut Ciphertext, bit_size: usize) {
        let ctx = self.params.poly_ctx(&c0.poly_type, c0.level);

        // sample biguint
        let mut rng = thread_rng();
        let biguints = (0..ctx.degree)
            .into_iter()
            .map(|_| rng.gen_biguint(bit_size as u64))
            .collect_vec();

        let mut noise_poly = ctx.try_convert_from_biguint(&biguints, Representation::Coefficient);
        ctx.change_representation(&mut noise_poly, c0.c_ref()[0].representation.clone());

        c0.c_ref_mut().iter_mut().for_each(|p| {
            ctx.add_assign(p, &noise_poly);
        });
        c0.seed = None;
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{relinearization_key::RelinearizationKey, utils::rot_to_galois_element};

    use super::*;

    #[test]
    fn test_encryption_decryption() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(3, 1 << 8);
        let m = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);

        let sk = SecretKey::random(params.degree, params.hw, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt = evaluator.plaintext_encode(&m, Encoding::default());
        let ct = evaluator.encrypt(&sk, &pt, &mut rng);

        println!("Noise: {}", evaluator.measure_noise(&sk, &ct));

        let rm = evaluator.plaintext_decode(&evaluator.decrypt(&sk, &ct), Encoding::default());
        assert_eq!(rm, m);
    }

    #[test]
    fn test_mul_relinearize() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(5, 1 << 3);

        // gen keys
        let sk = SecretKey::random(params.degree, params.hw, &mut rng);
        let ek = EvaluationKey::new(&params, &sk, &[0], &[], &[], &mut rng);

        let mut m0 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);
        let m1 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
        let pt1 = evaluator.plaintext_encode(&m1, Encoding::default());
        let ct0 = evaluator.encrypt(&sk, &pt0, &mut rng);
        let ct1 = evaluator.encrypt(&sk, &pt1, &mut rng);

        let ct01 = evaluator.mul(&ct0, &ct1);
        println!("Noise: {}", evaluator.measure_noise(&sk, &ct01,));

        // m0 = m0 * m1
        evaluator
            .params
            .plaintext_modulus_op
            .mul_mod_fast_vec(&mut m0, &m1);

        // decrypt ct01
        let res_m = evaluator.plaintext_decode(&evaluator.decrypt(&sk, &ct01), Encoding::default());
        assert!(ct01.c.len() == 3);
        assert_eq!(&res_m, &m0);

        // relinearize
        let ct01_relin = evaluator.relinearize(&ct01, &ek);
        println!(
            "Noise after Relinearizartion: {}",
            evaluator.measure_noise(&sk, &ct01_relin,)
        );
        let res_m_relin =
            evaluator.plaintext_decode(&evaluator.decrypt(&sk, &ct01_relin), Encoding::default());

        assert!(ct01_relin.c.len() == 2);
        assert_eq!(&res_m_relin, &m0);
    }

    #[test]
    fn test_rotations() {
        let mut rng = thread_rng();
        // let params = BfvParameters::default(15, 1 << 15);
        let mut params = BfvParameters::new(&[50; 15], 65537, 1 << 15);
        params.enable_hybrid_key_switching(&[50, 50, 50]);

        // gen keys
        let sk = SecretKey::random(params.degree, params.hw, &mut rng);
        let ek = EvaluationKey::new(&params, &sk, &[], &[0], &[1], &mut rng);

        let m0 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
        let ct0 = evaluator.encrypt(&sk, &pt0, &mut rng);

        println!("Noise original: {}", evaluator.measure_noise(&sk, &ct0,));
        // let ct_rotated = evaluator.rotate(&ct0, 1, &ek);

        let mut c = ct0.clone();
        for i in 0..250 {
            let tmp = evaluator.rotate(&c, 1, &ek);
            c = tmp;
            println!("Noise {i}: {}", evaluator.measure_noise(&sk, &c,));
        }

        // println!("Noise: {}", evaluator.measure_noise(&sk, &c,));

        // decrypt ct01
        // let res_m =
        //     evaluator.plaintext_decode(&evaluator.decrypt(&sk, &ct_rotated), Encoding::default());
        // dbg!(&res_m, &m0);
    }

    #[test]
    fn test_mul_lazy_add_and_relinearize() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 15);

        // gen keys
        let sk = SecretKey::random(params.degree, params.hw, &mut rng);

        let mut m0 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());

        let set0 = (0..256)
            .into_iter()
            .map(|_| evaluator.encrypt(&sk, &pt0, &mut rng))
            .collect_vec();
        let set1 = (0..256)
            .into_iter()
            .map(|_| evaluator.encrypt(&sk, &pt0, &mut rng))
            .collect_vec();

        // lazy method
        let mut now = std::time::Instant::now();
        let mut c_res_lazy = evaluator.mul_lazy(&set0[0], &set1[0]);
        izip!(set0.iter(), set1.iter())
            .skip(1)
            .for_each(|(s0, s1)| {
                evaluator.add_assign(&mut c_res_lazy, &evaluator.mul_lazy(s0, s1));
            });
        let c_res_lazy = evaluator.scale_and_round(&mut c_res_lazy);
        let lazy_time = now.elapsed();

        // normal method
        now = std::time::Instant::now();
        let mut c_res = evaluator.mul(&set0[0], &set1[0]);
        izip!(set0.iter(), set1.iter())
            .skip(1)
            .for_each(|(s0, s1)| {
                evaluator.add_assign(&mut c_res, &evaluator.mul(s0, s1));
            });
        let normal_time = now.elapsed();

        println!(
            "Noise: Lazy={}  Normal:{}",
            evaluator.measure_noise(&sk, &c_res_lazy),
            evaluator.measure_noise(&sk, &c_res)
        );

        println!("Time: Lazy={:?}  Normal:{:?}", lazy_time, normal_time);
    }

    #[test]
    fn add_noise_works() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 3);

        let sk = SecretKey::random(params.degree, params.hw, &mut rng);
        let mut m0 = params
            .plaintext_modulus_op
            .random_vec(params.degree, &mut rng);

        let evaluator = Evaluator::new(params);
        let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());
        let mut ct = evaluator.encrypt(&sk, &pt0, &mut rng);

        println!("Noise before: {}", evaluator.measure_noise(&sk, &ct));
        unsafe {
            evaluator.add_noise(&mut ct, 100);
        }
        println!("Noise after: {}", evaluator.measure_noise(&sk, &ct));
    }

    // #[test]
    // fn trial() {
    //     let mut rng = thread_rng();
    //     let moduli = vec![50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 60];
    //     let mut params = BfvParameters::new(&moduli, 65537, 1 << 15);
    //     params.enable_hybrid_key_switching(&[50, 50, 60]);

    //     // gen keys
    //     let sk = SecretKey::random(params.degree, &mut rng);

    //     let mut m0 = params
    //         .plaintext_modulus_op
    //         .random_vec(params.degree, &mut rng);

    //     let evaluator = Evaluator::new(params);
    //     let pt0 = evaluator.plaintext_encode(&m0, Encoding::default());

    //     let ek = EvaluationKey::new(
    //         evaluator.params(),
    //         &sk,
    //         &(0..12).into_iter().collect_vec(),
    //         &[],
    //         &[],
    //         &mut rng,
    //     );

    //     let set0 = (0..20)
    //         .into_iter()
    //         .map(|_| evaluator.encrypt(&sk, &pt0, &mut rng))
    //         .collect_vec();
    //     let mut set1 = (0..20)
    //         .into_iter()
    //         .map(|_| evaluator.encrypt(&sk, &pt0, &mut rng))
    //         .collect_vec();

    //     let mut s1 = evaluator.encrypt(&sk, &pt0, &mut rng);
    //     let mut s2 = evaluator.encrypt(&sk, &pt0, &mut rng);

    //     for i in 0..20 {
    //         s1 = evaluator.relinearize(&evaluator.mul(&s1, &set0[i]), &ek);

    //         evaluator.mod_down_level(&mut set1[i], s2.level());
    //         s2 = evaluator.relinearize(&evaluator.mul(&s2, &set1[i]), &ek);

    //         if (i + 1) % 2 == 0 {
    //             evaluator.mod_down_next(&mut s2);
    //         }
    //     }

    //     dbg!(evaluator.measure_noise(&sk, &s1));
    //     dbg!(evaluator.measure_noise(&sk, &s2));
    //     dbg!(s2.level());
    //     evaluator.mod_down_level(&mut s1, s2.level());

    //     dbg!("Mod down");
    //     dbg!(evaluator.measure_noise(&sk, &s1));
    //     dbg!(evaluator.measure_noise(&sk, &s2));

    //     evaluator.ciphertext_change_representation(&mut s1, Representation::Evaluation);
    //     evaluator.ciphertext_change_representation(&mut s2, Representation::Evaluation);

    //     let sp1 = evaluator.mul_poly(&s1, pt0.poly_ntt_ref());
    //     let sp2 = evaluator.mul_poly(&s2, pt0.poly_ntt_ref());

    //     dbg!(evaluator.measure_noise(&sk, &sp1));
    //     dbg!(evaluator.measure_noise(&sk, &sp2));
    // }
}
