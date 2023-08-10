use crate::modulus::Modulus;
use crate::nb_theory::generate_primes_vec;
use crate::{mod_inverse_biguint, mod_inverse_biguint_u64};
use crate::{poly::poly_context::PolyContext, Poly, Representation};
use itertools::Itertools;
use ndarray::Array2;
use num_bigint::BigUint;
use num_traits::{One, Pow, ToPrimitive};
use std::vec;
use traits::Ntt;

#[derive(PartialEq, Debug, Clone)]
pub enum PolyType {
    Q,
    P,
    PQ,
    SpecialP,
    QP,
}

#[derive(PartialEq, Clone, Debug)]
pub struct BfvParameters<T: Ntt> {
    pub ciphertext_moduli: Vec<u64>,
    pub extension_moduli: Vec<u64>,
    pub ciphertext_moduli_ops: Vec<Modulus>,
    pub extension_moduli_ops: Vec<Modulus>,
    pub ciphertext_ntt_ops: Vec<T>,
    pub extension_ntt_ops: Vec<T>,
    pub ciphertext_moduli_sizes: Vec<usize>,
    pub max_level: usize,

    pub variance: usize,
    pub hw: usize,

    pub plaintext_modulus: u64,
    pub plaintext_modulus_op: Modulus,
    pub plaintext_ntt_op: T,
    pub degree: usize,

    // Convert Utils
    // q: ciphertext_modulus
    // p: extension_modulus
    pub q_size: usize,
    pub p_size: usize,
    pub ql_hat: Vec<Vec<BigUint>>,
    pub ql_hat_inv: Vec<Vec<u64>>,
    pub ql: Vec<BigUint>,
    // pub pl_hat: Vec<Vec<u64>>,
    // pub pl_hat_inv: Vec<Vec<u64>>,
    pub pl: Vec<BigUint>,

    // Encryption
    pub ql_modt: Vec<u64>,
    pub neg_t_inv_modql: Vec<Poly>,
    pub matrix_reps_index_map: Vec<usize>,

    // Decryption
    pub t_ql_hat_inv_modql_divql_modt: Vec<Vec<u64>>,
    pub t_bql_hat_inv_modql_divql_modt: Vec<Vec<u64>>,
    pub t_ql_hat_inv_modql_divql_frac: Vec<Vec<f64>>,
    pub t_bql_hat_inv_modql_divql_frac: Vec<Vec<f64>>,
    pub max_bit_size_by2: usize,

    // Fast expand CRT basis Q to P to PQ
    // Fast conversion P over Q
    pub neg_pql_hat_inv_modql: Vec<Vec<u64>>,
    pub neg_pql_hat_inv_modql_shoup: Vec<Vec<u64>>,
    pub ql_inv_modpl: Vec<Array2<u64>>,
    //  Switch CRT basis P to Q //
    pub pl_hat_modql: Vec<Array2<u64>>,
    pub pl_hat_inv_modpl: Vec<Vec<u64>>,
    pub pl_hat_inv_modpl_shoup: Vec<Vec<u64>>,
    pub pl_inv: Vec<Vec<f64>>,
    pub alphal_modql: Vec<Array2<u64>>,

    // Scale and Round //
    pub tql_pl_hat_inv_modpl_divpl_modql: Vec<Array2<u64>>,
    pub tql_pl_hat_inv_modpl_divpl_frachi: Vec<Vec<u64>>,
    pub tql_pl_hat_inv_modpl_divpl_fraclo: Vec<Vec<u64>>,

    // Switch CRT basis Q to P //
    pub ql_hat_modpl: Vec<Array2<u64>>,
    pub ql_hat_inv_modql: Vec<Vec<u64>>,
    pub ql_hat_inv_modql_shoup: Vec<Vec<u64>>,
    pub ql_inv: Vec<Vec<f64>>,
    pub alphal_modpl: Vec<Array2<u64>>,

    // Hybrid key switching
    pub special_moduli: Option<Vec<u64>>,
    pub special_moduli_ops: Option<Vec<Modulus>>,
    pub special_moduli_ntt_ops: Option<Vec<T>>,
    pub dnum: Option<usize>,
    pub alpha: Option<usize>, // fixed to 3

    // Hybrid key switching key parameters
    pub hybrid_ksk_parameters: Option<Vec<HybridKeySwitchingParameters>>,

    // Mod Down //
    pub lastq_inv_modql: Vec<Vec<u64>>,
}

impl<T> BfvParameters<T>
where
    T: Ntt,
{
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
        degree: usize,
    ) -> BfvParameters<T> {
        assert!(degree.is_power_of_two() && degree >= 16);

        // generate Q moduli chain
        let ciphertext_moduli = generate_primes_vec(ciphertext_moduli_sizes, degree, &[]);

        // generate P moduli chain
        let extension_moduli =
            generate_primes_vec(ciphertext_moduli_sizes, degree, &ciphertext_moduli);

        // moduli ops
        let ciphertext_moduli_ops = ciphertext_moduli
            .iter()
            .map(|qi| Modulus::new(*qi))
            .collect_vec();
        let extension_moduli_ops = extension_moduli
            .iter()
            .map(|pi| Modulus::new(*pi))
            .collect_vec();

        // ntt ops
        let ciphertext_ntt_ops = ciphertext_moduli
            .iter()
            .map(|qi| T::new(degree, *qi))
            .collect_vec();
        let extension_ntt_ops = extension_moduli
            .iter()
            .map(|pi| T::new(degree, *pi))
            .collect_vec();

        let mut q_biguint = BigUint::one();
        let mut p_biguint = BigUint::one();
        ciphertext_moduli.iter().for_each(|qi| {
            q_biguint *= *qi;
        });
        extension_moduli.iter().for_each(|pi| {
            p_biguint *= *pi;
        });

        // What's happening here? Well we intend to store parameters for a level at levelth index, because
        // it makes it easier to retrieve params at level by simply calling params_xyz[level].
        // Since params at level 0 correspond to moduli chain q0,...,qmax and level 1 correspond to
        // q0,...,qmax-1 and so on and so forth, we need to process moduli chain starting at max length, which
        // implies processing ciphertext_moduli in reverese order below.
        let mut q_biguint_clone = q_biguint.clone();
        let mut ql = vec![];
        ciphertext_moduli.iter().rev().for_each(|qi| {
            ql.push(q_biguint_clone.clone());
            q_biguint_clone /= *qi;
        });

        let mut p_biguint_clone = p_biguint.clone();
        let mut pl = vec![];
        extension_moduli.iter().rev().for_each(|pi| {
            pl.push(p_biguint_clone.clone());
            p_biguint_clone /= *pi;
        });

        let q_size = ciphertext_moduli.len();
        let p_size = extension_moduli.len();

        let mut ql_hat = vec![];
        let mut ql_hat_inv = vec![];
        for i in 0..q_size {
            let q = &ql[i];
            let mut q_hat = vec![];
            let mut q_hat_inv = vec![];
            for j in 0..(q_size - i) {
                let qj = ciphertext_moduli[j];
                q_hat.push(q / qj);
                q_hat_inv.push(mod_inverse_biguint_u64(&(q / qj), qj).to_u64().unwrap());
            }
            ql_hat.push(q_hat);
            ql_hat_inv.push(q_hat_inv);
        }

        // ENCRYPTION //
        let mut ql_modt = vec![];
        let mut neg_t_inv_modql = vec![];
        for i in 0..q_size {
            let q = &ql[i];
            ql_modt.push((q % plaintext_modulus).to_u64().unwrap());
            let neg_t_inv_modq = mod_inverse_biguint(&(q - plaintext_modulus), &q);

            let ctx = PolyContext {
                moduli_count: q_size - i,
                moduli_ops: (&ciphertext_moduli_ops[..q_size - i], &[]),
                ntt_ops: (&ciphertext_ntt_ops[..q_size - i], &[]),
                degree,
            };

            let mut neg_t_inv_modq =
                ctx.try_convert_from_biguint(&vec![neg_t_inv_modq], Representation::Coefficient);
            ctx.change_representation(&mut neg_t_inv_modq, Representation::Evaluation);
            neg_t_inv_modql.push(neg_t_inv_modq);
        }

        // DECRYPTION //
        let b_bits = ciphertext_moduli_sizes.iter().max().unwrap() / 2;
        let mut t_ql_hat_inv_modql_divql_modt = vec![];
        let mut t_bql_hat_inv_modql_divql_modt = vec![];
        let mut t_ql_hat_inv_modql_divql_frac = vec![];
        let mut t_bql_hat_inv_modql_divql_frac = vec![];
        for i in 0..q_size {
            let q = &ql[i];

            // let a = [(Q/qi)^-1]_qi and b = 1<<b_bits, where qi is q[i]. The idea is to calculate
            // rational and fraction part of a/qi and a*([b]_qi)/qi
            // (t*a)/qi (mod t)
            let mut rationals = vec![];
            // (t*[a*b]_qi)/qi (mod t)
            let mut brationals = vec![];
            // ((t*a)%qi)/qi
            let mut fractionals = vec![];
            // ((t*[a*b]_qi)%qi)/qi
            let mut bfractionals = vec![];
            for j in 0..(q_size - i) {
                let qi = ciphertext_moduli[j];

                // [(Q/qi)^-1]_qi
                let qi_hat_inv = mod_inverse_biguint_u64(&(q / qi), qi);

                let b_modqi = (1 << b_bits) % qi;

                rationals.push(
                    (((plaintext_modulus * &qi_hat_inv) / qi) % plaintext_modulus)
                        .to_u64()
                        .unwrap(),
                );
                brationals.push(
                    (((plaintext_modulus * ((&qi_hat_inv * b_modqi) % qi)) / qi)
                        % plaintext_modulus)
                        .to_u64()
                        .unwrap(),
                );

                fractionals
                    .push(((plaintext_modulus * &qi_hat_inv) % qi).to_f64().unwrap() / qi as f64);
                bfractionals.push(
                    ((plaintext_modulus * ((&qi_hat_inv * b_modqi) % qi)) % qi)
                        .to_f64()
                        .unwrap()
                        / qi as f64,
                );
            }

            t_ql_hat_inv_modql_divql_modt.push(rationals);
            t_bql_hat_inv_modql_divql_modt.push(brationals);
            t_ql_hat_inv_modql_divql_frac.push(fractionals);
            t_bql_hat_inv_modql_divql_frac.push(bfractionals);
        }

        // Fast expand CRT basis Q to P to PQ //
        // (1) Fast Conv P Over Q //
        let mut neg_pql_hat_inv_modql = vec![];
        let mut neg_pql_hat_inv_modql_shoup = vec![];
        let mut ql_inv_modpl = vec![];
        for i in 0..q_size {
            let q = &ql[i];
            let p = &pl[i];

            // [-p*((Q/qi)^-1)]_qi for each qi in q
            let mut neg_pqi_hat_inv_modqi = vec![];
            let mut neg_pqi_hat_inv_modqi_shoup = vec![];

            for j in 0..(q_size - i) {
                let qi = ciphertext_moduli[j];
                let modqi = &ciphertext_moduli_ops[j];
                let tmp = (qi
                    - ((p * mod_inverse_biguint_u64(&(q / qi), qi.try_into().unwrap())) % qi))
                    .to_u64()
                    .unwrap();
                neg_pqi_hat_inv_modqi.push(tmp);
                neg_pqi_hat_inv_modqi_shoup.push(modqi.compute_shoup(tmp));
            }
            neg_pql_hat_inv_modql.push(neg_pqi_hat_inv_modqi);
            neg_pql_hat_inv_modql_shoup.push(neg_pqi_hat_inv_modqi_shoup);

            // [qi^-1]_pj
            let mut q_inv_modp = vec![];
            for k in 0..(p_size - i) {
                let modpk = &extension_moduli_ops[k];
                for j in 0..(q_size - i) {
                    let qi = ciphertext_moduli[j];
                    q_inv_modp.push(modpk.inv(qi % modpk.modulus()))
                }
            }
            ql_inv_modpl
                .push(Array2::from_shape_vec((p_size - i, q_size - i), q_inv_modp).unwrap());
        }

        // (2) Switch CRT basis P to Q //
        let mut pl_hat_modql = vec![];
        let mut pl_hat_inv_modpl = vec![];
        let mut pl_hat_inv_modpl_shoup = vec![];
        let mut pl_inv = vec![];
        let mut alphal_modql = vec![];
        for i in 0..p_size {
            let p = &pl[i];

            // (p/pj) % qi
            let mut p_hat_modq = vec![];
            for k in 0..(q_size - i) {
                let qi = ciphertext_moduli[k];
                for j in 0..(p_size - i) {
                    let pj = extension_moduli[j];
                    p_hat_modq.push(((p / pj) % qi).to_u64().unwrap());
                }
            }
            pl_hat_modql
                .push(Array2::from_shape_vec((q_size - i, p_size - i), p_hat_modq).unwrap());

            // [(p/pj)^-1]_pj
            let mut p_hat_inv_modp = vec![];
            let mut p_hat_inv_modp_shoup = vec![];
            let mut p_inv = vec![];
            for j in 0..(p_size - i) {
                let pj = extension_moduli[j];
                let modpj = &extension_moduli_ops[j];
                let tmp = mod_inverse_biguint_u64(&(p / pj), pj).to_u64().unwrap();
                p_hat_inv_modp.push(tmp);
                p_hat_inv_modp_shoup.push(modpj.compute_shoup(tmp));
                p_inv.push(1.0 / (pj as f64));
            }
            pl_hat_inv_modpl.push(p_hat_inv_modp);
            pl_hat_inv_modpl_shoup.push(p_hat_inv_modp_shoup);
            pl_inv.push(p_inv);

            // [\alpha*p]_qk
            // \alpha can be in range 0..(p_size-i)+1. So
            // we pre-compute its representation in modqk.
            let mut alpha_modq = vec![];
            for k in 0..(q_size - i) {
                let qk = ciphertext_moduli[k];
                for i in 0..(p_size - i + 1) {
                    let v = p * i;
                    alpha_modq.push((v % qk).to_u64().unwrap())
                }
            }
            alphal_modql
                .push(Array2::from_shape_vec((q_size - i, p_size - i + 1), alpha_modq).unwrap());
        }

        // Scale and Round //
        let mut tql_pl_hat_inv_modpl_divpl_modql = vec![];
        let mut tql_pl_hat_inv_modpl_divpl_frachi = vec![];
        let mut tql_pl_hat_inv_modpl_divpl_fraclo = vec![];
        for i in 0..p_size {
            let pq = &pl[i] * &ql[i];
            let q = &ql[i];

            // ([(pq/pj)^-1]_pj * tq)/pj (mod qi)
            let mut tq_p_hat_inv_modp_divp_modq = vec![];
            // ([(pq/pj)^-1]_pj * tq) % pj / pj
            let mut tq_p_hat_inv_modp_divp_frachi = vec![];
            let mut tq_p_hat_inv_modp_divp_fraclo = vec![];

            // ([(pq/pj)^-1]_pj * tq)
            let mut tmp = vec![];
            for j in 0..(p_size - i) {
                let pj = extension_moduli[j];
                tmp.push(plaintext_modulus * q * mod_inverse_biguint_u64(&(&pq / pj), pj));
            }

            for k in 0..(q_size - i) {
                let qk = ciphertext_moduli[k];
                let modqk = &ciphertext_moduli_ops[k];

                for j in 0..(p_size - i) {
                    let pj = extension_moduli[j];

                    // rational
                    tq_p_hat_inv_modp_divp_modq.push(((&tmp[j] / pj) % qk).to_u64().unwrap());

                    // fractional
                    let mut rem = &tmp[j] % pj;
                    rem <<= 127;
                    rem /= pj;
                    let rem = rem.to_u128().unwrap();
                    tq_p_hat_inv_modp_divp_frachi.push((rem >> 64) as u64);
                    tq_p_hat_inv_modp_divp_fraclo.push(rem as u64)
                }

                // Handle rational value for qi. Fractional is 0 since `qi|tQ`
                let v = plaintext_modulus * q * mod_inverse_biguint_u64(&(&pq / qk), qk);
                tq_p_hat_inv_modp_divp_modq.push(((v / qk) % qk).to_u64().unwrap());
            }

            tql_pl_hat_inv_modpl_divpl_modql.push(
                Array2::from_shape_vec((q_size - i, p_size - i + 1), tq_p_hat_inv_modp_divp_modq)
                    .unwrap(),
            );
            tql_pl_hat_inv_modpl_divpl_frachi.push(tq_p_hat_inv_modp_divp_frachi);
            tql_pl_hat_inv_modpl_divpl_fraclo.push(tq_p_hat_inv_modp_divp_fraclo);
        }

        // Switch CRT basis Q to P //
        let mut ql_hat_modpl = vec![];
        let mut ql_hat_inv_modql = vec![];
        let mut ql_hat_inv_modql_shoup = vec![];
        let mut ql_inv = vec![];
        let mut alphal_modpl = vec![];
        for i in 0..q_size {
            let q = &ql[i];

            // [(q/qj)^-1]_qj
            let mut q_hat_inv_modq = vec![];
            let mut q_hat_inv_modq_shoup = vec![];
            let mut q_inv = vec![];
            for j in 0..(q_size - i) {
                let qj = ciphertext_moduli[j];
                let modqj = &ciphertext_moduli_ops[j];
                let tmp = mod_inverse_biguint_u64(&(q / qj), qj).to_u64().unwrap();
                q_hat_inv_modq.push(tmp);
                q_hat_inv_modq_shoup.push(modqj.compute_shoup(tmp));
                q_inv.push(1.0 / qj as f64);
            }
            ql_hat_inv_modql.push(q_hat_inv_modq);
            ql_hat_inv_modql_shoup.push(q_hat_inv_modq_shoup);
            ql_inv.push(q_inv);

            // q/qi (mod pj)
            let mut q_hat_modp = vec![];
            for k in 0..(p_size - i) {
                let pk = extension_moduli[k];
                for j in 0..(q_size - i) {
                    let qj = ciphertext_moduli[j];
                    q_hat_modp.push(((q / qj) % pk).to_u64().unwrap());
                }
            }
            ql_hat_modpl
                .push(Array2::from_shape_vec((p_size - i, q_size - i), q_hat_modp).unwrap());

            // [\alpha * q]_p
            let mut alpha_modp = vec![];
            for k in 0..(p_size - i) {
                let pk = extension_moduli[k];
                for j in 0..(q_size - i + 1) {
                    alpha_modp.push(((q * j) % pk).to_u64().unwrap());
                }
            }
            alphal_modpl
                .push(Array2::from_shape_vec((p_size - i, q_size - i + 1), alpha_modp).unwrap());
        }

        // Mod down next //
        let mut lastq_inv_modql = vec![];
        for i in 0..q_size {
            let last_qi = ciphertext_moduli[q_size - i - 1];

            let mut lastq_inv_modq = vec![];
            for j in 0..(q_size - i - 1) {
                let modqi = &ciphertext_moduli_ops[j];
                lastq_inv_modq.push(modqi.inv(last_qi % modqi.modulus()));
            }
            lastq_inv_modql.push(lastq_inv_modq);
        }

        // To generate mapping for matrix representation index, we use: https://github.com/microsoft/SEAL/blob/82b07db635132e297282649e2ab5908999089ad2/native/src/seal/batchencoder.cpp
        let row = degree >> 1;
        let m = degree << 1;
        let gen = 3;
        let mut pos = 1;
        let mut matrix_reps_index_map = vec![0usize; degree];
        for i in 0..row {
            let index1 = (pos - 1) >> 1;
            let index2 = (m - pos - 1) >> 1;
            matrix_reps_index_map[i] = index1.reverse_bits() >> (degree.leading_zeros() + 1);
            matrix_reps_index_map[i | row] = index2.reverse_bits() >> (degree.leading_zeros() + 1);
            pos *= gen;
            pos &= m - 1;
        }

        let plaintext_modulus_op = Modulus::new(plaintext_modulus);
        let plaintext_ntt_op = T::new(degree, plaintext_modulus);

        // Default to Hamming weight set to N/2.
        let hw = degree / 2;

        BfvParameters {
            ciphertext_moduli,
            extension_moduli,
            ciphertext_moduli_ops,
            extension_moduli_ops,
            ciphertext_ntt_ops,
            extension_ntt_ops,
            ciphertext_moduli_sizes: ciphertext_moduli_sizes.to_vec(),
            max_level: q_size - 1,
            q_size,
            p_size,

            variance: 10,
            hw,

            plaintext_modulus,
            plaintext_modulus_op,
            plaintext_ntt_op,
            degree,

            // Convert Utils
            ql_hat,
            ql_hat_inv,
            ql,
            pl,

            // ENCRYPTION //
            ql_modt,
            neg_t_inv_modql,
            matrix_reps_index_map,

            // DECRYPTION //
            t_ql_hat_inv_modql_divql_modt,
            t_bql_hat_inv_modql_divql_modt,
            t_ql_hat_inv_modql_divql_frac,
            t_bql_hat_inv_modql_divql_frac,
            max_bit_size_by2: b_bits,

            // Fast expand CRT basis Q to P to PQ
            neg_pql_hat_inv_modql,
            neg_pql_hat_inv_modql_shoup,
            ql_inv_modpl,
            pl_hat_modql,
            pl_hat_inv_modpl,
            pl_hat_inv_modpl_shoup,
            pl_inv,
            alphal_modql,

            // Scale and Round //
            tql_pl_hat_inv_modpl_divpl_modql,
            tql_pl_hat_inv_modpl_divpl_frachi,
            tql_pl_hat_inv_modpl_divpl_fraclo,

            // Switch CRT basis Q to P //
            ql_hat_modpl,
            ql_hat_inv_modql,
            ql_hat_inv_modql_shoup,
            ql_inv,
            alphal_modpl,

            // Hybrid key switching //
            special_moduli: None,
            alpha: None,
            dnum: None,
            special_moduli_ntt_ops: None,
            special_moduli_ops: None,

            hybrid_ksk_parameters: None,

            // Mod down next //
            lastq_inv_modql,
        }
    }

    pub fn change_hamming_weight(&mut self, hw: usize) {
        self.hw = hw;
    }

    pub fn enable_hybrid_key_switching(&mut self, specialp_bits: &[usize; 3]) {
        const ALPHA: usize = 3;
        let dnum = (self.ciphertext_moduli.len() as f64 / ALPHA as f64).ceil() as usize;
        let special_moduli =
            generate_primes_vec(specialp_bits, self.degree, &self.ciphertext_moduli);
        let special_moduli_ops = special_moduli
            .iter()
            .map(|pj| Modulus::new(*pj))
            .collect_vec();
        let special_moduli_ntt_ops = special_moduli
            .iter()
            .map(|pj| T::new(self.degree, *pj))
            .collect_vec();

        self.special_moduli = Some(special_moduli);
        self.alpha = Some(ALPHA);
        self.dnum = Some(dnum);
        self.special_moduli_ntt_ops = Some(special_moduli_ntt_ops);
        self.special_moduli_ops = Some(special_moduli_ops);

        // generate hybrid key switching parameters for each level except the last one
        let params = (0..self.max_level)
            .into_iter()
            .map(|level| {
                let ksk_ctx = self.poly_ctx(&PolyType::Q, level);
                let specialp_ctx = self.poly_ctx(&PolyType::SpecialP, level);
                HybridKeySwitchingParameters::new(&ksk_ctx, &specialp_ctx, ALPHA)
            })
            .collect_vec();

        self.hybrid_ksk_parameters = Some(params);
    }

    pub fn poly_ctx(&self, poly_type: &PolyType, level: usize) -> PolyContext<'_, T> {
        let level_index = self.q_size - level;
        match poly_type {
            PolyType::Q => PolyContext {
                moduli_ops: (&self.ciphertext_moduli_ops[..level_index], &[]),
                ntt_ops: (&self.ciphertext_ntt_ops[..level_index], &[]),
                moduli_count: level_index,
                degree: self.degree,
            },
            PolyType::P => PolyContext {
                moduli_ops: (&self.extension_moduli_ops[..level_index], &[]),
                ntt_ops: (&self.extension_ntt_ops[..level_index], &[]),
                moduli_count: level_index,
                degree: self.degree,
            },
            PolyType::PQ => PolyContext {
                moduli_ops: (
                    &self.extension_moduli_ops[..level_index],
                    &self.ciphertext_moduli_ops[..level_index],
                ),
                ntt_ops: (
                    &self.extension_ntt_ops[..level_index],
                    &self.ciphertext_ntt_ops[..level_index],
                ),
                moduli_count: level_index * 2,
                degree: self.degree,
            },
            PolyType::SpecialP => {
                let tmp = PolyContext {
                    moduli_ops: (
                        self.special_moduli_ops
                            .as_ref()
                            .expect("SpecialP missing")
                            .as_slice(),
                        &[],
                    ),
                    ntt_ops: (
                        self.special_moduli_ntt_ops
                            .as_ref()
                            .expect("SpecialP missing")
                            .as_slice(),
                        &[],
                    ),
                    moduli_count: self.alpha.expect("SpecialP missing"),
                    degree: self.degree,
                };
                tmp
            }
            PolyType::QP => {
                let tmp = PolyContext {
                    moduli_ops: (
                        &self.ciphertext_moduli_ops[..level_index],
                        &self.special_moduli_ops.as_ref().expect("QP missing"),
                    ),
                    ntt_ops: (
                        &self.ciphertext_ntt_ops[..level_index],
                        &self.special_moduli_ntt_ops.as_ref().expect("QP missing"),
                    ),
                    moduli_count: level_index + self.alpha.expect("QP missing"),
                    degree: self.degree,
                };
                tmp
            }
        }
    }

    pub fn hybrid_key_switching_params_at_level(
        &self,
        level: usize,
    ) -> &HybridKeySwitchingParameters {
        &self
            .hybrid_ksk_parameters
            .as_ref()
            .expect("Hybrid Key Switching Parameters not initialized")[level]
    }

    pub fn default(moduli_count: usize, polynomial_degree: usize) -> BfvParameters<T> {
        let mut params = BfvParameters::new(&vec![50; moduli_count], 65537, polynomial_degree);
        params.enable_hybrid_key_switching(&[50, 50, 50]);
        params
    }
}

#[derive(PartialEq, Clone, Debug)]
pub struct HybridKeySwitchingParameters {
    pub(crate) dnum: usize,
    pub(crate) alpha: usize,

    pub(crate) g: Vec<BigUint>,

    // approx_switch_crt_basis //
    pub(crate) qj_hat_inv_modqj_parts: Vec<Vec<u64>>,
    pub(crate) qj_moduli_ops_parts: Vec<Vec<Modulus>>,
    pub(crate) qj_hat_modqpj_parts: Vec<Array2<u64>>,
    pub(crate) qpj_moduli_ops_parts: Vec<Vec<Modulus>>,

    // approx_mod_down //
    pub(crate) p_hat_inv_modp: Vec<u64>,
    pub(crate) p_hat_modq: Array2<u64>,
    pub(crate) p_inv_modq: Vec<u64>,
}

impl HybridKeySwitchingParameters {
    pub fn new<T: Ntt>(
        ksk_ctx: &PolyContext<'_, T>,
        specialp_ctx: &PolyContext<'_, T>,
        alpha: usize,
    ) -> HybridKeySwitchingParameters {
        let mut qj = vec![];
        ksk_ctx
            .iter_moduli_ops()
            .chunks(alpha)
            .into_iter()
            .for_each(|modqi_parts| {
                // Qj
                let mut qji = BigUint::one();
                modqi_parts.into_iter().for_each(|modqi| {
                    qji *= modqi.modulus();
                });
                qj.push(qji);
            });

        let mut maxbits = qj[0].bits();
        qj.iter().skip(1).for_each(|q| {
            maxbits = std::cmp::max(maxbits, q.bits());
        });

        // bits in P must be more or less equal to bits in max(Qj) to assure that noise growth is key to minimum
        let specialp_bits = specialp_ctx.big_q().bits();
        assert!(maxbits / specialp_bits <= 1);

        // P is special prime
        let p = specialp_ctx.big_q();

        // Calculating g values + Pre-comp for switching Qj to QP.
        let q = ksk_ctx.big_q();
        // g = P * Q/Qj * [(Q/Qj)^-1]_Qj
        let mut g = vec![];
        // we are forced to use Vec insted of `ndarray` because each of the
        // 2d vectors cannot be translated to a proper array in row major form.
        let mut qj_hat_inv_modqj_parts = vec![];
        let mut qj_hat_modqpj_parts = vec![];
        let mut qpj_moduli_ops_parts = vec![];
        let mut qj_moduli_ops_parts = vec![];

        let parts = (ksk_ctx.moduli_count as f64 / alpha as f64).ceil() as usize;

        ksk_ctx
            .iter_moduli_ops()
            .chunks(alpha)
            .into_iter()
            .enumerate()
            .for_each(|(chunk_index, modqj)| {
                let qj_moduli_ops = modqj.into_iter().map(|modqji| modqji.clone()).collect_vec();

                // Qj
                let mut qj = BigUint::one();
                qj_moduli_ops.iter().for_each(|modqi| qj *= modqi.modulus());

                // Q/Qj
                let qj_hat = &q / &qj;

                // [(Q/Qj)^-1]_Qj
                let qj_hat_inv_modqj = mod_inverse_biguint(&qj_hat, &qj);
                g.push(&p * qj_hat * qj_hat_inv_modqj);

                // precomp approx_switch_crt_basis
                let mut qj_hat_inv_modqj = vec![];
                qj_moduli_ops.iter().for_each(|modqji| {
                    let qji = modqji.modulus();
                    qj_hat_inv_modqj
                        .push(mod_inverse_biguint_u64(&(&qj / qji), qji).to_u64().unwrap());
                });
                qj_hat_inv_modqj_parts.push(qj_hat_inv_modqj);

                let mut qpj_moduli_ops = vec![];
                qpj_moduli_ops.extend_from_slice(&ksk_ctx.moduli_ops()[..alpha * chunk_index]);
                qpj_moduli_ops.extend_from_slice(
                    &ksk_ctx.moduli_ops()
                        [std::cmp::min(ksk_ctx.moduli_count, alpha * (chunk_index + 1))..],
                );
                qpj_moduli_ops.extend_from_slice(specialp_ctx.moduli_ops());

                let mut qj_hat_modqpj = vec![];
                qpj_moduli_ops.iter().for_each(|modqpji| {
                    qj_moduli_ops.iter().for_each(|modqji| {
                        qj_hat_modqpj.push(
                            ((&qj / modqji.modulus()) % modqpji.modulus())
                                .to_u64()
                                .unwrap(),
                        )
                    })
                });
                let qj_hat_modqpj = Array2::<u64>::from_shape_vec(
                    (qpj_moduli_ops.len(), qj_moduli_ops.len()),
                    qj_hat_modqpj,
                )
                .unwrap();

                qj_hat_modqpj_parts.push(qj_hat_modqpj);
                qpj_moduli_ops_parts.push(qpj_moduli_ops);
                qj_moduli_ops_parts.push(qj_moduli_ops);
            });

        // Precompute for P to Q (used for approx_switch_crt_basis in approx_mod_down)
        let mut p_hat_inv_modp = vec![];
        let mut p_hat_modq = vec![];
        specialp_ctx.iter_moduli_ops().for_each(|(modpi)| {
            let pi = modpi.modulus();
            let pi_hat = &p / pi;
            p_hat_inv_modp.push(mod_inverse_biguint_u64(&pi_hat, pi).to_u64().unwrap());
        });
        ksk_ctx.iter_moduli_ops().for_each(|modqj| {
            specialp_ctx.iter_moduli_ops().for_each(|(modpi)| {
                p_hat_modq.push(((&p / modpi.modulus()) % modqj.modulus()).to_u64().unwrap());
            });
        });
        let p_hat_modq = Array2::from_shape_vec((ksk_ctx.moduli_count, alpha), p_hat_modq).unwrap();
        let mut p_inv_modq = vec![];
        // Precompute for dividing values in basis Q by P (approx_mod_down)
        ksk_ctx.iter_moduli_ops().for_each(|modqi| {
            p_inv_modq.push(
                mod_inverse_biguint_u64(&p, modqi.modulus())
                    .to_u64()
                    .unwrap(),
            );
        });

        HybridKeySwitchingParameters {
            dnum: parts,
            alpha,

            g,

            // approx_switch_crt_basis //
            qj_hat_inv_modqj_parts,
            qj_moduli_ops_parts,
            qj_hat_modqpj_parts,
            qpj_moduli_ops_parts,

            // approx_mod_down //
            p_hat_inv_modp,
            p_hat_modq,
            p_inv_modq,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::BfvParameters;

    #[test]
    fn trial() {
        // dbg!(BfvParameters::v_norm(3.2, 1 << 15));
        dbg!(BfvParameters::noise_ks(10, 3.2, 1 << 15, 60));
        let params = BfvParameters::default(9, 1 << 15);
        let sp = params.poly_ctx(&crate::PolyType::SpecialP, 0);
        dbg!(sp.big_q());
    }
}
