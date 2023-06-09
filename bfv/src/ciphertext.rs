use crate::parameters::BfvParameters;
use crate::poly::{Poly, Representation};
use crate::warn;
use crate::Plaintext;
use itertools::Itertools;
use ndarray::azip;
use rayon::*;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};
use std::sync::Arc;
use traits::Ntt;

#[derive(Debug, Clone)]
pub struct Ciphertext<T: Ntt> {
    pub(crate) c: Vec<Poly<T>>,
    pub(crate) params: Arc<BfvParameters<T>>,
    pub level: usize,
}

impl<T> Ciphertext<T>
where
    T: Ntt,
{
    pub fn new(c: Vec<Poly<T>>, params: Arc<BfvParameters<T>>, level: usize) -> Ciphertext<T> {
        Ciphertext {
            c,
            params: params.clone(),
            level,
        }
    }

    pub fn zero(params: &Arc<BfvParameters<T>>, level: usize) -> Ciphertext<T> {
        Ciphertext {
            params: params.clone(),
            level,
            c: vec![],
        }
    }

    pub fn scale_and_round(&mut self) {
        let level = self.level;
        self.c.iter_mut().for_each(|p| {
            p.change_representation(Representation::Coefficient);
            *p = p.scale_and_round(
                &self.params.ciphertext_poly_contexts[level],
                &self.params.extension_poly_contexts[level],
                &self.params.ciphertext_poly_contexts[level],
                &self.params.tql_p_hat_inv_modp_divp_modql[level],
                &self.params.tql_p_hat_inv_modp_divp_frac_hi[level],
                &self.params.tql_p_hat_inv_modp_divp_frac_lo[level],
            )
        });
    }

    pub fn multiply1_lazy(&self, rhs: &Ciphertext<T>) -> Ciphertext<T> {
        debug_assert!(self.params == rhs.params);
        debug_assert!(self.c.len() == 2);
        debug_assert!(rhs.c.len() == 2);
        #[cfg(debug_assertions)]
        {
            // We save 2 ntts if polynomial passed to `fast_expand_crt_basis_p_over_q` is in coefficient form. Hence
            // it is cheaper to pass ciphertexts in coefficient form. But if you are stuck with two ciphertext one in coefficient
            // and another in evaluation, pass the one in evaluation form as `self`. This way ciphertext in coefficient
            // form is passed to `fast_expand_crt_basis_p_over_q`  giving us same saving as if both ciphertexts were
            // in coefficient form.
            if (self.c[0].representation != rhs.c[0].representation)
                && (rhs.c[0].representation != Representation::Coefficient)
            {
                panic!("Different representation in multiply1 only allows when self is in `Evalaution`")
            }
        }

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
        if c00.representation != Representation::Evaluation {
            c00.change_representation(Representation::Evaluation);
            c01.change_representation(Representation::Evaluation);
        }
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
        let c_r0 = &c00 * &c10;

        // c00 * c11 + c01 * c10
        c00 *= &c11;
        c10 *= &c01;
        c00 += &c10;

        // c01 * c11
        c01 *= &c11;
        // println!("Tensor {:?}", now.elapsed());

        Ciphertext {
            c: vec![c_r0, c00, c01],
            params: self.params.clone(),
            level: self.level,
        }
    }

    pub fn multiply1(&self, rhs: &Ciphertext<T>) -> Ciphertext<T> {
        let mut res_ct = self.multiply1_lazy(rhs);
        res_ct.scale_and_round();
        res_ct
    }

    pub fn fma_reverse_inplace(&mut self, ct: &Ciphertext<T>, pt: &Plaintext<T>) {
        self.c.iter_mut().zip(ct.c.iter()).for_each(|(a, b)| {
            a.fma_reverse_inplace(b, pt.poly_ntt.as_ref().expect("Missing poly_ntt!"))
        });
    }

    pub fn sub_reversed_inplace(&mut self, p: &Poly<T>) {
        self.c[0].sub_reversed_inplace(p);
        self.c[1].neg_assign();
    }

    pub fn change_representation(&mut self, to: &Representation) {
        self.c
            .iter_mut()
            .for_each(|p| p.change_representation(to.clone()))
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn params(&self) -> Arc<BfvParameters<T>> {
        self.params.clone()
    }

    pub fn c_ref(&self) -> &[Poly<T>] {
        &self.c
    }

    pub fn c_ref_mut(&mut self) -> &mut [Poly<T>] {
        &mut self.c
    }

    pub fn is_zero(&self) -> bool {
        self.c.is_empty()
    }
}

impl<T> Mul<&Plaintext<T>> for &Ciphertext<T>
where
    T: Ntt,
{
    type Output = Ciphertext<T>;
    fn mul(self, rhs: &Plaintext<T>) -> Self::Output {
        warn!(
            self.c[0].representation != Representation::Evaluation,
            "Ciphertext must be in Evaluation form for Plaintext multiplication"
        );
        debug_assert!(self.c[0].representation == Representation::Evaluation);
        let c = self
            .c
            .iter()
            .map(|ct| ct * rhs.poly_ntt.as_ref().expect("Missing poly_ntt"))
            .collect_vec();
        Ciphertext {
            c,
            params: self.params.clone(),
            level: self.level,
        }
    }
}

impl<T> AddAssign<&Ciphertext<T>> for Ciphertext<T>
where
    T: Ntt,
{
    fn add_assign(&mut self, rhs: &Ciphertext<T>) {
        self.c.iter_mut().zip(rhs.c.iter()).for_each(|(a, b)| {
            *a += b;
        })
    }
}

impl<T> Add<&Ciphertext<T>> for &Ciphertext<T>
where
    T: Ntt,
{
    type Output = Ciphertext<T>;

    fn add(self, rhs: &Ciphertext<T>) -> Self::Output {
        let c = self
            .c
            .iter()
            .zip(rhs.c.iter())
            .map(|(a, b)| a + b)
            .collect_vec();
        Ciphertext {
            c,
            params: self.params.clone(),
            level: self.level,
        }
    }
}

impl<T> AddAssign<&Poly<T>> for Ciphertext<T>
where
    T: Ntt,
{
    fn add_assign(&mut self, rhs: &Poly<T>) {
        self.c[0] += rhs;
    }
}

impl<T> Add<&Poly<T>> for &Ciphertext<T>
where
    T: Ntt,
{
    type Output = Ciphertext<T>;
    fn add(self, rhs: &Poly<T>) -> Self::Output {
        let mut ct = self.clone();
        ct += rhs;
        ct
    }
}

impl<T> AddAssign<&Plaintext<T>> for Ciphertext<T>
where
    T: Ntt,
{
    fn add_assign(&mut self, rhs: &Plaintext<T>) {
        let mut poly = rhs.to_poly();
        poly.change_representation(self.c[0].representation.clone());
        *self += &poly;
    }
}

impl<T> Add<&Plaintext<T>> for &Ciphertext<T>
where
    T: Ntt,
{
    type Output = Ciphertext<T>;
    fn add(self, rhs: &Plaintext<T>) -> Self::Output {
        let mut poly = rhs.to_poly();
        poly.change_representation(self.c[0].representation.clone());
        self + &poly
    }
}

impl<T> SubAssign<&Ciphertext<T>> for Ciphertext<T>
where
    T: Ntt,
{
    fn sub_assign(&mut self, rhs: &Ciphertext<T>) {
        self.c.iter_mut().zip(rhs.c.iter()).for_each(|(a, b)| {
            *a -= b;
        })
    }
}

impl<T> Sub<&Ciphertext<T>> for &Ciphertext<T>
where
    T: Ntt,
{
    type Output = Ciphertext<T>;

    fn sub(self, rhs: &Ciphertext<T>) -> Self::Output {
        let c = self
            .c
            .iter()
            .zip(rhs.c.iter())
            .map(|(a, b)| a - b)
            .collect_vec();
        Ciphertext {
            c,
            params: self.params.clone(),
            level: self.level,
        }
    }
}

impl<T> SubAssign<&Poly<T>> for Ciphertext<T>
where
    T: Ntt,
{
    fn sub_assign(&mut self, rhs: &Poly<T>) {
        self.c[0] -= rhs;
    }
}

impl<T> Sub<&Poly<T>> for &Ciphertext<T>
where
    T: Ntt,
{
    type Output = Ciphertext<T>;
    fn sub(self, rhs: &Poly<T>) -> Self::Output {
        let mut ct = self.clone();
        ct -= rhs;
        ct
    }
}

impl<T> SubAssign<&Plaintext<T>> for Ciphertext<T>
where
    T: Ntt,
{
    fn sub_assign(&mut self, rhs: &Plaintext<T>) {
        let mut poly = rhs.to_poly();
        poly.change_representation(self.c[0].representation.clone());
        *self -= &poly;
    }
}

impl<T> Sub<&Plaintext<T>> for &Ciphertext<T>
where
    T: Ntt,
{
    type Output = Ciphertext<T>;
    fn sub(self, rhs: &Plaintext<T>) -> Self::Output {
        let mut poly = rhs.to_poly();
        poly.change_representation(self.c[0].representation.clone());
        self - &poly
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
        let params = Arc::new(BfvParameters::default(10, 1 << 4));
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
    fn ciphertext_mul_lazy() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::default(4, 1 << 15));
        let sk = SecretKey::random(&params, &mut rng);

        let m0 = rng
            .clone()
            .sample_iter(Uniform::new(0, params.plaintext_modulus))
            .take(params.polynomial_degree)
            .collect_vec();
        let m1 = rng
            .clone()
            .sample_iter(Uniform::new(0, params.plaintext_modulus))
            .take(params.polynomial_degree)
            .collect_vec();
        let pt0 = Plaintext::encode(&m0, &params, Encoding::simd(0));
        let pt1 = Plaintext::encode(&m1, &params, Encoding::simd(0));

        let ct0s = (0..20).map(|_| sk.encrypt(&pt0, &mut rng)).collect_vec();
        let ct1s = (0..20).map(|_| sk.encrypt(&pt1, &mut rng)).collect_vec();

        // lazy
        let now = std::time::Instant::now();
        let mut res_lazy = Ciphertext::zero(&params, 0);
        izip!(ct0s.iter(), ct1s.iter()).for_each(|(c0, c1)| {
            if res_lazy.is_zero() {
                res_lazy = c0.multiply1_lazy(c1);
            } else {
                res_lazy += &c0.multiply1_lazy(c1);
            }
        });
        res_lazy.scale_and_round();
        let time_lazy = now.elapsed();

        // not lazy
        let now = std::time::Instant::now();
        let mut res_not_lazy = Ciphertext::zero(&params, 0);
        izip!(ct0s.iter(), ct1s.iter()).for_each(|(c0, c1)| {
            if res_not_lazy.is_zero() {
                res_not_lazy = c0.multiply1(c1);
            } else {
                res_not_lazy += &c0.multiply1(c1);
            }
        });
        let time_not_lazy = now.elapsed();

        println!("Time: Lazy={:?}, NotLazy={:?}", time_lazy, time_not_lazy);
        println!(
            "Noise: Lazy={:?}, NotLazy={:?}",
            sk.measure_noise(&res_lazy, &mut rng),
            sk.measure_noise(&res_not_lazy, &mut rng),
        );
    }

    #[test]
    fn ciphertext_plaintext_mul() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::default(3, 1 << 15));
        let sk = SecretKey::random(&params, &mut rng);

        let mut m1 = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let mut m2 = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let pt1 = Plaintext::encode(&m1, &params, Encoding::simd(0));
        let pt2 = Plaintext::encode(&m2, &params, Encoding::simd(0));

        let mut ct = sk.encrypt(&pt1, &mut rng);
        // change representation of ct to evalaution
        ct.change_representation(&Representation::Evaluation);
        let ct_pt = &ct * &pt2;

        let res = sk.decrypt(&ct_pt).decode(Encoding::simd(0));
        params.plaintext_modulus_op.mul_mod_fast_vec(&mut m1, &m2);

        assert_eq!(res, m1);
    }

    #[test]
    fn ciphertext_plaintext_add() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::default(1, 1 << 3));
        let mut m0 = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let m1 = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let sk = SecretKey::random(&params, &mut rng);

        let mut ct = sk.encrypt(
            &Plaintext::encode(&m0, &params, Encoding::simd(0)),
            &mut rng,
        );
        let pt1 = Plaintext::encode(&m1, &params, Encoding::simd(0));
        let mut pt1_poly = pt1.to_poly();
        pt1_poly.change_representation(Representation::Coefficient);

        ct += &pt1_poly;

        let res = sk.decrypt(&ct).decode(Encoding::simd(0));
        params.plaintext_modulus_op.add_mod_fast_vec(&mut m0, &m1);
        assert_eq!(res, m0);
    }

    #[test]
    fn ciphertext_plaintext_sub() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::default(1, 1 << 3));
        let mut m0 = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let m1 = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let sk = SecretKey::random(&params, &mut rng);

        let mut ct = sk.encrypt(
            &Plaintext::encode(&m0, &params, Encoding::simd(0)),
            &mut rng,
        );
        let pt1 = Plaintext::encode(&m1, &params, Encoding::simd(0));
        let mut pt1_poly = pt1.to_poly();
        pt1_poly.change_representation(Representation::Coefficient);

        ct -= &pt1_poly;

        let res = sk.decrypt(&ct).decode(Encoding::simd(0));
        params.plaintext_modulus_op.sub_mod_fast_vec(&mut m0, &m1);
        assert_eq!(res, m0);
    }

    #[test]
    fn plaintext_ciphertext_sub() {
        let mut rng = thread_rng();
        let params = Arc::new(BfvParameters::default(1, 1 << 3));
        let m0 = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let mut m1 = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let sk = SecretKey::random(&params, &mut rng);

        let mut ct = sk.encrypt(
            &Plaintext::encode(&m0, &params, Encoding::simd(0)),
            &mut rng,
        );
        let pt1 = Plaintext::encode(&m1, &params, Encoding::simd(0));
        let mut pt1_poly = pt1.to_poly();
        pt1_poly.change_representation(Representation::Coefficient);

        ct.sub_reversed_inplace(&pt1_poly);

        let res = sk.decrypt(&ct).decode(Encoding::simd(0));
        params.plaintext_modulus_op.sub_mod_fast_vec(&mut m1, &m0);
        assert_eq!(res, m1);
    }
    #[test]
    fn clone_perf() {
        let params = Arc::new(BfvParameters::default(10, 1 << 15));

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
