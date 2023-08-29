use std::mem::MaybeUninit;

use itertools::{izip, Itertools};
use num_bigint::U64Digits;
use num_bigint_dig::{prime::probably_prime, BigUint};
use num_traits::{One, ToPrimitive};
use rand::{distributions::Uniform, CryptoRng, Rng, RngCore};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Modulus {
    mu_hi: u64,
    mu_lo: u64,
    mu: u64,
    modulus: u64,
    mod_bits: u64,
}

impl Modulus {
    pub fn new(modulus: u64) -> Modulus {
        // mu = 2^(2n+3) / modulus
        let n = 64 - (modulus.leading_zeros() as u64);
        let mu = (1u128 << (2 * n + 3)) / (modulus as u128);

        // mu for 128 bits by 64 bits barrett reduction
        // mu = floor(2^128 / m)
        let mu_u128 = ((BigUint::one() << 128usize) / modulus).to_u128().unwrap();

        Modulus {
            mu_hi: (mu_u128 >> 64) as u64,
            mu_lo: mu_u128 as u64,
            mu: mu as u64,
            modulus,
            mod_bits: n,
        }
    }

    pub const fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Computes modulus exponentiation using binary exponentiation
    pub fn exp(&self, mut a: u64, mut e: usize) -> u64 {
        let mut r = 1u64;
        while e != 0 {
            if e & 1 == 1 {
                r = self.mul_mod_fast(r, a);
            }
            a = self.mul_mod_fast(a, a);
            e >>= 1;
        }
        r
    }

    /// Computes multiplicative inverse of a
    ///
    /// modulus must be prime
    pub fn inv(&self, a: u64) -> u64 {
        debug_assert!(probably_prime(&BigUint::from(self.modulus), 0));
        debug_assert!(a < self.modulus);
        self.exp(a, (self.modulus - 2) as usize)
    }

    /// Computes shoup representation
    ///
    /// a should be smaller than modulus
    pub const fn compute_shoup(&self, a: u64) -> u64 {
        debug_assert!(a < self.modulus);
        (((a as u128) << 64) / self.modulus as u128) as u64
    }

    /// Barrett modulus addition
    pub const fn add_mod(&self, mut a: u64, mut b: u64) -> u64 {
        if a >= self.modulus {
            a = self.reduce(a);
        }
        if b >= self.modulus {
            b = self.reduce(b);
        }

        a += b;
        if a >= self.modulus {
            a -= self.modulus;
        }

        a
    }

    /// Naive modulus addition. Uses %
    pub const fn add_mod_naive(&self, a: u64, b: u64) -> u64 {
        (a + b) % self.modulus
    }

    /// Modulus addition
    ///
    /// Assumes both a and b are smaller than modulus
    pub const fn add_mod_fast(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.modulus);
        debug_assert!(b < self.modulus);

        let mut c = a + b;
        if c >= self.modulus {
            c -= self.modulus;
        }
        c
    }

    /// Barret modulus subtraction
    fn sub_mod(&self, mut a: u64, mut b: u64) -> u64 {
        if a >= self.modulus {
            a = self.reduce(a);
        }
        if b >= self.modulus {
            b = self.reduce(b);
        }

        if a >= b {
            a - b
        } else {
            (a + self.modulus) - b
        }
    }

    /// Naive modulus subtraction. Uses %
    pub const fn sub_mod_naive(&self, a: u64, b: u64) -> u64 {
        (a + self.modulus - b) % self.modulus
    }

    /// Modulus subtraction
    ///
    /// Assumes both a and b < modulus
    pub fn sub_mod_fast(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.modulus);
        debug_assert!(b < self.modulus);

        if a >= b {
            a - b
        } else {
            (a + self.modulus) - b
        }
    }

    pub fn neg_mod_fast(&self, a: u64) -> u64 {
        debug_assert!(a < self.modulus);

        #[cfg(debug_assertions)]
        {
            if a == 0 {
                return 0;
            } else {
                return self.modulus - a;
            }
        };

        return self.modulus - a;
    }

    /// Naive modulus multiplication. Uses %
    pub const fn mul_mod_naive(&self, a: u64, b: u64) -> u64 {
        ((a as u128 * b as u128) % (self.modulus as u128)) as u64
    }

    /// Barrett modulur multiplication. Assumes that a and b are < modulus.
    ///
    /// Refer to implementation notes for more details
    pub fn mul_mod_fast(&self, a: u64, b: u64) -> u64 {
        // TODO: uncomment this
        // debug_assert!(a < self.modulus);
        debug_assert!(b < self.modulus);

        let mut ab = a as u128 * b as u128;
        let n = self.mod_bits;
        let alpha = n + 3;
        // beta = -2

        let mut q = ab >> (n - 2);
        q *= (self.mu as u128);
        q >>= (alpha + 2);

        ab -= q * (self.modulus as u128);
        let mut ab = ab as u64;

        // correction
        if ab >= self.modulus {
            ab -= self.modulus;
        }

        ab
    }

    /// Shoup modulur multiplication
    ///
    /// a and b should be smaller than modulus
    pub const fn mul_mod_shoup(&self, a: u64, b: u64, b_shoup: u64) -> u64 {
        debug_assert!(self.compute_shoup(b) == b_shoup);
        debug_assert!(a < self.modulus);
        debug_assert!(b < self.modulus);

        let q = (a as u128 * b_shoup as u128) >> 64;
        let mut r = ((a as u128 * b as u128) - (q * self.modulus as u128)) as u64;

        if r >= self.modulus {
            r -= self.modulus
        }

        r
    }

    /// Barrett modulus reduction of 64 bits value.
    pub const fn reduce(&self, a: u64) -> u64 {
        let n = self.mod_bits;
        let alpha = n + 3;
        // beta = -2

        let mut a = a as u128;
        let mut q = a >> (n - 2);
        q *= (self.mu as u128);
        q >>= (alpha + 2);

        a -= q * (self.modulus as u128);
        let mut a = a as u64;

        // correction
        if a >= self.modulus {
            a -= self.modulus;
        }
        a
    }

    pub const fn reduce_naive(&self, a: u64) -> u64 {
        a % self.modulus
    }

    pub const fn reduce_naive_u128(&self, a: u128) -> u64 {
        (a % (self.modulus as u128)) as u64
    }

    /// BarretReduction of 128 bits value by 64 bits modulus
    /// Source: Menezes, Alfred; Oorschot, Paul; Vanstone, Scott. Handbook of Applied Cryptography, Section 14.3.3.
    ///
    /// Implementation reference: https://github.com/openfheorg/openfhe-development/blob/055c89778d3d0dad00479150a053124137f4c3ca/src/core/include/utils/utilities-int.h#L59
    pub fn barret_reduction_u128(&self, a: u128) -> u64 {
        // We need to calculate a * mu / 2^128
        // Notice that we don't need lower 128 bits of 256 bit product
        let a_hi = (a >> 64) as u64;
        let a_lo = a as u64;

        let mu_lo_a_lo_hi = ((self.mu_lo as u128 * a_lo as u128) >> 64) as u64;

        // carry part 1
        let middle = self.mu_hi as u128 * a_lo as u128;
        let middle_lo = middle as u64;
        let mut middle_hi = (middle >> 64) as u64;
        let (carry_acc, carry) = middle_lo.overflowing_add(mu_lo_a_lo_hi);
        middle_hi += carry as u64;

        // carry part 2
        let middle = a_hi as u128 * self.mu_lo as u128;
        let middle_lo = middle as u64;
        let mut middle_hi2 = (middle >> 64) as u64;
        let (_, carry2) = middle_lo.overflowing_add(carry_acc);
        middle_hi2 += carry2 as u64;

        // we only need lower 64 bits from higher 128 bits of (a*m / 2^128)
        let tmp = a_hi
            .wrapping_mul(self.mu_hi)
            .wrapping_add(middle_hi)
            .wrapping_add(middle_hi2);
        let mut result = a_lo.wrapping_sub(tmp.wrapping_mul(self.modulus));

        while result >= self.modulus {
            result -= self.modulus;
        }

        result
    }

    /// Implementation to compare perf against
    ///
    /// Taken from https://github.com/tlepoint/fhe.rs/blob/725c2217a1f71cc2701bbbe9cc86ef8fb0b097cb/crates/fhe-math/src/zq/mod.rs#L619
    pub const fn barret_reduction_u128_v2(&self, a: u128) -> u64 {
        let a_lo = a as u64;
        let a_hi = (a >> 64) as u64;
        let p_lo_lo = ((a_lo as u128) * (self.mu_lo as u128)) >> 64;
        let p_hi_lo = (a_hi as u128) * (self.mu_lo as u128);
        let p_lo_hi = (a_lo as u128) * (self.mu_lo as u128);

        let q = ((p_lo_hi + p_hi_lo + p_lo_lo) >> 64) + (a_hi as u128) * (self.mu_hi as u128);
        let mut r = (a - q * (self.modulus as u128)) as u64;

        while r >= self.modulus {
            r -= self.modulus;
        }
        r
    }

    pub fn compute_shoup_vec(&self, a: &[u64]) -> Vec<u64> {
        a.iter().map(|v| self.compute_shoup(*v)).collect()
    }

    pub fn add_mod_naive_vec(&self, a: &mut [u64], b: &[u64]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(va, vb)| *va = self.add_mod_naive(*va, *vb));
    }

    pub fn add_mod_fast_vec(&self, a: &mut [u64], b: &[u64]) {
        #[cfg(not(feature = "hexl"))]
        izip!(a.iter_mut(), b.iter()).for_each(|(va, vb)| *va = self.add_mod_fast(*va, *vb));

        #[cfg(feature = "hexl")]
        hexl_rs::elwise_add_mod(a, b, self.modulus, a.len() as u64)
    }

    pub fn add_mod_fast_vec_uninit(&self, r: &mut [MaybeUninit<u64>], a: &[u64], b: &[u64]) {
        #[cfg(not(feature = "hexl"))]
        izip!(r.iter_mut(), a.iter(), b.iter()).for_each(|(vr, va, vb)| {
            vr.write(self.add_mod_fast(*va, *vb));
        });

        #[cfg(feature = "hexl")]
        hexl_rs::elwise_add_mod_uinit(r, a, b, self.modulus, r.len() as u64)
    }

    pub fn sub_mod_naive_vec(&self, a: &mut [u64], b: &[u64]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(va, vb)| *va = self.sub_mod_naive(*va, *vb));
    }

    /// Modulus subtraction
    ///
    /// Assumes each element in vec a and b are smaller than modulus
    pub fn sub_mod_fast_vec(&self, a: &mut [u64], b: &[u64]) {
        #[cfg(not(feature = "hexl"))]
        izip!(a.iter_mut(), b.iter()).for_each(|(va, vb)| *va = self.sub_mod_fast(*va, *vb));

        #[cfg(feature = "hexl")]
        hexl_rs::elwise_sub_mod(a, b, self.modulus, a.len() as u64)
    }

    pub fn sub_mod_fast_vec_uninit(&self, r: &mut [MaybeUninit<u64>], a: &[u64], b: &[u64]) {
        #[cfg(not(feature = "hexl"))]
        izip!(r.iter_mut(), a.iter(), b.iter()).for_each(|(vr, va, vb)| {
            vr.write(self.sub_mod_fast(*va, *vb));
        });

        #[cfg(feature = "hexl")]
        hexl_rs::elwise_sub_mod_uinit(r, a, b, self.modulus, r.len() as u64)
    }

    pub fn neg_mod_fast_vec(&self, a: &mut [u64]) {
        izip!(a.iter_mut()).for_each(|va| *va = self.neg_mod_fast(*va));
    }

    pub fn neg_mod_fast_vec_uninit(&self, r: &mut [MaybeUninit<u64>], a: &[u64]) {
        izip!(r.iter_mut(), a.iter()).for_each(|(vr, va)| {
            vr.write(self.neg_mod_fast(*va));
        });
    }

    /// subracts a from b
    ///
    /// Adding this because we don't haave a way to express that perform b - a and consume a.
    pub fn sub_mod_fast_vec_reversed(&self, a: &mut [u64], b: &[u64]) {
        #[cfg(not(feature = "hexl"))]
        izip!(a.iter_mut(), b.iter()).for_each(|(va, vb)| *va = self.sub_mod_fast(*vb, *va));

        #[cfg(feature = "hexl")]
        hexl_rs::elwise_sub_reversed_mod(a, b, self.modulus, a.len() as u64)
    }

    pub fn mul_mod_naive_vec(&self, a: &mut [u64], b: &[u64]) {
        izip!(a.iter_mut(), b.iter()).for_each(|(va, vb)| *va = self.mul_mod_naive(*va, *vb));
    }

    pub fn mul_mod_fast_vec(&self, a: &mut [u64], b: &[u64]) {
        #[cfg(not(feature = "hexl"))]
        a.iter_mut()
            .zip(b.iter())
            .for_each(|(va, vb)| *va = self.mul_mod_fast(*va, *vb));

        #[cfg(feature = "hexl")]
        hexl_rs::elwise_mult_mod(a, b, self.modulus, a.len() as u64, 1)
    }

    pub fn mul_mod_fast_vec_uninit(&self, r: &mut [MaybeUninit<u64>], a: &[u64], b: &[u64]) {
        #[cfg(not(feature = "hexl"))]
        izip!(r.iter_mut(), a.iter(), b.iter()).for_each(|(vr, va, vb)| {
            vr.write(self.mul_mod_fast(*va, *vb));
        });

        #[cfg(feature = "hexl")]
        hexl_rs::elwise_mult_mod_uinit(r, a, b, self.modulus, r.len() as u64, 1);
    }

    pub fn mul_mod_shoup_vec(&self, a: &mut [u64], b: &[u64], b_shoup: &[u64]) {
        izip!(a.iter_mut(), b.iter(), b_shoup.iter())
            .for_each(|(va, vb, vb_shoup)| *va = self.mul_mod_shoup(*va, *vb, *vb_shoup));
    }

    /// Barrett modulus multiplication of scalar with vector a
    ///
    /// Assumes scalar and all elements in a are smaller than modulus
    pub fn scalar_mul_mod_fast_vec(&self, a: &mut [u64], b: u64) {
        #[cfg(not(feature = "hexl"))]
        a.iter_mut().for_each(|v| {
            *v = self.mul_mod_fast(*v, b);
        });

        #[cfg(feature = "hexl")]
        hexl_rs::elwise_mult_scalar_mod(a, b, self.modulus, a.len() as u64, 1)
    }

    // FIXME: reduce_vec performs a lot worse than reduce_naive_vec on macos (not on x86)
    pub fn reduce_vec(&self, a: &mut [u64]) {
        #[cfg(not(feature = "hexl"))]
        a.iter_mut().for_each(|v| {
            *v = self.reduce(*v);
        });

        #[cfg(feature = "hexl")]
        hexl_rs::elem_reduce_mod(a, self.modulus, a.len() as u64, self.modulus, 1);
    }

    pub fn reduce_naive_vec(&self, a: &mut [u64]) {
        a.iter_mut().for_each(|v| {
            *v = self.reduce_naive(*v);
        });
    }

    pub fn reduce_naive_u128_vec(&self, a: &[u128]) -> Vec<u64> {
        a.iter().map(|v| self.reduce_naive_u128(*v)).collect()
    }

    pub fn barret_reduction_u128_vec(&self, a: &[u128]) -> Vec<u64> {
        a.iter().map(|v| self.barret_reduction_u128(*v)).collect()
    }

    pub fn barret_reduction_u128_v2_vec(&self, a: &[u128]) -> Vec<u64> {
        a.iter()
            .map(|v| self.barret_reduction_u128_v2(*v))
            .collect()
    }

    /// Modulus reduction of i64 values with small bound
    ///
    /// Assumes magnitude of all values is smaller than modulus
    pub fn reduce_vec_i64_small(&self, a: &[i64]) -> Vec<u64> {
        a.iter()
            .map(|v| {
                if *v < 0 {
                    ((self.modulus as i64) + *v) as u64
                } else {
                    *v as u64
                }
            })
            .collect()
    }

    pub fn random_vec<R: CryptoRng + RngCore>(&self, size: usize, rng: &mut R) -> Vec<u64> {
        rng.sample_iter(Uniform::new(0, self.modulus))
            .take(size)
            .collect()
    }

    /// Switch modulus of values from old_modulus to new_modulus
    ///
    /// delta = abs(o - n)
    /// if n >= o:
    ///     if v > o/2:
    ///         v' = v + delta
    ///     else:
    ///         v' = v
    /// else:
    ///     if v > o/2:
    ///         v' = (v - delta) % n
    ///     else:
    ///         v' = v % n
    pub fn switch_modulus(values: &mut [u64], old_modulus: u64, new_modulus: u64) {
        let delta = if old_modulus > new_modulus {
            old_modulus - new_modulus
        } else {
            new_modulus - old_modulus
        };
        let o_half = old_modulus >> 1;

        #[cfg(not(feature = "hexl"))]
        {
            values.iter_mut().for_each(|a| {
                if new_modulus > old_modulus {
                    if *a > o_half {
                        *a += delta;
                    }
                } else {
                    if *a > o_half {
                        *a = (*a - (delta % new_modulus)) % new_modulus;
                    } else {
                        *a = *a % new_modulus;
                    }
                }
            });
        }

        #[cfg(feature = "hexl")]
        {
            if new_modulus > old_modulus {
                hexl_rs::elwise_cmp_add(values, o_half, delta, 6, values.len() as u64);
            } else {
                hexl_rs::elwise_cmp_sub_mod(
                    values,
                    o_half,
                    delta % new_modulus,
                    6,
                    new_modulus,
                    values.len() as u64,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nb_theory::generate_prime;
    use itertools::Itertools;
    use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
    use num_traits::Zero;
    use rand::{thread_rng, Rng};

    #[test]
    fn inverse_works() {
        let prime = generate_prime(60, 16, 1 << 60).unwrap();
        let modulus = Modulus::new(prime);

        let mut rng = thread_rng();
        for _ in 0..1000 {
            let a = rng.gen::<u64>() % prime;
            let res = modulus.inv(a);
            let expected = BigUintDig::from(a)
                .mod_inverse(BigUintDig::from(prime))
                .unwrap()
                .to_u64()
                .unwrap();
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn exponentiation_works() {
        let prime = generate_prime(60, 16, 1 << 60).unwrap();
        let modulus = Modulus::new(prime);

        let mut rng = thread_rng();
        for _ in 0..1000 {
            let a = rng.gen::<u64>() % prime;
            let e = rng.gen::<usize>();
            let res = modulus.exp(a, e);
            let expected = BigUintDig::from(a)
                .modpow(&BigUintDig::from(e), &BigUintDig::from(prime))
                .to_u64()
                .unwrap();
            assert_eq!(res, expected);
        }
    }

    #[test]
    fn reduce_works() {
        let prime = generate_prime(60, 16, 1 << 60).unwrap();
        let modulus = Modulus::new(prime);

        let mut rng = thread_rng();
        for _ in 0..1000 {
            let a = rng.gen::<u64>();
            let res = modulus.reduce(a);
            assert_eq!(res, a % prime);
        }
    }

    #[test]
    fn add_mod_works() {
        let prime = generate_prime(60, 16, 1 << 60).unwrap();
        let modulus = Modulus::new(prime);

        let mut rng = thread_rng();
        for _ in 0..1000 {
            let a = rng.gen::<u64>();
            let b = rng.gen::<u64>();
            let res1 = modulus.add_mod(a, b);
            let res2 = modulus.add_mod_fast(a % prime, b % prime);
            let expected = ((a % prime) + (b % prime)) % prime;
            assert_eq!(res1, expected);
            assert_eq!(res2, expected);
        }
    }

    #[test]
    fn sub_mod_works() {
        let prime = generate_prime(60, 16, 1 << 60).unwrap();
        let modulus = Modulus::new(prime);

        let mut rng = thread_rng();
        for _ in 0..1000 {
            let mut a = rng.gen::<u64>();
            let mut b = rng.gen::<u64>();
            let res1 = modulus.sub_mod(a, b);
            let res2 = modulus.sub_mod(a % prime, b % prime);
            a %= prime;
            b %= prime;
            let expected = ((a + prime) - b) % prime;
            assert_eq!(res1, expected);
            assert_eq!(res2, expected);
        }
    }

    // TODO: write perf for mul_mod_fast
    #[test]
    fn mul_mod_fast_works() {
        let prime = generate_prime(60, 16, 1 << 60).unwrap();
        let modulus = Modulus::new(prime);
        let mut rng = thread_rng();
        for _ in 0..1000 {
            let a = rng.gen::<u64>() % prime;
            let b = rng.gen::<u64>() % prime;

            let res = modulus.mul_mod_fast(a, b);
            assert_eq!(res, ((a as u128 * b as u128) % (prime as u128)) as u64);
        }
    }

    #[test]
    fn mul_mod_shoup_works() {
        let prime = generate_prime(60, 16, 1 << 60).unwrap();
        let modulus = Modulus::new(prime);
        let mut rng = thread_rng();
        for _ in 0..1000 {
            let a = rng.gen::<u64>() % prime;
            let b = rng.gen::<u64>() % prime;

            let b_shoup = modulus.compute_shoup(b);

            let res = modulus.mul_mod_shoup(a, b, b_shoup);
            assert_eq!(res, ((a as u128 * b as u128) % (prime as u128)) as u64);
        }
    }

    #[test]
    fn barret_reduction_u128_works() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let prime = generate_prime(60, 16, 1 << 60).unwrap();
            let modulus = Modulus::new(prime);
            let value: u128 = rng.gen();
            let res = modulus.barret_reduction_u128(value);
            assert_eq!(res, (value % (prime as u128)) as u64);
        }
    }

    #[test]
    #[ignore]
    fn barret_reduction_u128_perf() {
        let mut rng = thread_rng();
        let prime = 1152921504606846577u64;
        let modulus = Modulus::new(prime);
        let values = rng
            .sample_iter(Uniform::new(0, u128::MAX))
            .take(1 << 26)
            .collect_vec();

        // let now = std::time::Instant::now();
        // let res = values
        //     .iter()
        //     .map(|v| barret_reduction_u128_m2(*v, modulus, mu_hi, mu_lo))
        //     .collect_vec();
        // println!("time barret_reduction_u128_m2: {:?}", now.elapsed());

        let now = std::time::Instant::now();
        let res = values
            .iter()
            .map(|v| modulus.barret_reduction_u128(*v))
            .collect_vec();
        println!("time barret_reduction_u128: {:?}", now.elapsed());

        let now = std::time::Instant::now();
        let expected = values
            .iter()
            .map(|v| (*v % (prime as u128)) as u64)
            .collect_vec();
        println!("time baseline: {:?}", now.elapsed());

        assert!(res == expected);
        // assert!(res2 == expected)
    }

    #[test]
    fn switch_modulus_works() {
        let mut rng = thread_rng();

        let prime = generate_prime(60, 16, 1 << 60).unwrap();
        let prime2 = generate_prime(52, 16, 1 << 52).unwrap();

        let vp = Modulus::new(prime).random_vec(8, &mut rng);
        let mut vp_res = vp.clone();
        let mut vp_res2 = vp.clone();
        Modulus::switch_modulus(&mut vp_res, prime, prime2);
        assert_eq!(
            vp_res,
            vp.iter()
                .map(|v| {
                    if *v > (prime >> 1) {
                        prime2 - ((prime - v) % prime2)
                    } else {
                        *v % prime2
                    }
                })
                .collect_vec()
        );
    }

    #[test]
    fn test_perf_fma() {
        let size = 1 << 3;
        let count = 512;
        let mut rng = thread_rng();
        let prime = generate_prime(60, 1 << 16, 1 << 60).unwrap();
        let modulus = Modulus::new(prime);
        dbg!(size, prime);
        for _ in 0..1 {
            let mut a = (0..count)
                .map(|_| modulus.random_vec(size, &mut rng))
                .collect_vec();
            let b = (0..count)
                .map(|_| modulus.random_vec(size, &mut rng))
                .collect_vec();

            // Note: this method overflows at addition if log(size) > 128 - log(prime)*2
            let now = std::time::Instant::now();
            let mut d = vec![0u128; size];
            izip!(a.iter(), b.iter()).for_each(|(a0, b0)| {
                izip!(d.iter_mut(), a0.iter(), b0.iter()).for_each(|(r, a1, b1)| {
                    *r += (*a1 as u128 * *b1 as u128);
                });
            });
            let r = modulus.barret_reduction_u128_vec(&d);
            println!("time u128: {:?}", now.elapsed());

            let mut a_clone = a.clone();
            let now = std::time::Instant::now();
            let mut r1 = vec![0u64; size];
            izip!(a_clone.iter_mut(), b.iter()).for_each(|(a0, b0)| {
                modulus.mul_mod_fast_vec(a0, b0);
                modulus.add_mod_fast_vec(&mut r1, a0);
            });
            println!("time:  {:?}", now.elapsed());
            assert_eq!(r1, r);
        }
    }

    #[test]
    fn test_perf_lazy_add() {
        let size = 1 << 3;
        let count = 256;
        let mut rng = thread_rng();
        let prime = generate_prime(60, 1 << 16, 1 << 60).unwrap();
        let modulus = Modulus::new(prime);
        dbg!(size, prime);
        for _ in 0..1 {
            let mut a = (0..count)
                .map(|_| modulus.random_vec(size, &mut rng))
                .collect_vec();
            let b = (0..count)
                .map(|_| modulus.random_vec(size, &mut rng))
                .collect_vec();

            // Note: this method overflows at addition if log(size) > 128 - log(prime)*2

            let mut a_clone = a.clone();
            let now = std::time::Instant::now();
            let mut r1 = vec![0u64; size];
            a_clone.iter_mut().for_each(|a0| {
                modulus.add_mod_fast_vec(&mut r1, a0);
            });
            println!("time:  {:?}", now.elapsed());

            let now = std::time::Instant::now();
            let mut d = vec![0u128; size];
            a.iter().for_each(|a0| {
                izip!(d.iter_mut(), a0.iter()).for_each(|(r, a1)| {
                    *r += *a1 as u128;
                });
            });
            let r = modulus.barret_reduction_u128_vec(&d);
            println!("time u128: {:?}", now.elapsed());

            assert_eq!(r1, r);
        }
    }
}
