use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use std::panic::RefUnwindSafe;

/// Returns `mu` for 128 bit by 64 bit barrett reduction
///
/// mu = floor(2^128 / modulus)
fn compute_mu_double_reduction(modulus: u64) -> (u64, u64) {
    let mu = ((BigUint::one() << 128usize) / modulus).to_u128().unwrap();
    ((mu >> 64) as u64, mu as u64)
}

/// BarretReduction of 128 bits modulus 64 bits
/// Source: Menezes, Alfred; Oorschot, Paul; Vanstone, Scott. Handbook of Applied Cryptography, Section 14.3.3.
///
/// Implementation reference: https://github.com/openfheorg/openfhe-development/blob/055c89778d3d0dad00479150a053124137f4c3ca/src/core/include/utils/utilities-int.h#L59
pub fn barret_reduction_u128(a: u128, modulus: u64, mu_hi: u64, mu_lo: u64) -> u64 {
    // We need to calculate a * mu / 2^128
    // Notice that we don't need lower 128 bits of 256 bit product
    let a_hi = (a >> 64) as u64;
    let a_lo = a as u64;

    let mu_lo_a_lo_hi = ((mu_lo as u128 * a_lo as u128) >> 64) as u64;

    // carry part 1
    let middle = mu_hi as u128 * a_lo as u128;
    let middle_lo = middle as u64;
    let mut middle_hi = (middle >> 64) as u64;
    let (carry_acc, carry) = middle_lo.overflowing_add(mu_lo_a_lo_hi);
    middle_hi += carry as u64;

    // carry part 2
    let middle = a_hi as u128 * mu_lo as u128;
    let middle_lo = middle as u64;
    let mut middle_hi2 = (middle >> 64) as u64;
    let (_, carry2) = middle_lo.overflowing_add(carry_acc);
    middle_hi2 += carry2 as u64;

    // we only need lower 64 bits from higher 128 bits of (a*m / 2^128)
    let tmp = a_hi
        .wrapping_mul(mu_hi)
        .wrapping_add(middle_hi)
        .wrapping_add(middle_hi2);
    let mut result = a_lo.wrapping_sub(tmp.wrapping_mul(modulus));

    while result >= modulus {
        result -= modulus;
    }

    result
}

/// Implementation to compare perf against
///
/// Taken from https://github.com/tlepoint/fhe.rs/blob/725c2217a1f71cc2701bbbe9cc86ef8fb0b097cb/crates/fhe-math/src/zq/mod.rs#L619
///
/// Note: If no difference is observed on intel as well, then it's better to use this one.
pub const fn barret_reduction_u128_m2(a: u128, p: u64, barrett_hi: u64, barrett_lo: u64) -> u64 {
    let a_lo = a as u64;
    let a_hi = (a >> 64) as u64;
    let p_lo_lo = ((a_lo as u128) * (barrett_lo as u128)) >> 64;
    let p_hi_lo = (a_hi as u128) * (barrett_lo as u128);
    let p_lo_hi = (a_lo as u128) * (barrett_hi as u128);

    let q = ((p_lo_hi + p_hi_lo + p_lo_lo) >> 64) + (a_hi as u128) * (barrett_hi as u128);
    let mut r = (a - q * (p as u128)) as u64;

    while r >= p {
        r -= p;
    }
    r
}
#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::{
        distributions::{uniform::SampleUniform, Standard, Uniform},
        thread_rng, Rng,
    };

    use super::*;
    use crate::nb_theory::generate_prime;

    #[test]
    fn barret_reduction_u128_works() {
        let mut rng = thread_rng();
        for _ in 0..1000 {
            let modulus = generate_prime(60, 16, 1 << 60).unwrap();
            let value: u128 = rng.gen();
            dbg!(value);
            let (mu_hi, mu_lo) = compute_mu_double_reduction(modulus);
            let res = barret_reduction_u128(value, modulus, mu_hi, mu_lo);
            assert_eq!(res, (value % (modulus as u128)) as u64);
        }
    }

    #[test]
    fn barret_reduction_u128_perf() {
        let mut rng = thread_rng();
        let modulus = 1152921504606846577u64;
        let (mu_hi, mu_lo) = compute_mu_double_reduction(modulus);
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
            .map(|v| barret_reduction_u128(*v, modulus, mu_hi, mu_lo))
            .collect_vec();
        println!("time barret_reduction_u128: {:?}", now.elapsed());

        let now = std::time::Instant::now();
        let expected = values
            .iter()
            .map(|v| (*v % (modulus as u128)) as u64)
            .collect_vec();
        println!("time baseline: {:?}", now.elapsed());

        assert!(res == expected);
        // assert!(res2 == expected)
    }
}
