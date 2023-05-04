use std::panic::RefUnwindSafe;

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};

/// `mu` is the precomputed factor needed for barret reduction
///
/// mu = floor(2^2k / modulus ), where k=64
///
/// Returns mu_hi and mu_lo
fn compute_mu(modulus: u64) -> (u64, u64) {
    let mu = ((BigUint::one() << 128usize) / modulus).to_u128().unwrap();
    ((mu >> 64) as u64, mu as u64)
}

/// BarretReduction of 128bit values
fn barret_reduction_u128(a: u128, modulus: u64, mu_hi: u64, mu_lo: u64) -> u64 {
    // We need to calculate a * mu / 2^128
    // Notice that we don't need lower 128 bits of 256 bit product
    let a_hi = (a >> 64) as u64;
    let a_lo = a as u64;

    let mu_lo_a_lo_hi = ((mu_lo as u128 * a_lo as u128) >> 64) as u64;

    // carry part 1
    let middle = mu_hi as u128 * a_lo as u128;
    let middle_lo = middle as u64;
    let mut middle_hi = (middle >> 64) as u64;
    let carry_acc = middle_lo + mu_lo_a_lo_hi;
    if carry_acc < mu_lo_a_lo_hi {
        middle_hi += 1;
    }

    // carry part 2
    let middle = a_hi as u128 * mu_lo as u128;
    let middle_lo = middle as u64;
    let mut middle_hi2 = (middle >> 64) as u64;
    if middle_lo + carry_acc < middle_lo {
        middle_hi2 += 1;
    }

    // we only need lower 64 bits from higher 128 bits of (a*m / 2^128)
    let tmp = a_hi
        .wrapping_mul(mu_hi)
        .wrapping_add(middle_hi)
        .wrapping_add(middle_hi2);
    let mut result = a_lo - tmp.wrapping_mul(modulus);

    while result >= modulus {
        result -= modulus;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nb_theory::generate_prime;

    #[test]
    fn barret_reduction_u128_works() {
        let modulus = generate_prime(60, 16, 1 << 60).unwrap();
        let value = 10231039103u128 * 2131312313u128;

        let (mu_hi, mu_lo) = compute_mu(modulus);
        let res = barret_reduction_u128(value, modulus, mu_hi, mu_lo);
        dbg!(res, value % (modulus as u128));
    }
}
