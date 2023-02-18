use fhe_math::zq::Modulus;
use num_bigint_dig::{prime::probably_prime, BigUint};
use rand::{thread_rng, Rng};

/// Finds prime such that prime % 2 * n == 1
pub fn generate_prime(num_bits: usize, modulo: u64, upper_bound: u64) -> Option<u64> {
    let leading_zeros = (64 - num_bits) as u32;

    let mut tentative_prime = upper_bound - 1;
    while tentative_prime % modulo != 1 && tentative_prime.leading_zeros() == leading_zeros {
        tentative_prime -= 1;
    }

    while !probably_prime(&BigUint::from(tentative_prime), 0)
        && tentative_prime.leading_zeros() == leading_zeros
        && tentative_prime >= modulo
    {
        tentative_prime -= modulo;
    }

    if probably_prime(&BigUint::from(tentative_prime), 0)
        && tentative_prime.leading_zeros() == leading_zeros
    {
        Some(tentative_prime)
    } else {
        None
    }
}

// Finds 2n_th primitive root of unity in field mod p
pub fn primitive_element(p: u64, n: usize) -> Option<u64> {
    let mut rng = thread_rng();
    let p = Modulus::new(p).unwrap();
    let m = (n as u64) * 2;

    let lambda = (p.p - 1) / m;

    for _ in 0..100 {
        let mut root = rng.gen_range(0..p.p);
        root = p.pow(root, lambda);
        if p.pow(root, m) == 1 && p.pow(root, m / 2) != 1 {
            return Some(root);
        }
    }
    None
}

//TODO: write tests for the above functions
