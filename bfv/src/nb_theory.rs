use crate::modulus::Modulus;
use num_bigint_dig::{prime::probably_prime, BigUint};
use rand::{thread_rng, Rng};

pub fn generate_primes_vec(
    sizes: &[usize],
    polynomial_degree: usize,
    skip_list: &[u64],
) -> Vec<u64> {
    let mut primes = vec![];
    sizes.iter().for_each(|s| {
        let mut upper_bound = 1u64 << s;
        loop {
            if let Some(p) = generate_prime(*s, (2 * polynomial_degree) as u64, upper_bound) {
                if !primes.contains(&p) && !skip_list.contains(&p) {
                    primes.push(p);
                    break;
                } else {
                    upper_bound = p;
                }
            } else {
                panic!("Not enough primes");
            }
        }
    });
    primes
}

/// Finds prime such that prime % n == 1
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
    let p = Modulus::new(p);
    let m = (n as u64) * 2;

    let lambda = (p.modulus() - 1) / m;

    for _ in 0..100 {
        let mut root = rng.gen_range(0..p.modulus());
        root = p.exp(root, lambda as usize);
        if p.exp(root, m as usize) == 1 && p.exp(root, (m / 2) as usize) != 1 {
            return Some(root);
        }
    }
    None
}

//TODO: write tests for the above functions

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_prime_works() {
        let prime = generate_prime(51, 1 << 15, 1 << 51);
        dbg!(prime);
    }
}
