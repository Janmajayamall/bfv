struct Modulus {
    mu_hi: u64,
    mu_lo: u64,
    mu: u64,
    modulus: u64,
    mod_bits: u64,
}

impl Modulus {
    fn new(modulus: u64) -> Modulus {
        // mu = 2^(2n+3) / modulus
        let n = 64 - (modulus.leading_zeros() as u64);
        dbg!(n);
        let mu = (1u128 << (2 * n + 3)) / (modulus as u128);

        Modulus {
            mu_hi: 2,
            mu_lo: 2,
            mu: mu as u64,
            modulus,
            mod_bits: n,
        }
    }

    /// Barrett modulur multiplication. Assumes that a and b are < modulus
    fn mul_mod_fast(&self, a: u64, b: u64) -> u64 {
        let mut ab = a as u128 * b as u128;

        let n = self.mod_bits;
        let alpha = n + 3;
        // beta = -2

        let mut q = ab >> (n - 2);
        q *= (self.mu as u128);
        q >>= (alpha + 2);

        ab -= q * (self.modulus as u128);
        let mut ab = ab as u64;

        if ab >= self.modulus {
            ab -= self.modulus;
        }

        ab
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    use crate::nb_theory::generate_prime;

    #[test]
    fn mul_mod_fast_works() {
        let prime = generate_prime(54, 16, 1 << 54).unwrap();
        let modulus = Modulus::new(prime);

        let mut rng = thread_rng();
        let a = rng.gen::<u64>();
        let b = rng.gen::<u64>();

        dbg!(128 - (a as u128 * b as u128).leading_zeros());

        let res = modulus.mul_mod_fast(a, b);
        dbg!(res, (a as u128 * b as u128) % (prime as u128));
    }
}
