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
        dbg!(mu);
        Modulus {
            mu_hi: 2,
            mu_lo: 2,
            mu: mu as u64,
            modulus,
            mod_bits: n,
        }
    }

    /// Barrett modulur multiplication. Assumes that a and b are < modulus
    ///
    /// If either of a and b are greater than modulus then q*mu overflows.
    fn mul_mod_fast(&self, a: u64, b: u64) -> u64 {
        // let mut ab = a as u128 * b as u128;
        let mut ab = (1u128 << 122) - 1;

        {
            dbg!(128 - ab.leading_zeros());
            dbg!(ab % (self.modulus as u128));
        }

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

        dbg!(ab);

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
        let prime = generate_prime(61, 16, 1 << 61).unwrap();
        let modulus = Modulus::new(prime);

        let mut rng = thread_rng();
        let a = rng.gen::<u64>();
        let b = rng.gen::<u64>();

        let p_bits = (128 - (a as u128 * b as u128).leading_zeros()) as u64;

        // let mu_bits = (64 - (modulus.mu.leading_zeros())) as u64;
        // let prod_bits = p_bits - (modulus.mod_bits - 2) + mu_bits;
        // dbg!(p_bits, mu_bits, prod_bits);

        let res = modulus.mul_mod_fast(a, b);
        // dbg!(res, (a as u128 * b as u128) % (prime as u128));
    }
}
