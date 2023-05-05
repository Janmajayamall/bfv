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
        let mu = (1u128 << (2 * n + 3)) / (modulus as u128);
        Modulus {
            mu_hi: 2,
            mu_lo: 2,
            mu: mu as u64,
            modulus,
            mod_bits: n,
        }
    }

    /// Computes shoup representation
    ///
    /// a should be smaller than modulus
    fn compute_shoup(&self, a: u64) -> u64 {
        debug_assert!(a < self.modulus);
        (((a as u128) << 64) / self.modulus) as u64
    }

    /// Barrett modulus reduction of 64 bits value.
    fn reduce(&self, a: u64) -> u64 {
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

    /// Barrett modulus addition
    fn add_mod(&self, mut a: u64, mut b: u64) -> u64 {
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

    /// Modulus addition
    ///
    /// Assumes both a and b are smaller than modulus
    fn add_mod_fast(&self, a: u64, b: u64) -> u64 {
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

    /// Modulus subtraction
    ///
    /// Assumes both a and b < modulus
    fn sub_mod_fast(&self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.modulus);
        debug_assert!(b < self.modulus);

        if a >= b {
            a - b
        } else {
            (a + self.modulus) - b
        }
    }

    /// Barrett modulur multiplication. Assumes that a and b are < modulus.
    ///
    /// Refer to implementation notes for more details
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

        // correction
        if ab >= self.modulus {
            ab -= self.modulus;
        }

        ab
    }

    /// Shoup modulur multiplication
    ///
    /// a and b should be smaller than modulus
    fn mul_mod_shoup(&self, a: u64, b: u64, b_shoup: u64) -> u64 {
        debug_assert!(self.compute_shoup(b) == b_shoup);
        debug_assert!(a < self.modulus);
        debug_assert!(b < self.modulus);

        let q = (a as u128 * b_shoup as u128) >> 64;
        let r = ((a as u128 * b as u128) - (q * self.modulus as u128)) as u64;

        if r >= self.modulus {
            r -= self.modulus
        }

        r
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use super::*;
    use crate::nb_theory::generate_prime;

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
}
