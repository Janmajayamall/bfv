use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};
use std::panic::RefUnwindSafe;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::{
        distributions::{uniform::SampleUniform, Standard, Uniform},
        thread_rng, Rng,
    };

    use super::*;
    use crate::nb_theory::generate_prime;
}
