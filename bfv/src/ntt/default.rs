use itertools::Itertools;

use crate::{primitive_element, Modulus};

// Implementation is taken from [gauss](https://github.com/gaussian-dev/gauss)

/// Forward butterfly routine for Number theoretic transform. Given inputs `x <
/// 4q` and `y < 4q` mutates x and y in place to equal x' and y' such that
/// x' = x + wy
/// y' = x - wy
/// where both x' and y' are in range [0, 4q)
///
/// We implement Algorithm 4 of [FASTER ARITHMETIC FOR NUMBER-THEORETIC TRANSFORMS](https://arxiv.org/pdf/1205.2926.pdf)
pub unsafe fn forward_butterly(
    x: *mut u64,
    y: *mut u64,
    w: &u64,
    w_shoup: &u64,
    q: &u64,
    q_twice: &u64,
) {
    debug_assert!(*x < *q * 4, "{} >= (4q){}", *x, 4 * q);
    debug_assert!(*y < *q * 4, "{} >= (4q){}", *y, 4 * q);

    if *x >= *q_twice {
        *x = *x - q_twice;
    }

    // TODO (Jay): Hot path expected. How expensive is it?
    let k = ((*w_shoup as u128 * *y as u128) >> 64) as u64;
    let t = w.wrapping_mul(*y).wrapping_sub(k.wrapping_mul(*q));

    *y = *x + q_twice - t;
    *x = *x + t;
}

/// Inverse butterfly routine of Inverse Number theoretic transform. Given
/// inputs `x < 2q` and `y < 2q` mutates x and y to equal x' and y' such that x'
/// = x + y y' = w(x - y)
/// where x' and y' in range [0, 2q)
///
/// We implement Algorithm 3 of [FASTER ARITHMETIC FOR NUMBER-THEORETIC TRANSFORMS](https://arxiv.org/pdf/1205.2926.pdf)
pub unsafe fn inverse_butterfly(
    x: *mut u64,
    y: *mut u64,
    w_inv: &u64,
    w_inv_shoup: &u64,
    q: &u64,
    q_twice: &u64,
) {
    debug_assert!(*x < *q_twice, "{} >= (2q){q_twice}", *x);
    debug_assert!(*y < *q_twice, "{} >= (2q){q_twice}", *y);

    let mut x_dash = *x + *y;
    if x_dash >= *q_twice {
        x_dash -= q_twice
    }

    let t = *x + q_twice - *y;
    let k = ((*w_inv_shoup as u128 * t as u128) >> 64) as u64; // TODO (Jay): Hot path
    *y = w_inv.wrapping_mul(t).wrapping_sub(k.wrapping_mul(*q));

    *x = x_dash;
}

/// Number theoretic transform of vector `a` with each
/// element in range [0, 2q).Outputs NTT(a) where each element is in range
/// [0,2q)
///
/// We implement Cooley-tukey based forward NTT as given in Algorithm 1 of https://eprint.iacr.org/2016/504.pdf.
pub fn ntt_lazy(a: &mut [u64], psi: &[u64], psi_shoup: &[u64], q: u64, q_twice: u64) {
    debug_assert!(a.len() == psi.len());

    let n = a.len();
    let mut t = n;

    let mut m = 1;
    while m < n {
        t >>= 1;

        for i in 0..m {
            let j_1 = 2 * i * t;
            let j_2 = j_1 + t;

            unsafe {
                let w = psi.get_unchecked(m + i);
                let w_shoup = psi_shoup.get_unchecked(m + i);
                for j in j_1..j_2 {
                    let x = a.get_unchecked_mut(j) as *mut u64;
                    let y = a.get_unchecked_mut(j + t) as *mut u64;
                    forward_butterly(x, y, w, w_shoup, &q, &q_twice);
                }
            }
        }

        m <<= 1;
    }

    a.iter_mut().for_each(|a0| {
        if *a0 >= q_twice {
            *a0 -= q_twice
        }
    });
}

/// Inverse number theoretic transform of input vector `a` with each element in
/// range [0, 2q). Outputs vector INTT(a) with each element in range [0, 2q)
///
/// We implement backward number theorectic transform using GS algorithm as given in Algorithm 2 of https://eprint.iacr.org/2016/504.pdf
pub fn ntt_inv_lazy(
    a: &mut [u64],
    psi_inv: &[u64],
    psi_inv_shoup: &[u64],
    n_inv: u64,
    q: u64,
    q_twice: u64,
) {
    debug_assert!(a.len() == psi_inv.len());

    let mut m = a.len();
    let mut t = 1;
    while m > 1 {
        let mut j_1: usize = 0;
        let h = m >> 1;
        for i in 0..h {
            let j_2 = j_1 + t;
            unsafe {
                let w_inv = psi_inv.get_unchecked(h + i);
                let w_inv_shoup = psi_inv_shoup.get_unchecked(h + i);

                for j in j_1..j_2 {
                    let x = a.get_unchecked_mut(j) as *mut u64;
                    let y = a.get_unchecked_mut(j + t) as *mut u64;
                    inverse_butterfly(x, y, w_inv, w_inv_shoup, &q, &q_twice);
                }
            }
            j_1 = j_1 + 2 * t;
        }
        t *= 2;
        m >>= 1;
    }

    a.iter_mut()
        .for_each(|a0| *a0 = ((*a0 as u128 * n_inv as u128) % q as u128) as u64);
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct NttOperator {
    q: u64,
    q_twice: u64,
    n: u64,
    n_inv: u64,
    psi_powers_bo: Box<[u64]>,
    psi_inv_powers_bo: Box<[u64]>,
    psi_powers_bo_shoup: Box<[u64]>,
    psi_inv_powers_bo_shoup: Box<[u64]>,
}

impl NttOperator {
    fn reduce_from_lazy(&self, a: &mut [u64]) {
        a.iter_mut().for_each(|a0| {
            let q = self.q;
            if *a0 >= q {
                *a0 = *a0 - q;
            }
        });
    }
}

impl traits::Ntt for NttOperator {
    fn new(n: usize, prime: u64) -> Self {
        // \psi = 2n^{th} primitive root of unity in F_q
        let psi = primitive_element(prime, n).expect("Unable to find 2n^th root of unity");

        let modulus = Modulus::new(prime);
        let psi_inv = modulus.inv(psi);

        let mut psi_powers = Vec::with_capacity(n as usize);
        let mut psi_inv_powers = Vec::with_capacity(n as usize);
        let mut running_psi = 1;
        let mut running_psi_inv = 1;
        for _ in 0..n {
            psi_powers.push(running_psi);
            psi_inv_powers.push(running_psi_inv);

            running_psi = modulus.mul_mod_fast(running_psi, psi);
            running_psi_inv = modulus.mul_mod_fast(running_psi_inv, psi_inv);
        }

        // powers stored in bit reversed order
        let mut psi_powers_bo = vec![0u64; n as usize];
        let mut psi_inv_powers_bo = vec![0u64; n as usize];
        let shift_by = n.leading_zeros() + 1;
        for i in 0..n as usize {
            // i in bit reversed order
            let bo_index = i.reverse_bits() >> shift_by;

            psi_powers_bo[bo_index] = psi_powers[i];
            psi_inv_powers_bo[bo_index] = psi_inv_powers[i];
        }

        // shoup representation
        let psi_powers_bo_shoup = psi_powers_bo
            .iter()
            .map(|v| modulus.compute_shoup(*v))
            .collect_vec();
        let psi_inv_powers_bo_shoup = psi_inv_powers_bo
            .iter()
            .map(|v| modulus.compute_shoup(*v))
            .collect_vec();

        // n^{-1} \mod{q}
        let n_inv = modulus.inv(n as u64);

        NttOperator {
            q: prime,
            q_twice: 2 * prime,
            n: n as u64,
            n_inv,
            psi_powers_bo: psi_powers_bo.into_boxed_slice(),
            psi_inv_powers_bo: psi_inv_powers_bo.into_boxed_slice(),
            psi_powers_bo_shoup: psi_powers_bo_shoup.into_boxed_slice(),
            psi_inv_powers_bo_shoup: psi_inv_powers_bo_shoup.into_boxed_slice(),
        }
    }

    fn forward(&self, a: &mut [u64]) {
        ntt_lazy(
            a,
            &self.psi_powers_bo,
            &self.psi_powers_bo_shoup,
            self.q,
            self.q_twice,
        );
        self.reduce_from_lazy(a);
    }

    fn backward(&self, a: &mut [u64]) {
        ntt_inv_lazy(
            a,
            &self.psi_inv_powers_bo,
            &self.psi_inv_powers_bo_shoup,
            self.n_inv,
            self.q,
            self.q_twice,
        );
        self.reduce_from_lazy(a);
    }

    fn forward_lazy(&self, a: &mut [u64]) {
        ntt_lazy(
            a,
            &self.psi_powers_bo,
            &self.psi_powers_bo_shoup,
            self.q,
            self.q_twice,
        );
    }

    // fn backward_lazy(&self, a: &mut [Self::Scalar]) {
    //     ntt_inv_lazy(
    //         a,
    //         &self.psi_inv_powers_bo,
    //         &self.psi_inv_powers_bo_shoup,
    //         self.n_inv,
    //         self.q,
    //         self.q_twice,
    //     );
    // }
}
