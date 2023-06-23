use crate::Modulus;
use num_bigint::BigUint;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::ToPrimitive;

#[macro_export]
macro_rules! warn {
    ($con:expr, $($txt:tt)*) => {
        #[cfg(debug_assertions)]
        if $con {
            println!($($txt)*)
        }
    };
}

/// Retursn galois element correponding to desired rotation by i.
///
/// Galois element: 3^i % M is will rotate left by i
/// Galois element: 3^(N/2-i) will rotate right by i
pub fn rot_to_galois_element(i: isize, n: usize) -> usize {
    let m = 2 * n;
    let modm = Modulus::new(m as u64);
    if i > 0 {
        modm.exp(3, i as usize) as usize
    } else {
        modm.exp(3, n / 2 - (i.abs().to_usize().unwrap())) as usize
    }
}

pub fn mod_inverse_biguint(a: &BigUint, m: &BigUint) -> BigUint {
    let a_dig = BigUintDig::from_bytes_le(&a.to_bytes_le());
    let m_dig = BigUintDig::from_bytes_le(&m.to_bytes_le());
    BigUint::from_bytes_le(
        &a_dig
            .mod_inverse(m_dig)
            .unwrap()
            .to_biguint()
            .unwrap()
            .to_bytes_le(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn galois_el_works() {
        let mut v = vec![];
        for i in 0..4 {
            v.push(rot_to_galois_element(i, 8));
        }
        dbg!(v);
    }
}
