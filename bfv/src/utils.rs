use crate::Modulus;
use itertools::Itertools;
use num_bigint::BigUint;
use num_bigint_dig::{BigUint as BigUintDig, ModInverse};
use num_traits::{FromPrimitive, ToPrimitive};
use rand::{CryptoRng, RngCore};

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

pub fn mod_inverse_biguint_u64(a: &BigUint, m: u64) -> BigUint {
    let a_dig = BigUintDig::from_bytes_le(&a.to_bytes_le());
    let m_dig = BigUintDig::from_u64(m).unwrap();
    BigUint::from_bytes_le(
        &a_dig
            .mod_inverse(m_dig)
            .unwrap()
            .to_biguint()
            .unwrap()
            .to_bytes_le(),
    )
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

pub fn convert_ternary_to_bytes(values: &[i64]) -> Vec<u8> {
    // map ternary distrubtion {-1,0,1} to {2,0,1}
    let values = values
        .iter()
        .map(|v| if *v == -1 { 2u64 } else { *v as u64 })
        .collect_vec();

    let bits = 2;
    let mask = (1u64 << bits) - 1;

    let mut bytes = vec![];

    let mut value_index = 0;
    let mut curr_byte = 0u8;
    while value_index != values.len() {
        // since 8/2 = 4, a byte can fit 4 2 bit values.
        for i in 0..4 {
            curr_byte |= ((values[value_index] & mask) << (i * 2)) as u8;
            value_index += 1;

            if value_index >= values.len() {
                // if curr_byte holds some value, then append it
                if i != 0 {
                    bytes.push(curr_byte);
                }
                break;
            }
        }
        bytes.push(curr_byte);
        curr_byte = 0;
    }

    bytes
}

pub fn convert_bytes_to_ternary(bytes: &[u8], length: usize) -> Vec<i64> {
    // extract 4 2 bits value from each byte
    let mut values = vec![];
    let bits = 2;
    let mask = (1u8 << bits) - 1;
    bytes.iter().for_each(|v| {
        for i in 0..4 {
            values.push((*v >> (i * 2)) & mask);
        }
    });

    // map {2,0,1} to {-1,0,1}
    let values = values
        .iter()
        .take(length)
        .map(|v| if *v == 2 { -1i64 } else { *v as i64 })
        .collect_vec();

    values
}

pub fn convert_to_bytes(values: &[u64], modulus: u64) -> Vec<u8> {
    let bits = 64 - modulus.leading_zeros();
    let mask = (1 << bits) - 1;

    // we assume that modulus has atleast 8 bits
    assert!(bits >= 8);

    let bytes_count = ((bits as usize * values.len()) as f64 / 8.0).ceil() as usize;
    let mut bytes = Vec::with_capacity(bytes_count);

    let mut value_index = 0;

    let mut curr_val = values[value_index] & mask;
    let mut curr_val_left = bits;

    loop {
        if curr_val_left < 8 {
            // extract left over bits in curr_val
            let mut byte = (curr_val & ((1 << curr_val_left) - 1)) as u8;

            value_index += 1;

            if value_index != values.len() {
                curr_val = values[value_index] & mask;
                let left_over_space = 8 - curr_val_left;

                // extract bits equivalent to space left in byte and set them in byte
                byte |= ((curr_val & ((1 << left_over_space) - 1)) as u8) << curr_val_left;

                curr_val_left = bits - left_over_space;
                curr_val >>= left_over_space;
                bytes.push(byte);
            } else {
                // since curr_val is last, push the extracted bits
                bytes.push(byte);
                break;
            }
        } else {
            // extract a byte at once
            let byte = (curr_val & ((1 << 8) - 1)) as u8;
            bytes.push(byte);

            curr_val >>= 8;
            curr_val_left -= 8;
        }
    }

    bytes
}

pub fn convert_from_bytes(bytes: &[u8], modulus: u64) -> Vec<u64> {
    let bits = 64 - modulus.leading_zeros();

    let values_count = (bytes.len() * 8) / bits as usize;
    let mut values = Vec::with_capacity(values_count);
    let mut byte_index = 0;
    let mut curr_value_fill = 0;
    let mut curr_value = 0u64;

    let mut value_index = 0;

    loop {
        if bits - curr_value_fill < 8 {
            let left_over_bits = bits - curr_value_fill;
            let mut b = bytes[byte_index] as u64;

            // extract left over bits and set them in their position in current value
            curr_value |= ((b & ((1 << left_over_bits) - 1)) << curr_value_fill);
            // curr_val is filled
            values.push(curr_value);
            value_index += 1;

            curr_value = 0;
            curr_value_fill = 0;

            // lose left_over_bits
            b >>= left_over_bits;

            // `8 - let_over_bits` are for the next value
            curr_value |= b;

            if value_index != values_count {
                curr_value_fill += 8 - left_over_bits;
            } else {
                assert!(byte_index + 1 == bytes.len());
                break;
            }
        } else {
            curr_value |= ((bytes[byte_index] as u64) << curr_value_fill);
            curr_value_fill += 8;
        }

        byte_index += 1;
    }

    values
}

/// Sample a vector of independent centered binomial distributions of a given
/// variance. Returns an error if the variance is strictly larger than 16.
///
/// Credit [fhe.rs](https://github.com/tlepoint/fhe.rs)
pub fn sample_vec_cbd<R: RngCore + CryptoRng>(
    vector_size: usize,
    variance: usize,
    rng: &mut R,
) -> Result<Vec<i64>, &'static str> {
    if !(1..=16).contains(&variance) {
        return Err("The variance should be between 1 and 16");
    }

    let mut out = Vec::with_capacity(vector_size);

    let number_bits = 4 * variance;
    let mask_add = ((u64::MAX >> (64 - number_bits)) >> (2 * variance)) as u128;
    let mask_sub = mask_add << (2 * variance);

    let mut current_pool = 0u128;
    let mut current_pool_nbits = 0;

    for _ in 0..vector_size {
        if current_pool_nbits < number_bits {
            current_pool |= (rng.next_u64() as u128) << current_pool_nbits;
            current_pool_nbits += 64;
        }
        debug_assert!(current_pool_nbits >= number_bits);
        out.push(
            ((current_pool & mask_add).count_ones() as i64)
                - ((current_pool & mask_sub).count_ones() as i64),
        );
        current_pool >>= number_bits;
        current_pool_nbits -= number_bits;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::generate_prime;

    use super::*;

    #[test]
    fn galois_el_works() {
        let mut v = vec![];
        for i in 0..4 {
            v.push(rot_to_galois_element(i, 8));
        }
        dbg!(v);
    }

    #[test]
    fn convert_to_and_from_bytes() {
        for prime_bits in [17, 43, 50, 59] {
            let prime = generate_prime(prime_bits, 16, 1 << prime_bits).unwrap();
            let modq = Modulus::new(prime);

            let mut rng = thread_rng();
            let values = modq.random_vec(1 << 8, &mut rng);

            let bytes = convert_to_bytes(&values, modq.modulus());
            let values_res = convert_from_bytes(&bytes, modq.modulus());
            assert_eq!(values, values_res);
        }
    }
}
