use num_traits::ToPrimitive;
use rand::{CryptoRng, RngCore};

use crate::{
    Ciphertext, HybridKeySwitchingKey, Modulus, Poly, PolyContext, Representation, SecretKey,
    Substitution,
};
use std::sync::Arc;

pub struct GaloisKey {
    ciphertext_ctx: Arc<PolyContext>,
    substitution: Substitution,
    ksk_key: HybridKeySwitchingKey,
}

impl GaloisKey {
    pub fn new<R: CryptoRng + RngCore>(
        exponent: usize,
        ciphertext_ctx: &Arc<PolyContext>,
        secret_key: &SecretKey,
        rng: &mut R,
    ) -> GaloisKey {
        let substitution = Substitution::new(exponent, ciphertext_ctx.degree);

        // Substitute secret key
        let mut sk_poly = Poly::try_convert_from_i64_small(
            secret_key.coefficients.as_ref(),
            ciphertext_ctx,
            &crate::Representation::Coefficient,
        );
        sk_poly.change_representation(crate::Representation::Evaluation);
        let sk_poly = sk_poly.substitute(&substitution);

        // Generate key switching key for substituted secret key
        let ksk_key = HybridKeySwitchingKey::new(&sk_poly, secret_key, ciphertext_ctx, rng);

        GaloisKey {
            ciphertext_ctx: ciphertext_ctx.clone(),
            substitution,
            ksk_key,
        }
    }

    pub fn rotate(&self, ct: &Ciphertext) -> Ciphertext {
        debug_assert!(ct.c.len() == 2);

        // Key switch c1
        let mut c1 = ct.c[1].substitute(&self.substitution);
        if c1.representation == Representation::Evaluation {
            c1.change_representation(crate::Representation::Coefficient);
        }

        let (mut cs0, mut cs1) = self.ksk_key.switch(&c1);

        // Key switch returns polynomial in Evaluation form
        if ct.c[0].representation != cs0.representation {
            cs0.change_representation(ct.c[0].representation.clone());
            cs1.change_representation(ct.c[0].representation.clone());
        }

        cs0 += &ct.c[0].substitute(&self.substitution);

        Ciphertext {
            c: vec![cs0, cs1],
            params: ct.params.clone(),
            level: ct.level,
        }
    }

    /// Retursn galois element correponding to desired rotation by i.
    ///
    /// Galois element: 3^i % M is will rotate left by i
    /// Galois element: 3^(N/2-1) will rotate right by i
    pub fn rot_to_galois_element(i: isize, n: usize) -> usize {
        let m = 2 * n;
        let modm = Modulus::new(m as u64);
        if i > 0 {
            modm.exp(3, i as usize) as usize
        } else {
            modm.exp(3, n / 2 - (i.abs().to_usize().unwrap())) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use super::*;
    use crate::{BfvParameters, Encoding, Plaintext, SecretKey};

    #[test]
    fn rotation_works() {
        let bfv_params = Arc::new(BfvParameters::default(3, 1 << 15));
        let mut rng = thread_rng();
        let sk = SecretKey::random(&bfv_params, &mut rng);

        let m = bfv_params
            .plaintext_modulus_op
            .random_vec(bfv_params.polynomial_degree, &mut rng);
        let pt = Plaintext::encode(&m, &bfv_params, Encoding::simd(0));
        let ct = sk.encrypt(&pt, &mut rng);
        dbg!(sk.measure_noise(&ct, &mut rng));
        // rotate left by 1
        let galois_key = GaloisKey::new(3, &bfv_params.ciphertext_poly_contexts[0], &sk, &mut rng);

        let ct_rotated = galois_key.rotate(&ct);
        dbg!(sk.measure_noise(&ct_rotated, &mut rng));

        let res_m = sk.decrypt(&ct_rotated).decode(Encoding::simd(0));
        // dbg!(m, res_m);
    }

    #[test]
    fn galois_el_works() {
        let mut v = vec![];
        for i in 0..4 {
            v.push(GaloisKey::rot_to_galois_element(i, 8));
        }
        dbg!(v);
    }
}
