use crate::{
    poly::{Poly, PolyContext, Representation},
    SecretKey,
};
use crypto_bigint::rand_core::CryptoRngCore;
use itertools::{izip, Itertools};
use ndarray::s;
use rand::{CryptoRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

struct KeySwitchingKey {
    c0s: Box<[Poly]>,
    c1s: Box<[Poly]>,
    seed: <ChaCha8Rng as SeedableRng>::Seed,
    ciphertext_ctx: Arc<PolyContext>,
    ksk_ctx: Arc<PolyContext>,
}

impl KeySwitchingKey {
    pub fn new<R: CryptoRng + CryptoRngCore>(
        poly: &Poly,
        sk: &SecretKey,
        ciphertext_ctx: &Arc<PolyContext>,
        rng: &mut R,
    ) -> KeySwitchingKey {
        // check that ciphertext context has more than on moduli, otherwise key switching does not makes sense
        debug_assert!(ciphertext_ctx.moduli.len() > 1);

        let ksk_ctx = &poly.context;

        // c1s
        let mut seed = <ChaCha8Rng as SeedableRng>::Seed::default();
        rng.fill_bytes(&mut seed);
        let c1s = Self::generate_c1(ciphertext_ctx.moduli.len(), ksk_ctx, seed);
        let c0s = Self::generate_c0(ciphertext_ctx, ksk_ctx, poly, &c1s, sk, rng);

        KeySwitchingKey {
            c0s: c0s.into_boxed_slice(),
            c1s: c1s.into_boxed_slice(),
            seed,
            ciphertext_ctx: ciphertext_ctx.clone(),
            ksk_ctx: ksk_ctx.clone(),
        }
    }

    pub fn switch(&self, poly: &Poly) -> Vec<Poly> {
        debug_assert!(poly.context == self.ciphertext_ctx);
        debug_assert!(poly.representation == Representation::Coefficient);

        let mut p = Poly::try_convert_from_u64(
            poly.coefficients.slice(s![0, ..]).as_slice().unwrap(),
            &self.ksk_ctx,
            &Representation::Coefficient,
        );
        p.change_representation(Representation::Evaluation);
        let mut c1_out = &self.c1s[0] * &p;
        p *= &self.c0s[0];
        let mut c0_out = p;

        izip!(
            self.c0s.iter(),
            self.c1s.iter(),
            poly.coefficients.outer_iter()
        )
        .skip(1)
        .for_each(|(c0, c1, rests)| {
            let mut p = Poly::try_convert_from_u64(
                rests.as_slice().unwrap(),
                &self.ksk_ctx,
                &Representation::Coefficient,
            );
            p.change_representation(Representation::Evaluation);

            c1_out += &(c1 * &p);
            p *= c0;
            c0_out += &p;
        });

        vec![c0_out, c1_out]
    }

    pub fn generate_c1(
        count: usize,
        ksk_ctx: &Arc<PolyContext>,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
    ) -> Vec<Poly> {
        let mut rng = ChaCha8Rng::from_seed(seed);
        (0..count)
            .into_iter()
            .map(|_| Poly::random(ksk_ctx, &Representation::Evaluation, &mut rng))
            .collect_vec()
    }

    pub fn generate_c0<R: CryptoRng + CryptoRngCore>(
        ciphertext_ctx: &Arc<PolyContext>,
        ksk_ctx: &Arc<PolyContext>,
        poly: &Poly,
        c1s: &[Poly],
        sk: &SecretKey,
        rng: &mut R,
    ) -> Vec<Poly> {
        // encrypt g corresponding to every qi in ciphertext
        // make sure that you have enough c1s
        debug_assert!(ciphertext_ctx.moduli.len() == c1s.len());
        debug_assert!(poly.representation == Representation::Evaluation);

        let mut sk =
            Poly::try_convert_from_i64(&sk.coefficients, ksk_ctx, &Representation::Coefficient);
        sk.change_representation(Representation::Evaluation);

        izip!(ciphertext_ctx.g.into_iter(), c1s.iter())
            .map(|(g, c1)| {
                let mut g = Poly::try_convert_from_biguint(
                    vec![g.clone(); ksk_ctx.degree].as_slice(),
                    ksk_ctx,
                    &Representation::Evaluation,
                );
                // m
                g *= poly;
                let mut e = Poly::random_gaussian(ksk_ctx, &Representation::Coefficient, 10, rng);
                e.change_representation(Representation::Evaluation);
                e += &g;
                e -= &(c1 * &sk);
                e
            })
            .collect_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BfvParameters;
    use num_bigint::BigUint;
    use rand::thread_rng;

    #[test]
    fn key_switching_works() {
        let bfv_params = Arc::new(BfvParameters::new(&[60, 60, 60, 60, 60], 65537, 8));
        let ct_ctx = bfv_params.ciphertext_poly_contexts[0].clone();
        let ksk_ctx = ct_ctx.clone();

        let mut rng = thread_rng();

        let sk = SecretKey::random(&bfv_params, &mut rng);

        let poly = Poly::random(&ksk_ctx, &Representation::Evaluation, &mut rng);
        let ksk = KeySwitchingKey::new(&poly, &sk, &ct_ctx, &mut rng);

        let mut other_poly = Poly::random(&ct_ctx, &Representation::Coefficient, &mut rng);
        let cs = ksk.switch(&other_poly);

        let mut sk_poly =
            Poly::try_convert_from_i64(&sk.coefficients, &ksk_ctx, &Representation::Coefficient);
        sk_poly.change_representation(Representation::Evaluation);
        let mut res = &cs[0] + &(&cs[1] * &sk_poly);

        // expected
        other_poly.change_representation(Representation::Evaluation);
        other_poly *= &poly;
        // other_poly.change_representation(Representation::Coefficient);

        res -= &other_poly;
        res.change_representation(Representation::Coefficient);

        izip!(Vec::<BigUint>::from(&res).iter(),).for_each(|v| {
            let diff_bits = std::cmp::min(v.bits(), (ksk_ctx.modulus() - v).bits());
            dbg!(diff_bits);
        });
    }
}
