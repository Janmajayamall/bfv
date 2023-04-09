use crate::{
    poly::{Poly, PolyContext, Representation},
    SecretKey,
};
use crypto_bigint::rand_core::CryptoRngCore;
use itertools::izip;
use rand::{CryptoRng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

struct KeySwitchingKey {
    c0s: Box<[Poly]>,
    c1s: Box<[Poly]>,
    seed: <ChaCha8Rng as SeedableRng>::Seed,
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
        let c1s = Self::generate_c1(ciphertext_ctx.moduli.len(), ksk_ctx, &seed);
        let c0s = Self::generate_c0(ciphertext_ctx, ksk_ctx, poly, &c1s, sk, rng);

        KeySwitchingKey {
            c0s: c0s.into_boxed_slice(),
            c1s: c1s.into_boxed_slice(),
            seed,
        }
    }

    pub fn generate_c1(
        count: usize,
        ksk_ctx: &Arc<PolyContext>,
        seed: <ChaCha8Rng as SeedableRng>::Seed,
    ) -> Vec<Poly> {
        let mut rng = ChaCha8Rng::from_seed(seed);
        (0..count)
            .into_iter()
            .map(|| Poly::random(poly_context, &Representation::Evaluation, &mut rng))
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

        izip!(ciphertext_ctx.g.iter(), c1s.iter()).map(|(g, c1)| {
            let mut g = Poly::try_convert_from_biguint(
                vec![g; ksk_ctx.degree],
                ksk_ctx,
                &Representation::Evaluation,
            );
            // m
            g *= poly;
            let mut e = Poly::random_gaussian(ksk_ctx, &Representation::Coefficient, 10, rng);
            e.change_representation(Representation::Evaluation);
            e += &g;
            e -= &(&c1 * &sk);
            e
        })
    }
}

#[cfg(test)]
mod tests {
    fn key_switching_works() {}
}
