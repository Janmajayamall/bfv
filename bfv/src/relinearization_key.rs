use itertools::DedupBy;
use rand::{CryptoRng, RngCore};
use traits::Ntt;

use crate::{
    BfvParameters, Ciphertext, HybridKeySwitchingKey, Poly, PolyContext, Representation, SecretKey,
};
use std::sync::Arc;

pub struct RelinearizationKey<T: Ntt> {
    ciphertext_ctx: Arc<PolyContext<T>>,
    ksk: HybridKeySwitchingKey<T>,
}

impl<T> RelinearizationKey<T>
where
    T: Ntt,
{
    pub fn new<R: CryptoRng + RngCore>(
        params: &Arc<BfvParameters<T>>,
        sk: &SecretKey<T>,
        level: usize,
        rng: &mut R,
    ) -> RelinearizationKey<T> {
        let ctx = params.ciphertext_ctx_at_level(level);
        let mut sk_poly =
            Poly::try_convert_from_i64_small(&sk.coefficients, &ctx, &Representation::Coefficient);
        sk_poly.change_representation(Representation::Evaluation);

        // sk^2
        let sk_sq = &sk_poly * &sk_poly;

        // Key switching key
        let ksk = HybridKeySwitchingKey::new(&sk_sq, sk, &ctx, rng);

        RelinearizationKey {
            ciphertext_ctx: ctx,
            ksk,
        }
    }

    pub fn relinearize(&self, ct: &Ciphertext<T>) -> Ciphertext<T> {
        // switch fn in ksk already checks for matchin poly ctx. Don't check here
        debug_assert!(ct.c.len() == 3); // otherwise invalid relinerization
        debug_assert!(ct.c[0].representation == Representation::Coefficient);

        let (mut cs0, mut cs1) = self.ksk.switch(&ct.c[2]);
        cs0.change_representation(Representation::Coefficient);
        cs1.change_representation(Representation::Coefficient);
        cs0 += &ct.c[0];
        cs1 += &ct.c[1];

        Ciphertext {
            c: vec![cs0, cs1],
            params: ct.params.clone(),
            level: ct.level,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;

    use crate::{Encoding, Plaintext};

    use super::*;

    #[test]
    fn relinerization_works() {
        let params = Arc::new(BfvParameters::default(3, 8));

        let mut rng = thread_rng();
        let sk = SecretKey::random(&params, &mut rng);

        let m = params
            .plaintext_modulus_op
            .random_vec(params.polynomial_degree, &mut rng);
        let pt = Plaintext::encode(&m, &params, Encoding::simd(0));

        let ct = sk.encrypt(&pt, &mut rng);
        let ct2 = sk.encrypt(&pt, &mut rng);

        let ct3 = ct.multiply1(&ct2);

        // rlk key
        let rlk = RelinearizationKey::new(&params, &sk, 0, &mut rng);

        // relinearize
        let ct3_rl = rlk.relinearize(&ct3);

        // decrypt and check equivalence!
        let res_m = sk.decrypt(&ct3_rl).decode(Encoding::simd(0));
        let mut m_clone = m.clone();
        params
            .plaintext_modulus_op
            .mul_mod_fast_vec(&mut m_clone, &m);
        assert_eq!(m_clone, res_m);
    }
}