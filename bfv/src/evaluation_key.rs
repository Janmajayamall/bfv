use crate::{rot_to_galois_element, BfvParameters, GaloisKey, RelinearizationKey, SecretKey};
use itertools::{izip, Itertools};
use rand::{CryptoRng, RngCore};
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub struct EvaluationKey {
    pub(crate) rlks: HashMap<usize, RelinearizationKey>,
    pub(crate) rtgs: HashMap<(isize, usize), GaloisKey>,
}

impl EvaluationKey {
    pub fn new<R: CryptoRng + RngCore>(
        params: &BfvParameters,
        sk: &SecretKey,
        rlk_levels: &[usize],
        rtg_levels: &[usize],
        rtg_indices: &[isize],
        rng: &mut R,
    ) -> EvaluationKey {
        assert!(rtg_levels.len() == rtg_indices.len());

        let mut rlks = HashMap::new();
        rlk_levels.iter().for_each(|l| {
            rlks.insert(*l, RelinearizationKey::new(params, sk, *l, rng));
        });

        let mut rtgs = HashMap::new();
        izip!(rtg_indices.iter(), rtg_levels.iter()).for_each(|(index, level)| {
            let el = {
                if *index == (2 * params.degree - 1) as isize {
                    2 * params.degree - 1
                } else {
                    rot_to_galois_element(*index, params.degree)
                }
            };
            rtgs.insert(
                (*index, *level),
                GaloisKey::new(el, params, *level, sk, rng),
            );
        });

        EvaluationKey { rlks, rtgs }
    }

    /// Create a new evaluation key by manually supplying
    /// (1) relinerization keys with corresponding levels
    /// (2) Galois key with corresponding levels and indices
    pub(crate) fn new_raw(
        rlk_levels: &[usize],
        rlks: Vec<RelinearizationKey>,
        rtg_levels: &[usize],
        rtg_indices: &[isize],
        gks: Vec<GaloisKey>,
    ) -> EvaluationKey {
        assert!(
            rlk_levels.len() == rlks.len(),
            "Specified rlk levels do not match with supplied rlks"
        );
        assert!(
            rtg_levels.len() == gks.len() && rtg_indices.len() == gks.len(),
            "Specified rtg levels/indices do not match with supplied gks"
        );

        let mut rlks_map = HashMap::new();
        izip!(rlk_levels.iter(), rlks.into_iter()).for_each(|(l, key)| {
            rlks_map.insert(*l, key);
        });

        let mut rtgs_map = HashMap::new();
        izip!(rtg_levels.iter(), rtg_indices.iter(), gks.into_iter()).for_each(
            |(level, index, key)| {
                rtgs_map.insert((*index, *level), key);
            },
        );

        EvaluationKey {
            rlks: rlks_map,
            rtgs: rtgs_map,
        }
    }

    pub fn get_rtg_ref(&self, rot_by: isize, level: usize) -> &GaloisKey {
        self.rtgs.get(&(rot_by, level)).expect("Rtg missing!")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
}
