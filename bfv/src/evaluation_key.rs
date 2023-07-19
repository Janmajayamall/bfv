use std::collections::HashMap;

use itertools::{izip, Itertools};
use rand::{CryptoRng, RngCore};

use crate::{
    proto, rot_to_galois_element, traits::TryFromWithParameters, BfvParameters, GaloisKey,
    RelinearizationKey, SecretKey,
};

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

    pub fn get_rtg_ref(&self, rot_by: isize, level: usize) -> &GaloisKey {
        self.rtgs.get(&(rot_by, level)).expect("Rtg missing!")
    }
}

impl TryFromWithParameters for proto::EvaluationKey {
    type Parameters = BfvParameters;
    type Value = EvaluationKey;
    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        // since HashMap iterates over values in arbitrary order seralisation of same `EvaluationKey`
        // twice can produce different `proto::EvaluationKey`s.
        let rlks = value
            .rlks
            .iter()
            .map(|(i, k)| proto::RelinearizationKey::try_from_with_parameters(&k, parameters))
            .collect_vec();
        let mut rot_indices = vec![];
        let rtgs = value
            .rtgs
            .iter()
            .map(|(i, k)| {
                rot_indices.push(i.0 as i32);
                proto::GaloisKey::try_from_with_parameters(&k, parameters)
            })
            .collect_vec();

        proto::EvaluationKey {
            rlks,
            rtgs,
            rot_indices,
        }
    }
}

impl TryFromWithParameters for EvaluationKey {
    type Parameters = BfvParameters;
    type Value = proto::EvaluationKey;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self {
        let mut rlks = HashMap::new();
        value.rlks.iter().for_each(|v| {
            let v = RelinearizationKey::try_from_with_parameters(v, parameters);
            rlks.insert(v.level, v);
        });

        let mut rtgs = HashMap::new();
        value
            .rtgs
            .iter()
            .zip(value.rot_indices.iter())
            .for_each(|(gk, rot_index)| {
                let v = GaloisKey::try_from_with_parameters(gk, parameters);
                rtgs.insert((*rot_index as isize, v.level), v);
            });

        EvaluationKey { rlks, rtgs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use rand::thread_rng;

    #[test]
    fn serialize_and_deserialize_ek() {
        let mut rng = thread_rng();
        let params = BfvParameters::default(15, 1 << 8);

        let sk = SecretKey::random(params.degree, params.hw, &mut rng);

        let ek = EvaluationKey::new(
            &params,
            &sk,
            &[0, 1, 2, 3, 4, 5],
            &[0, 1, 2, 3, 4, 5],
            &[1, 2, 3, -1, -2, -3],
            &mut rng,
        );

        let ek_proto = proto::EvaluationKey::try_from_with_parameters(&ek, &params);
        let ek_back = EvaluationKey::try_from_with_parameters(&ek_proto, &params);

        assert_eq!(ek, ek_back);
    }
}
