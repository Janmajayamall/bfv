use crate::{convert_from_bytes, convert_to_bytes, proto, traits::TryFromWithPolyContext};
use fhe_math::zq::ntt::NttOperator;
use itertools::{izip, Itertools};
use ndarray::Array2;
pub mod poly_context;

pub use poly_context::PolyContext;

#[derive(Clone, PartialEq, Debug, Eq)]
pub enum Representation {
    Evaluation,
    Coefficient,
    Unknown,
}

#[derive(Debug, PartialEq)]
pub struct Substitution {
    pub(crate) exponent: usize,
    pub(crate) power_bitrev: Box<[usize]>,
    pub(crate) bit_rev: Box<[usize]>,
    pub(crate) degree: usize,
}

impl Substitution {
    /// Computes substitution map for polynomial degree
    ///
    /// exponent must be an odd integer not a multiple of 2 * degree.
    pub fn new(exponent: usize, degree: usize) -> Substitution {
        assert!(exponent & 1 == 1);
        let exponent = exponent % (2 * degree);
        let mask = degree - 1;
        let mut power = (exponent - 1) / 2;
        let power_bitrev = (0..degree)
            .map(|_| {
                let r = (power & mask).reverse_bits() >> (degree.leading_zeros() + 1);
                power += exponent;
                r
            })
            .collect_vec()
            .into_boxed_slice();

        let bit_rev = (0..degree)
            .into_iter()
            .map(|i| i.reverse_bits() >> (degree.leading_zeros() + 1))
            .collect_vec()
            .into_boxed_slice();

        Substitution {
            exponent,
            power_bitrev,
            degree,
            bit_rev,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Poly {
    pub(crate) coefficients: Array2<u64>,
    pub(crate) representation: Representation,
}

impl Poly {
    pub fn new(coefficients: Array2<u64>, representation: Representation) -> Poly {
        Poly {
            coefficients,
            representation,
        }
    }

    pub fn placeholder() -> Poly {
        Poly {
            coefficients: Array2::default((0, 0)),
            representation: Representation::Unknown,
        }
    }

    pub fn coefficients(&self) -> &Array2<u64> {
        &self.coefficients
    }

    pub fn coefficients_mut(&mut self) -> &mut Array2<u64> {
        &mut self.coefficients
    }

    pub fn representation(&self) -> &Representation {
        &self.representation
    }
}

impl<'a> TryFromWithPolyContext<'a> for Poly {
    type Value = proto::Poly;
    type PolyContext = crate::PolyContext<'a>;

    fn try_from_with_context(poly: &Self::Value, poly_ctx: &'a Self::PolyContext) -> Self {
        let coefficients = izip!(poly.coefficients.iter(), poly_ctx.iter_moduli_ops())
            .flat_map(|(xi, modqi)| {
                let values = convert_from_bytes(xi, modqi.modulus());
                assert!(values.len() == poly_ctx.degree());
                values
            })
            .collect_vec();
        let coefficients =
            Array2::from_shape_vec((poly_ctx.moduli_count(), poly_ctx.degree()), coefficients)
                .unwrap();

        Poly {
            coefficients,
            representation: Representation::Coefficient,
        }
    }
}

impl<'a> TryFromWithPolyContext<'a> for proto::Poly {
    type Value = Poly;
    type PolyContext = crate::PolyContext<'a>;

    fn try_from_with_context(poly: &Self::Value, poly_ctx: &'a Self::PolyContext) -> Self {
        assert!(poly.representation == Representation::Coefficient);

        let bytes = izip!(poly.coefficients.outer_iter(), poly_ctx.iter_moduli_ops())
            .map(|(xi, modqi)| convert_to_bytes(xi.as_slice().unwrap(), modqi.modulus()))
            .collect_vec();

        proto::Poly {
            coefficients: bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BfvParameters;
    use prost::Message;
    use rand::thread_rng;

    #[test]
    fn serialization_and_deserialization() {
        let params = BfvParameters::default(3, 1 << 15);
        let ctx = params.poly_ctx(&crate::PolyType::Q, 0);

        let mut rng = thread_rng();
        let poly = ctx.random(Representation::Coefficient, &mut rng);
        let proto = proto::Poly::try_from_with_context(&poly, &ctx);
        let bytes = proto.encode_to_vec();
        dbg!(bytes.len());
        let poly_back = Poly::try_from_with_context(&proto, &ctx);

        assert_eq!(poly, poly_back);
    }
}
