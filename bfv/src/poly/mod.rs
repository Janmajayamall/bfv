use crate::{convert_from_bytes, convert_to_bytes, proto};
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

#[derive(Debug)]
pub struct Substitution {
    exponent: usize,
    power_bitrev: Box<[usize]>,
    bit_rev: Box<[usize]>,
    degree: usize,
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

    pub fn coefficients_ref(&self) -> &Array2<u64> {
        &self.coefficients
    }
}

impl Poly {
    fn to_proto(&self, poly_ctx: &crate::PolyContext<'_>) -> proto::Poly {
        let bytes = izip!(self.coefficients.outer_iter(), poly_ctx.iter_moduli_ops())
            .map(|(xi, modqi)| convert_to_bytes(xi.as_slice().unwrap(), modqi.modulus()))
            .collect_vec();
        let mut repr = proto::Representation::Unknown;
        if self.representation == Representation::Coefficient {
            repr = proto::Representation::Coefficient;
        } else if self.representation == Representation::Evaluation {
            repr = proto::Representation::Evaluation;
        }

        proto::Poly {
            coefficients: bytes,
            representation: repr.into(),
        }
    }

    fn from_proto_with_context(
        proto_poly: &proto::Poly,
        poly_ctx: &crate::PolyContext<'_>,
    ) -> Poly {
        let coefficients = izip!(proto_poly.coefficients.iter(), poly_ctx.iter_moduli_ops())
            .flat_map(|(xi, modqi)| {
                let values = convert_from_bytes(xi, modqi.modulus());
                assert!(values.len() == poly_ctx.degree());
                values
            })
            .collect_vec();
        let coefficients =
            Array2::from_shape_vec((poly_ctx.moduli_count(), poly_ctx.degree()), coefficients)
                .unwrap();

        let mut representation = Representation::Unknown;
        if proto_poly.representation == proto::Representation::Coefficient.into() {
            representation = Representation::Coefficient;
        } else if proto_poly.representation == proto::Representation::Evaluation.into() {
            representation = Representation::Evaluation;
        }

        Poly {
            coefficients,
            representation,
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
        let proto = poly.to_proto(&ctx);
        let bytes = proto.encode_to_vec();
        dbg!(bytes.len());
        let poly_back = Poly::from_proto_with_context(&proto, &ctx);

        assert_eq!(poly, poly_back);
    }
}
