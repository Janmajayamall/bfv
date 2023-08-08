use std::fmt::Debug;

pub trait Ntt: Sync + Send + PartialEq + Clone + Debug {
    fn new(degree: usize, prime: u64) -> Self;

    fn forward(&self, a: &mut [u64]);

    fn forward_lazy(&self, a: &mut [u64]);

    fn backward(&self, a: &mut [u64]);
}

pub trait TryFromWithPolyContext<'a>: Sized {
    type Value;
    type PolyContext;

    fn try_from_with_context(value: &Self::Value, poly_ctx: &'a Self::PolyContext) -> Self;
}

pub trait TryFromWithParameters: Sized {
    type Value;
    type Parameters;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self;
}
