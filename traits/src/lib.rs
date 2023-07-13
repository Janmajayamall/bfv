use std::fmt::Debug;

pub trait Ntt: Sync + Send + PartialEq + Clone + Debug {
    fn new(degree: usize, prime: u64) -> Self;

    fn forward(&self, a: &mut [u64]);

    fn forward_lazy(&self, a: &mut [u64]);

    fn backward(&self, a: &mut [u64]);
}

/// Try convert from Self with poly context.
/// TODO: Implement Error
pub trait TryFromWithPolyContext: Sized {
    type PolyContext;

    fn try_from_with_context(poly_ctx: Self::PolyContext) -> Self;
}
