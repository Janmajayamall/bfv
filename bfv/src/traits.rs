/// Try convert from Self with poly context.
/// TODO: Implement Error
pub trait TryFromWithPolyContext<'a>: Sized {
    type Poly;
    type PolyContext;

    fn try_from_with_context(poly: &Self::Poly, poly_ctx: &'a Self::PolyContext) -> Self;
}

pub trait TryFromWithParameters: Sized {
    type Value;
    type Parameters;

    fn try_from_with_parameters(value: &Self::Value, parameters: &Self::Parameters) -> Self;
}
