/// Try convert from Self with poly context.
/// TODO: Implement Error
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
