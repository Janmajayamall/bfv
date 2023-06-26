mod evaluation_key;
mod evaluator;
mod galois_key;
mod key_switching_key;
mod modulus;
mod nb_theory;
mod parameters;
mod plaintext;
mod poly;
mod relinearization_key;
mod secret_key;
mod utils;

pub use evaluation_key::*;
pub use evaluator::*;
pub use galois_key::*;
pub use key_switching_key::*;
pub use modulus::*;
pub use nb_theory::*;
pub use parameters::PolyType;
pub use plaintext::*;
pub use poly::{Poly, Representation, Substitution};
pub use relinearization_key::*;
pub use secret_key::*;
pub use utils::*;

#[cfg(not(feature = "hexl"))]
use fhe_math::zq::ntt::NttOperator;
#[cfg(feature = "hexl")]
use hexl_rs::NttOperator;

pub type BfvParameters = parameters::BfvParameters<NttOperator>;
pub type PolyContext<'a> = poly::PolyContext<'a, NttOperator>;
