pub mod evaluator;
pub mod galois_key;
pub mod key_switching_key;
pub mod modulus;
pub mod nb_theory;
pub mod parameters;
pub mod plaintext;
pub mod poly;
pub mod relinearization_key;
pub mod secret_key;
pub mod utils;

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
