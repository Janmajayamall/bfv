#[cfg(not(feature = "hexl-ntt"))]
mod default;
#[cfg(not(feature = "hexl-ntt"))]
pub use default::NttOperator;

#[cfg(feature = "hexl-ntt")]
mod hexl;
#[cfg(feature = "hexl-ntt")]
pub use hexl::NttOperator;
