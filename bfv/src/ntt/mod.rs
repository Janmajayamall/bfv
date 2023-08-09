// use conrete-ntt as default
#[cfg(not(feature = "hexl-ntt"))]
mod concrete;
#[cfg(not(feature = "hexl-ntt"))]
pub use concrete::NttOperator;

#[cfg(feature = "hexl-ntt")]
mod hexl;
#[cfg(feature = "hexl-ntt")]
pub use hexl::NttOperator;
