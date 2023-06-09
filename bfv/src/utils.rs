#[macro_export]
macro_rules! warn {
    ($con:expr, $($txt:tt)*) => {
        #[cfg(debug_assertions)]
        if $con {
            println!($($txt)*)
        }
    };
}
