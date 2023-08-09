fn main() -> std::io::Result<()> {
    #[cfg(feature = "serialize")]
    prost_build::compile_protos(&["src/proto/bfv.proto"], &["src/proto"])?;
    Ok(())
}
