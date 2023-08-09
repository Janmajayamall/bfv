fn main() -> std::io::Result<()> {
    #[cfg(feature = "seralize")]
    prost_build::compile_protos(&["src/proto/bfv.proto"], &["src/proto"])?;
    Ok(())
}
