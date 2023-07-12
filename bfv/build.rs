fn main() -> std::io::Result<()> {
    prost_build::compile_protos(&["src/proto/bfv.proto"], &["src/proto"])?;
    Ok(())
}
