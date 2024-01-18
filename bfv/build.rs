use prost_wkt_build::*;
use std::{env, path::PathBuf};

fn main() -> std::io::Result<()> {
    #[cfg(feature = "serialize")]
    {
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        let descriptor_file = out.join("descriptors.bin");
        let mut prost_build = prost_build::Config::new();
        prost_build
            .type_attribute(
                ".my.bfv",
                "#[derive(serde::Serialize, serde::Deserialize)] #[serde(default, rename_all=\"camelCase\")]",
            )
            // .type_attribute(
            //     ".my.messages.Foo",
            //     "#[derive(serde::Serialize, serde::Deserialize)] #[serde(default, rename_all=\"camelCase\")]",
            // )
            // .type_attribute(
            //     ".my.messages.Content",
            //     "#[derive(serde::Serialize, serde::Deserialize)] #[serde(rename_all=\"camelCase\")]",
            // )
            .extern_path(".google.protobuf.Any", "::prost_wkt_types::Any")
            .extern_path(".google.protobuf.Value", "::prost_wkt_types::Value")
            .file_descriptor_set_path(&descriptor_file)
            .compile_protos(&["src/proto/bfv.proto"], &["src/proto"])
            .unwrap();

        let descriptor_bytes = std::fs::read(descriptor_file).unwrap();
        let descriptor = FileDescriptorSet::decode(&descriptor_bytes[..]).unwrap();

        prost_wkt_build::add_serde(out, descriptor);
    }
    Ok(())
}
