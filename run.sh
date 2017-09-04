cargo build --release && ./target/release/rlsl --crate-name rlsl_example rlsl-example/src/main.rs -A warnings && spirv-val shader.spv && spirv-cross shader.spv && spirv-dis shader.spv
