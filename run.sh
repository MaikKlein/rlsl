cargo build --release && ./target/release/rlsl --crate-name rlsl_example rlsl-example/src/main.rs -A warnings && spirv-cross shader.spv && spirv-dis shader.spv && spirv-val shader.spv
