cargo run --bin rlsl --release -- rlsl-example/src/main.rs --extern core=libcore.rlib --extern std=libstd.rlib -L . -A warnings -Z mir-opt-level=1 -Z always-encode-mir
