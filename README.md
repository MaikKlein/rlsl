# RLSL - Rust Like Shading Langauge (Highly experimental)

Experimental rust compiler from `mir` -> `spirv`.


Compile Rust in `compiler/rust` and create a new toolchain
``` 
rustup toolchain link rlsl_rust compiler/rust/build/x86_64-unknown-linux-gnu/stage2/
```
and override the current dir
``` 
rustup override set rlsl_rust
```
