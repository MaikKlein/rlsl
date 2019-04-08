use std::process::Command;
// The whole point of this wrapper is to call `rlsl_internal` in the correct enviroment. We use
// rustup to force the correct toolchain.
fn main() {
    let args: Vec<String> = std::env::args_os()
        .map(|arg| arg.to_str().unwrap().into())
        .collect();
    let is_build_script = args
        .iter()
        .find(|arg| arg.as_str() == "build_script_build")
        .is_some();
    // We do not call into rlsl when we want to compile a build script. This should run on the CPU,
    // we still want to make sure that we use the correct version of rustc.
    if is_build_script {
        Command::new("rustup")
            .arg("run")
            .arg("rlsl")
            .arg("rustc")
            .args(&args[1..])
            .status()
            .expect("Unable to start rustc");
        return;
    }
    Command::new("rustup")
        .arg("run")
        .arg("rlsl")
        .arg("rlsl_internal")
        .args(&args[1..])
        .status()
        .expect("Unable to start rlsl");
}
