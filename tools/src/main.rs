use std::process::Command;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tools::check_rustup()?;
    let version = tools::compute_nightly_name();
    println!("Installing version {}", version);
    tools::install_nightly(&version)?;
    let toollchain_path =
        tools::find_toolchain_dir(&version).expect("Unable to find suitable toolchain");
    println!("Linking toolchain");
    tools::link_toolchain("rlsl", &toollchain_path)?;
    println!("working");

    Ok(())
}
