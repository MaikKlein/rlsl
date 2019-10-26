fn main() -> Result<(), Box<dyn std::error::Error>> {
    tools::check_rustup()?;
    tools::check_xargo()?;

    println!("Creating .rlsl at home dir");
    tools::create_rlsl_dirs()?;

    let version = tools::compute_nightly_name();

    println!("Installing version {}", version);
    tools::install_nightly(&version)?;

    let toollchain_path =
        tools::find_toolchain_dir(&version).expect("Unable to find suitable toolchain");

    println!("Linking toolchain rlsl to {}", version);
    tools::link_toolchain("rlsl", &toollchain_path)?;

    println!("Adding component rust-src to toolchain {}", version);
    tools::install_toolchain_src(&version)?;

    println!("Building custom libstd/libcore");
    tools::build_custom_libstd();

    println!("Installing libstd/libcore");
    tools::install_custom_libstd();
    Ok(())
}
