use anyhow::Context;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug)]
pub enum Error {
    Rustup,
}

pub type ToolResult<T> = std::result::Result<T, anyhow::Error>;

impl std::error::Error for Error {}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Rustup => write!(f, "Unable to find rustup, please install it!"),
        }
    }
}

pub fn check_rustc_version(hash: &str) -> bool {
    let output = Command::new("rustc").arg("--version").output().unwrap();
    let output = std::str::from_utf8(&output.stdout).unwrap();
    println!("{}", output);
    output.contains(hash)
}

pub fn compute_nightly_name() -> String {
    let date = include_str!("../../version.in");
    format!("nightly-{}", date).replace("\n", "")
}

pub fn install_nightly(version: &str) -> ToolResult<()> {
    Command::new("rustup")
        .arg("install")
        .arg(version)
        .status()
        .map_err(|_| Error::Rustup)?;
    Ok(())
}
pub fn check_rustup() -> ToolResult<()> {
    Command::new("rustup").output().map_err(|_| Error::Rustup)?;
    Ok(())
}

pub fn check_xargo() -> ToolResult<()> {
    Command::new("xargo").output().map_err(|_| Error::Rustup)?;
    Ok(())
}

pub fn find_toolchain_dir(version: &str) -> Option<PathBuf> {
    PathBuf::from(dirs::home_dir().unwrap())
        .join(".rustup/toolchains")
        .read_dir()
        .ok()?
        .filter_map(|dir| {
            let dir = dir.ok()?;
            let name = dir.file_name().into_string().ok()?;
            if name.matches(version).count() > 0 {
                Some(dir.path())
            } else {
                None
            }
        })
        .nth(0)
}
pub fn link_toolchain(name: &str, path: &Path) -> ToolResult<()> {
    Command::new("rustup")
        .arg("toolchain")
        .arg("link")
        .arg(name)
        .arg(format!("{}", path.display()))
        .status()
        .map_err(|_| Error::Rustup)?;
    Ok(())
}

pub fn install_toolchain_src(version: &str) -> ToolResult<()> {
    Command::new("rustup")
        .arg("component")
        .arg("add")
        .arg("rust-src")
        .arg("--toolchain")
        .arg(format!("{}", version))
        .status()
        .map_err(|_| Error::Rustup)?;
    Ok(())
}

pub fn build_custom_libstd() -> ToolResult<()> {
    let src_path = PathBuf::from(dirs::home_dir().unwrap())
        .join(".rustup/toolchains/rlsl/lib/rustlib/src/rust/src");

    Command::new("rustup")
        .env("XARGO_RUST_SRC", format!("{}", src_path.display()))
        .env("RUSTFLAGS", "-Z always-encode-mir")
        .arg("run")
        .arg("rlsl")
        .arg("xargo")
        .arg("build")
        .arg("--release")
        .arg("--manifest-path")
        .arg("libstd/Cargo.toml")
        .status()
        .map_err(|_| Error::Rustup)?;

    Ok(())
}

pub fn install_custom_libstd() -> ToolResult<()> {
    use std::fs;
    let triple = platforms::guess_current().unwrap().target_triple;
    let home_dir = dirs::home_dir().unwrap();
    let path = home_dir.join(format!(".xargo/HOST/lib/rustlib/{}/lib", triple));

    let find_rlib = |name: &str| -> Option<PathBuf> {
        path.read_dir().ok()?.find_map(|entry| {
            let file_path = entry.ok()?.path();
            if file_path.extension()? != "rlib" {
                return None;
            }

            let file_name = file_path.file_stem()?.to_str()?;
            if file_name.starts_with(name) {
                Some(file_path)
            } else {
                None
            }
        })
    };
    let libcore = find_rlib("libcore").context("Unable to find libcore")?;
    let libcompiler_builtins =
        find_rlib("libcompiler_builtins").context("Unable to find compiler builtins")?;

    fs::copy(
        "target/release/libstd.rlib",
        home_dir.join(".rlsl/lib/libstd.rlib"),
    )?;
    fs::copy(libcore, home_dir.join(".rlsl/lib/libcore.rlib"))?;
    fs::copy(
        libcompiler_builtins,
        home_dir.join(".rlsl/lib/libcompiler_builtins.rlib"),
    )?;

    Ok(())
}

pub fn create_rlsl_dirs() -> ToolResult<()> {
    use std::fs;
    let home_dir = dirs::home_dir().unwrap();

    fs::create_dir_all(home_dir.join(".rlsl"))?;
    fs::create_dir_all(home_dir.join(".rlsl/lib"))?;

    Ok(())
}
