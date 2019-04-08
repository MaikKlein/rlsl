use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug)]
pub enum Error {
    Rustup,
}

pub type ToolResult<T> = std::result::Result<T, Error>;

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
