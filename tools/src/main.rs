use std::env;
use std::fs;
use std::path::Path;
fn main() {
    env::var("SUDO").expect("Please run sudo");
    let home_dir = env::home_dir().expect("no home dir");
    let rlsl_path = home_dir.join(".rlsl");

    if !rlsl_path.exists() {
        fs::create_dir(rlsl_path).expect("unable to create .rlsl");
    }
}
