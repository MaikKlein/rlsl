extern crate ash;
extern crate clap;
extern crate image;
extern crate quad;
#[macro_use]
extern crate structopt;
use quad::*;
use std::fmt::Display;
use std::fmt::{Debug, Error, Formatter};
use std::path::{PathBuf};
use std::str::FromStr;
use structopt::StructOpt;

#[derive(Debug, Copy, Clone)]
pub enum ShaderCompiler {
    Rlsl,
    Glsl,
}
#[derive(Debug, Copy, Clone)]
pub struct ParseErrorShaderCompiler;
impl FromStr for ShaderCompiler {
    type Err = ParseErrorShaderCompiler;
    fn from_str(s: &str) -> Result<ShaderCompiler, Self::Err> {
        match s {
            "rlsl" => Ok(ShaderCompiler::Rlsl),
            "glsl" => Ok(ShaderCompiler::Glsl),
            _ => Err(ParseErrorShaderCompiler),
        }
    }
}

impl Display for ParseErrorShaderCompiler {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        Debug::fmt(self, f)
    }
}

#[derive(StructOpt, Debug)]
#[structopt(name = "quad")]
pub struct Opt {
    #[structopt(short = "c", long = "compiler", default_value = "rlsl", parse(try_from_str))]
    compiler: ShaderCompiler,
    #[structopt(subcommand)]
    command: Command,
}

#[derive(StructOpt, Debug)]
enum Command {
    #[structopt(name = "single")]
    Single { file: String },
    #[structopt(name = "compile")]
    Compile,
    #[structopt(name = "compute")]
    Compute,
}

impl Opt {
    pub fn get_shader_path(&self) -> PathBuf {
        match self.compiler {
            ShaderCompiler::Rlsl => PathBuf::from("./../.shaders/"),
            ShaderCompiler::Glsl => PathBuf::from("./../issues/.shaders-glsl/"),
        }

    }
    pub fn get_entry_names(&self) -> (&str, &str) {
        match self.compiler {
            ShaderCompiler::Rlsl => ("vertex", "fragment"),
            ShaderCompiler::Glsl => ("main", "main"),
        }
    }
}

fn main() {
    let opt = Opt::from_args();
    let app = Opt::clap();
    //app.gen_completions("myapp", Shell::Bash, "");
    let base_path = opt.get_shader_path();
    match opt.command {
        Command::Single { ref file } => {
            let mut quad = Quad::new();
            let vert_path = base_path.join("vertex.spv");
            let frag_path = base_path.join(&file).with_extension("spv");
            println!("{:#?}", frag_path.canonicalize());
            let (vert_name, frag_name) = opt.get_entry_names();
            quad.render((vert_name, &vert_path), (frag_name, &frag_path));
        }
        Command::Compute => {
            //compute::compute();
        }
        Command::Compile => {
            //quad.compile_all(&base_path);
        }
    }
}
