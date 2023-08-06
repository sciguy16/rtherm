use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
pub struct Args {
    #[clap(long, help = "Path to video device")]
    pub device: PathBuf,
}

impl Args {
    pub fn parse() -> Self {
        Parser::parse()
    }
}
