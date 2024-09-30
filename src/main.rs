
use miette::IntoDiagnostic;
use tracing::{info,debug};

mod tokens;
mod lexer;
mod parser;

fn main() -> miette::Result<()> {
	tracing_subscriber::fmt::init();

	let mut args = std::env::args();
	args.next();

	let file_name = args.next()
		.expect("missing source file");

	let file = std::fs::read_to_string(&file_name.trim())
		.into_diagnostic()?;

	info!("lexing");
	let tokens = lexer::eval(&file)?;
	debug!("[{}]", tokens.iter()
		.map(|t| t.to_string())
		.collect::<Vec<_>>()
		.join(" "));
	info!("parsing");
	let ast = parser::eval(&file, tokens)?;
	debug!("{ast:?}");

	Ok(())
}

