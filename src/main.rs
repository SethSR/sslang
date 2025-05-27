
use miette::IntoDiagnostic;
use tracing::{info,debug};

mod tokens;
mod lexer;
mod parser;

const TEST_INPUT: &'static str = "
rec vec {
	x:fd,
	y:fd,
}

rec quat {
	s:fd,
	v:fd,
}

fn main() {
	var x:fd12 = 0.44
	var y:fd12 = 0.01

	var p = vec {x:x, y:y}
	var q = vec {x:1.5, y:2.6}

	fn vmul(a:vec, b:vec) -> quat {
		quat {
			s: a.x * b.x + a.y * b.y,
			v: a.x * b.y - b.x * a.y,
		}
	}

	vmul(p, q)
}";

use clap::Parser;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
	Lexer,
	Parser,
}

impl From<String> for Stage {
	fn from(s: String) -> Self {
		match s.to_lowercase().as_str() {
			"lexer" => Self::Lexer,
			"parser" => Self::Parser,
			_ => unimplemented!(),
		}
	}
}

#[derive(Parser)]
struct Options {
	#[arg(short,long)]
	debug: Vec<Stage>,

	#[arg(short,long,default_value_t=String::from("a.out"))]
	output_file: String,

	source_file: String,
}

fn main() -> miette::Result<()> {
	let mut options = Options::parse();
	options.debug.dedup();

	tracing_subscriber::fmt()
		.without_time()
		//.with_file(false)
		.with_max_level(tracing::Level::TRACE)
		.compact()
		.init();

	/*
	let in_file_name = options.source_file;

	let out_file_name = options.output_file;

	let source = std::fs::read_to_string(&in_file_name.trim())
		.into_diagnostic()?;
	*/

	let source = TEST_INPUT;

	info!("lexing");
	let tokens = lexer::eval(&source)?;
	debug!("Tokens: [{}]", tokens.iter()
		.map(|t| t.to_string())
		.collect::<Vec<_>>()
		.join(", "));
	info!("parsing");
	let (start, ast) = parser::eval(&source, tokens)?;
	debug!("Start ID: {start}");
	debug!("AST: {ast:?}");

	let output = format!("Start ID: {start}\nAST: {ast:?}");

	std::fs::write("test.out", output)
//	std::fs::write(&out_file_name, output)
		.into_diagnostic()
}

