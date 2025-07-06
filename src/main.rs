
use miette::IntoDiagnostic;
use tracing::{info,debug};

mod checker;
mod context;
mod tokens;
mod lexer;
mod parser;
mod parser2;
mod reducer;

const TEST_INPUT: &str = "
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

	if x < y {
		x = y
	} else {
		y = x
	}

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
	Checker,
	Reducer,
}

impl From<String> for Stage {
	fn from(s: String) -> Self {
		match s.to_lowercase().as_str() {
			"lexer" => Self::Lexer,
			"parser" => Self::Parser,
			"checker" => Self::Checker,
			"reducer" => Self::Reducer,
			_ => unimplemented!(),
		}
	}
}

#[derive(Parser)]
struct Options {
	#[arg(short,long)]
	debug: Vec<Stage>,

	#[arg(short,long,default_value_t=tracing::Level::INFO)]
	level: tracing::Level,

	#[arg(short,long,default_value_t=false)]
	release: bool,

	#[arg(short,long,default_value_t=String::from("a.out"))]
	output_file: String,

	source_file: String,
}

fn main() -> miette::Result<()> {
	let mut options = Options::parse();
	options.debug.dedup();

	tracing_subscriber::fmt()
		.compact()
		.with_max_level(options.level)
		.without_time()
		.init();

	/*
	let in_file_name = options.source_file;

	let out_file_name = options.output_file;

	let source = std::fs::read_to_string(&in_file_name.trim())
		.into_diagnostic()?;
	*/

	let source = TEST_INPUT;

	info!("lexing");
	let tokens = lexer::eval(source)?;
	if options.debug.contains(&Stage::Lexer) {
		let token_str = tokens.iter()
			.map(|t| t.to_string())
			.collect::<Vec<_>>()
			.join(", ");
		debug!("Tokens: [{token_str}]");
	}

	info!("parsing");
	let mut stepper = parser::stepper(source, &tokens);
	while let Some(result) = stepper.step() {
		match result {
			Ok(msg) => println!("[step] {msg}"),
			Err(e) => eprintln!("[erro] {e}"),
		}
	}
	let out = stepper.finish()
		.ok_or(miette::miette! {
			"Unable to retrieve program from Stepper"
		})?;
	if options.debug.contains(&Stage::Parser) {
		debug!("Start ID: {}", out.start);
		for (nx, node) in out.store.iter() {
			debug!("[{nx:>3}]: {node}");
		}
	}

	info!("type-checking");
	let mut out = checker::eval(out);
	if options.debug.contains(&Stage::Checker) {
		debug!("checked AST: {out:?}");
	}

	if options.release {
		info!("reduction");
		let ast = reducer::eval(&mut out.store, out.start);
		if options.debug.contains(&Stage::Reducer) {
			debug!("reduced AST: {ast:?}");
		}
	}

	let output = format!("AST: {out:?}");

	std::fs::write("test.out", output)
//	std::fs::write(&out_file_name, output)
		.into_diagnostic()
}

