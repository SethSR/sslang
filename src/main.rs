
use miette::IntoDiagnostic;
use tracing::info;

// mod checker;
mod context;
mod tokens;
// mod lexer;
// mod parser;
// mod reducer;

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

	let in_file_name = options.source_file;

	// let out_file_name = options.output_file;

	let source = std::fs::read_to_string(in_file_name.trim())
		.into_diagnostic()?;

	// info!("lexing");
	// let tokens = lexer::eval(source)?;
	// if options.debug.contains(&Stage::Lexer) {
	// 	let token_str = tokens.iter()
	// 		.map(|t| t.to_string())
	// 		.collect::<Vec<_>>()
	// 		.join(", ");
	// 	debug!("Tokens: [{token_str}]");
	// }

	info!("parsing");
	let out = parser::eval(&source);
	if options.debug.contains(&Stage::Parser) {
		// debug!("Start ID: {}", out.start);
		for node in &out {
			println!("{node}");
			// debug!("[{nx:>3}]: {node}");
		}
	}

	// info!("type-checking");
	// let mut out = checker::eval(out);
	// if options.debug.contains(&Stage::Checker) {
	// 	debug!("checked AST: {out:?}");
	// }

	// if options.release {
	// 	info!("reduction");
	// 	let ast = reducer::eval(&mut out.store, out.start);
	// 	if options.debug.contains(&Stage::Reducer) {
	// 		debug!("reduced AST: {ast:?}");
	// 	}
	// }

	let output = format!("AST: {out:?}");

	std::fs::write("test.out", output)
//	std::fs::write(&out_file_name, output)
		.into_diagnostic()
}

mod parser {

	use pest::Parser;
	use pest::iterators::Pairs;
	use pest_derive::Parser;

	#[derive(Parser)]
	#[grammar="sslang.pest"]
	struct SSLangParser;

	pub fn eval(source: &str) -> Vec<String> {
		let pairs = SSLangParser::parse(Rule::main, source)
			.unwrap_or_else(|e| panic!("{e}"));
		let mut out = vec![];
		output(pairs, &mut out);
		out
	}

	fn output(pairs: Pairs<Rule>, out: &mut Vec<String>) {
		for pair in pairs {
			match pair.as_rule() {
				Rule::EOI => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::WHITESPACE => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::COMMENT => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::main => output(pair.into_inner(), out),
				Rule::expr => output(pair.into_inner(), out),
				Rule::expr_var => output(pair.into_inner(), out),
				Rule::expr_assign => output(pair.into_inner(), out),
				Rule::k_var => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::k_rec => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::k_fun => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::k_if => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::k_else => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::k_while => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::k_true => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::k_false => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_andb => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_andl => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_ref => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_bang => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_ne => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_orb => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_orl => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_xorb => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_xorl => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_cbrace => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_cparen => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_colon => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_comma => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_deref => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_dot => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_equal => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_eq => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_lt => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_shl => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_rotl => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_le => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_dash => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_obrace => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_oparen => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_mod => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_plus => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_gt => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_shr => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_rotr => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_ge => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_return => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_semicolon => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_div => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_divmod => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::c_star => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::val_spec => output(pair.into_inner(), out),
				Rule::val_ret => output(pair.into_inner(), out),
				Rule::expr_rec => output(pair.into_inner(), out),
				Rule::expr_rec_init => output(pair.into_inner(), out),
				Rule::expr_fun => output(pair.into_inner(), out),
				Rule::expr_if => output(pair.into_inner(), out),
				Rule::expr_while => output(pair.into_inner(), out),
				Rule::expr_bool => output(pair.into_inner(), out),
				Rule::expr_bit => output(pair.into_inner(), out),
				Rule::expr_eq => output(pair.into_inner(), out),
				Rule::expr_cmp => output(pair.into_inner(), out),
				Rule::expr_add => output(pair.into_inner(), out),
				Rule::expr_mul => output(pair.into_inner(), out),
				Rule::expr_shift => output(pair.into_inner(), out),
				Rule::expr_rot => output(pair.into_inner(), out),
				Rule::unary => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::term => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::call => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::args => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::params => output(pair.into_inner(), out),
				Rule::param => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::field_inits => output(pair.into_inner(), out),
				Rule::field => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::val_type => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::block => output(pair.into_inner(), out),
				Rule::ident => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::integer => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
				Rule::fixed => out.push(format!("{:?} | {}", pair.as_rule(), pair.as_str())),
			}
		}
	}
}

