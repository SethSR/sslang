
use miette::{IntoDiagnostic, LabeledSpan, WrapErr};

use crate::tokens::{Token, TokenType};

pub(crate) type TypedIdent = (String, ValueType);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ValueType {
	U8, U16, U32,
	S8, S16, S32,
	F16(u8), // 16-bit fixed point with <u8> integer bits
	F32(u8), // 32-bit fixed point with <u8> integer bits
	TypeName(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
	Add,
	AndB,
	AndL,
	CmpEq,
	CmpGE,
	CmpGT,
	CmpLE,
	CmpLT,
	CmpNE,
	Deref,
	Div,
	DivMod,
	FieldAccess,
	LRotate,
	LShift,
	Mod,
	Mul,
	Neg,
	Not,
	OrB,
	OrL,
	Ref,
	RRotate,
	RShift,
	Sub,
	XorB,
	XorL,
}

#[derive(Debug, Clone, PartialEq)]
enum S<'a> {
	Atom(Token<'a>),
	Cons(Token<'a>, Vec<S<'a>>),
}

use std::fmt;

impl fmt::Display for S<'_> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		match self {
			S::Atom(token) => write!(fmt, "{token}"),
			S::Cons(head,rest) => {
				let out = rest.iter().map(|i| format!("{i}")).collect::<Vec<String>>().join(" ");
				write!(fmt, "({head} {out})")
			}
		}
	}
}

pub fn eval<'a>(
	input: Vec<Token<'a>>,
) -> miette::Result<Vec<Stmt<'a>>> {
	if input.len() == 0 {
		miette::bail!("Empty input");
	}

	let mut parser = Parser {
		input: &input,
		index: 0,
	};

	program(&mut parser)
}

macro_rules! error {
	(eof, $parser:expr, $msg:expr) => {
		Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(
					$parser.peek(-1).range.clone(),
					"after here")
			],
			"Expected {}, Found EoF", stringify!($msg)
		}.with_source_code($parser.peek(-1).src.to_owned()))
	};
	($tt:expr, $parser:expr, $msg:expr) => {
		Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(
					$parser.peek(0).range.clone(),
					"here")
			],
			"Expected {}, Found {:?}", stringify!($msg), $tt,
		}.with_source_code($parser.peek(0).src.to_owned()))
	};
}

fn num(
	parser: &mut Parser,
) -> miette::Result<u64> {
	match parser.peek(0).tt {
		TokenType::Number => {
			parser.index += 1;
			Ok(float_to_fixed(parser.peek(-1).to_string()
				.parse::<f64>()
				.into_diagnostic()
				.wrap_err("lexer should not allow invalid floating-point values")?))
		}
		TokenType::EOF => error!(eof, parser, "Number"),
		tt => error!(tt, parser, "Number"),
	}
}

fn float_to_fixed(n: f64) -> u64 {
	(n * (1u64 << 32) as f64) as u64
}

#[test]
fn convert_float_to_fixed() {
	assert_eq!(0, float_to_fixed(0.0));
	assert_eq!(0x00000000_80000000, float_to_fixed(0.5));
	assert_eq!(0x00000005_4CCCCCCC, float_to_fixed(5.3));
	assert_eq!(0x00000006_66666666, float_to_fixed(6.4));
}

fn ident(
	parser: &mut Parser,
) -> miette::Result<String> {
	match parser.peek(0).tt {
		TokenType::Ident => {
			parser.index += 1;
			Ok(parser.peek(-1).to_string())
		}
		TokenType::EOF => error!(eof, parser, "Identifier"),
		tt => error!(tt, parser, "Identifier"),
	}
}

fn value_type(
	parser: &mut Parser,
) -> miette::Result<ValueType> {
	let token = parser.input[parser.index].clone();
	let out = match token.tt {
		TokenType::U8  => ValueType::U8,
		TokenType::U16 => ValueType::U16,
		TokenType::U32 => ValueType::U32,
		TokenType::S8  => ValueType::S8,
		TokenType::S16 => ValueType::S16,
		TokenType::S32 => ValueType::S32,
		TokenType::F16 => {
			let token_str = token.to_string();
			let Some(bit_spec) = token_str.strip_prefix("fw") else {
				miette::bail!("Parsed a Fixed-point Word that doesn't start with 'fw'");
			};

			let Ok(bits) = bit_spec.parse::<u8>()
			else {
				miette::bail!("Unable to parse {token} into Fixed-point Word type");
			};

			if bits > 16 {
				miette::bail!("{}",
					expected("Bit specifier between 0..=16"));
			}
			ValueType::F16(bits)
		}
		TokenType::F32 => {
			let token_str = token.to_string();
			let Some(bit_spec) = token_str.strip_prefix("fl") else {
				miette::bail!("Parsed a Fixed-point Long that doesn't start with 'fl'");
			};

			let Ok(bits) = bit_spec.parse::<u8>() else {
				miette::bail!("Unable to parse {token} into Fixed-point Long type");
			};

			if bits > 32 {
				miette::bail!("{}",
					expected("Bit specifier between 0..=32"));
			}
			ValueType::F32(bits)
		}
		TokenType::Ident => ValueType::TypeName(token.to_string().to_owned()),
		_ => miette::bail!("Expected Value Type"),
	};
	parser.index += 1;
	Ok(out)
}

fn ident_typed_opt(
	parser: &mut Parser,
) -> miette::Result<(String,Option<ValueType>)> {
	let id = ident(parser)?;
	let val_type = value_type(parser).ok();
	Ok((id, val_type))
}

fn match_token(
	parser: &mut Parser,
	tt: TokenType,
) -> miette::Result<()> {
	match parser.peek(0).tt {
		t if t != tt => if t == TokenType::EOF {
			error!(t, parser, tt)
		} else {
			error!(eof, parser, tt)
		}
		_ => Ok(parser.index += 1),
	}
}

fn ident_typed(
	parser: &mut Parser,
) -> miette::Result<TypedIdent> {
	let id = ident(parser)?;
	let val_type = value_type(parser)?;
	Ok((id, val_type))
}

struct Parser<'a,'b>
where 'a: 'b
{
	input: &'b [Token<'a>],
	index: usize,
}

impl<'a,'b> Parser<'a,'b> {
	fn peek(&self, offset: isize) -> &Token<'a> {
		&self.input[self.index.saturating_add_signed(offset)]
	}
}

enum Expr {}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Stmt<'a> {
	Rec {
		name: String,
		fields: Vec<(String, ValueType)>,
	},
	Fun {
		name: String,
		params: Vec<(String, ValueType)>,
		body: Vec<Stmt<'a>>,
		output: Option<S<'a>>,
	},
	Var {
		name: String,
		vtype: Option<ValueType>,
		body: S<'a>,
	},
}

/// bin_op := expr '+' expr
///         | expr '-' expr
///         | expr '*' expr
///         | expr '/' expr
///         | expr '%' expr
///         | expr '&' expr
///         | expr '&&' expr
///         | expr '|' expr
///         | expr '||' expr
///         | expr '^' expr
///         | expr '<' expr
///         | expr '<<' expr
///         | expr '<<<' expr
///         | expr '<=' expr
///         | expr '>' expr
///         | expr '>>' expr
///         | expr '>>>' expr
///         | expr '>=' expr
///         | expr '==' expr
///         | expr '<>' expr

/// un_op := '!' expr
///        | '~' expr
///        | '$' expr
///        | '@' expr
///        | '+' expr
///        | '-' expr

/// args := (literal (',' args)?)?

/// ident_or_call := ident ('(' args ')')?

/// literal := ident_or_call | num

fn prefix_binding_power(tt: TokenType) -> ((),u8) {
	use TokenType as TT;

	match tt {
		TT::Plus | TT::Minus => ((),11),
		TT::Dollar | TT::At => ((),13),
		TT::Bang => ((),15),
		_ => panic!("bad unary op"),
	}
}

fn infix_binding_power(tt: TokenType) -> (u8,u8) {
	use TokenType as TT;

	match tt {
		TT::Amp2 | TT::Bar2 | TT::Carrot2 => (1,2),
		TT::Amp1 | TT::Bar1 | TT::Carrot1 => (3,4),
		TT::Eq2 | TT::BangEq |
		TT::RArrow1 | TT::RArrEq |
		TT::LArrow1 | TT::LArrEq => (5,6),
		TT::Plus | TT::Minus => (7,8),
		TT::Star | TT::Slash | TT::Percent | TT::SlashPer |
		TT::LArrow2 | TT::RArrow2 => (9,10),
		TT::Dot => (18,17),
		_ => panic!("bad infix op"),
	}
}

/// expr := literal
///       | bin_op
///       | un_op
fn expr<'a>(
	parser: &mut Parser<'a,'_>,
	min_bp: u8,
) -> miette::Result<S<'a>> {
	use TokenType as TT;

	let left_token = parser.peek(0).clone();
	let mut lhs = match left_token.tt {
		TT::Ident |
		TT::Number => S::Atom(left_token),
		tt @ TT::Plus |
		tt @ TT::Minus |
		tt @ TT::Dollar |
		tt @ TT::At |
		tt @ TT::Bang => {
			let ((),r_bp) = prefix_binding_power(tt);
			let rhs = expr(parser, r_bp)?;
			S::Cons(left_token, vec![rhs])
		}
		TT::EOF => return error!(eof, parser,
			"Identifier, Function Call, or Literal"),
		tt => return error!(tt, parser,
			"Identifier, Function Call, or Literal"),
	};
	parser.index += 1;

	loop {
		let op_token = parser.peek(0).clone();
		if matches!(op_token.tt,
			TT::Ident | TT::Number |
			TT::If | TT::Else | TT::While |
			TT::Fun | TT::Rec | TT::Var |
			TT::U8 | TT::U16 | TT::U32 |
			TT::S8 | TT::S16 | TT::S32 |
			TT::F16 | TT::F32 |
			TT::Colon | TT::CParen | TT::OParen |
			TT::EOF) {
			break;
		};
		let (l_bp, r_bp) = infix_binding_power(op_token.tt);

		if l_bp < min_bp {
			break;
		}

		parser.index += 1;
		let rhs = expr(parser, r_bp)?;
		lhs = S::Cons(op_token, vec![lhs, rhs]);
	}

	Ok(lhs)
}

/// params := ( ident ':' value_type )*
fn params(
	parser: &mut Parser,
) -> miette::Result<Vec<(String, ValueType)>> {
	let mut out = Vec::new();
	while parser.peek(0).tt != TokenType::CParen {
		let id = ident(parser)?;
		match_token(parser, TokenType::Colon)?;
		let vt = value_type(parser)?;
		out.push((id,vt));
	}
	Ok(out)
}

/// rec := 'rec' ident '(' params ')'
fn rec<'a>(
	parser: &mut Parser,
) -> miette::Result<Stmt<'a>> {
	match_token(parser, TokenType::Rec)?;
	let name = ident(parser)?;
	match_token(parser, TokenType::OParen)?;
	let fields = params(parser)?;
	match_token(parser, TokenType::CParen)?;
	Ok(Stmt::Rec { name, fields })
}

/// fun := 'fun' ident '(' params ')' '(' statement* expr? ')'
fn fun<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt<'a>> {
	match_token(parser, TokenType::Fun)?;
	let name = ident(parser)?;
	match_token(parser, TokenType::OParen)?;
	let params = params(parser)?;
	match_token(parser, TokenType::CParen)?;
	match_token(parser, TokenType::OParen)?;
	let mut body = Vec::new();
	while let Ok(stmt) = statement(parser) {
		body.push(stmt);
	}
	let output = expr(parser, 0).ok();
	match_token(parser, TokenType::CParen)?;
	Ok(Stmt::Fun { name, params, body, output })
}

/// var := 'var' ident (':' value_type)? '=' expr
fn var<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt<'a>> {
	match_token(parser, TokenType::Var)?;
	let name = ident(parser)?;
	let vtype = match_token(parser, TokenType::Colon)
		.and_then(|_| value_type(parser))
		.ok();
	match_token(parser, TokenType::Eq1)?;
	let body = expr(parser, 0)?;
	Ok(Stmt::Var { name, vtype, body })
}

/// statement := rec | fun | var
fn statement<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt<'a>> {
	match parser.peek(0).tt {
		TokenType::Rec => rec(parser),
		TokenType::Fun => fun(parser),
		TokenType::Var => var(parser),
		_ => Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(parser.peek(0).range.clone(), "here"),
			],
			"Expected Statement"
		}.with_source_code(parser.peek(0).src.to_owned())),
	}
}

/// program := statement*
fn program<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Vec<Stmt<'a>>> {
	let mut program = Vec::default();
	while parser.peek(0).tt != TokenType::EOF {
		program.push(statement(parser)?);
	}
	Ok(program)
}

fn expected(s: &str) -> String {
	format!("{s} Expected")
}

#[cfg(test)]
mod test {
	use crate::tokens::{Token, TokenType as TT};
	use crate::lexer;
	use crate::parser::{self, S, Stmt, ValueType as VT};

	fn atom(tt: TT, s: &str) -> S {
		S::Atom(Token::new(tt, s, 0..s.len()))
	}

	fn var<'a>(
		name: &'_ str,
		vtype: Option<VT>,
		body: S<'a>,
	) -> Stmt<'a> {
		Stmt::Var {
			name: name.to_string(),
			vtype,
			body,
		}
	}

	fn fun<'a>(
		name: &'_ str,
		params: Vec<(String, VT)>,
		body: Vec<Stmt<'a>>,
		output: Option<S<'a>>,
	) -> Stmt<'a> {
		Stmt::Fun {
			name: name.to_string(),
			params,
			body,
			output,
		}
	}

	fn rec<'a>(
		name: &'_ str,
		fields: Vec<(String, VT)>,
	) -> Stmt<'a> {
		Stmt::Rec {
			name: name.to_string(),
			fields,
		}
	}

	fn parse_test(
		input: &str,
		stmts: &[Stmt],
	) -> miette::Result<()> {
		let ast = crate::parser::eval(crate::lexer::eval(input)?)?;
		assert_eq!(ast, stmts.to_vec());
		Ok(())
	}

	#[test]
	fn empty_input() -> miette::Result<()> {
		parse_test("", &[])
	}

	#[test]
	fn simple_var_stmt() -> miette::Result<()> {
		parse_test("var a = 0", &[
			var("a", None, atom(TT::Number, "0")),
		])
	}

	#[test]
	fn var_with_vtype() -> miette::Result<()> {
		parse_test("var a: u8 = 0", &[
			var("a", Some(VT::U8),
				atom(TT::Number, "0")),
		])
	}

	#[test]
	fn simple_fun_stmt() -> miette::Result<()> {
		parse_test("fun a () ()", &[
			fun("a", vec![], vec![], None),
		])
	}

	#[test]
	fn simple_rec_stmt() -> miette::Result<()> {
		parse_test("rec a ()", &[
			rec("a", vec![]),
		])
	}
}

/*
#[test]
fn let_expressions_bind_value_types(
) -> miette::Result<()> {
	let input = "let (a u8) (3)";
	let ast = eval(input, crate::lexer::eval(input)?)?;
	assert_eq!(ast, vec![
		S::Expr(Token::new(TokenType::Let, 0..3)),
		S::List(vec![
			S::Expr(Token::new(TokenType::Ident, 5..6)),
			S::Expr(Token::new(TokenType::U8, 7..9)),
		]),
		S::Expr(Token::new(TokenType::Num, 12..13)),
	]);
	Ok(())
}

#[test]
fn field_accessor_in_call_position_captures_all_accesses(
) -> miette::Result<()> {
	let input = "let (a) (b.c.d)";
	let ast = eval(input, crate::lexer::eval(input)?)?;
	assert_eq!(ast.len(), 3);
	assert_eq!(ast, vec![
		S::Expr(Token::new(TokenType::Let, 0..3)),
		S::Expr(Token::new(TokenType::Ident, 5..6)),
		S::List(vec![
			S::Expr(Token::new(TokenType::Ident, 9..10)),
			S::Expr(Token::new(TokenType::Dot, 10..11)),
			S::Expr(Token::new(TokenType::Ident, 11..12)),
			S::Expr(Token::new(TokenType::Dot, 12..13)),
			S::Expr(Token::new(TokenType::Ident, 13..14)),
		]),
	]);
	Ok(())
}
*/

