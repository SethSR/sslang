
use miette::{IntoDiagnostic, LabeledSpan, WrapErr};

use crate::tokens::{Token, TokenType};

pub(crate) type TypedIdent = (String, ValueType);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ValueType {
	U8, U16, U32,
	S8, S16, S32,
	F16(u8), // 16-bit fixed point with <u8> integer bits
	F32(u8), // 32-bit fixed point with <u8> integer bits
	UDT(String), // User Defined Type
}

#[derive(Clone, PartialEq)]
pub(crate) enum S<'a> {
	Atom(Token<'a>),
	Cons(Token<'a>, Vec<S<'a>>),
}

use std::fmt;

impl fmt::Debug for S<'_> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		match self {
			S::Atom(token) => write!(fmt, "{token}"),
			S::Cons(head, rest) => {
				let out = rest.iter().map(|i| format!("{i}")).collect::<Vec<String>>().join(" ");
				write!(fmt, "({head} {out})")
			}
		}
	}
}

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
	source: &'a str,
	input: Vec<Token<'a>>,
) -> miette::Result<Vec<Stmt<'a>>> {
	if input.len() == 0 {
		miette::bail!("Empty input");
	}

	let mut parser = Parser {
		source,
		input: &input,
		index: 0,
	};

	program(&mut parser)
}

struct Parser<'a,'b>
where 'a: 'b
{
	input: &'b [Token<'a>],
	source: &'a str,
	index: usize,
}

impl<'a,'b> Parser<'a,'b> {
	fn peek(&self, offset: isize) -> &Token<'a> {
		&self.input[self.index.saturating_add_signed(offset)]
	}
}

macro_rules! error {
	(eof, $parser:expr, $msg:expr) => {
		Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(
					$parser.peek(-1).range(),
					"after here")
			],
			"Expected {:?}, Found EoF", $msg
		}.with_source_code($parser.source.to_owned()))
	};
	($tt:expr, $parser:expr, $msg:expr) => {
		Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(
					$parser.peek(0).range(),
					"here")
			],
			"Expected {:?}, Found {:?}", $msg, $tt,
		}.with_source_code($parser.source.to_owned()))
	};
}

fn num(
	parser: &mut Parser,
) -> miette::Result<u64> {
	match parser.peek(0).tt {
		TokenType::Number(s) => {
			parser.index += 1;
			Ok(float_to_fixed(s
				.chars()
				.filter(|c| *c != '_')
				.collect::<String>()
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
		TokenType::Ident(s) => {
			parser.index += 1;
			Ok(s.to_string())
		}
		TokenType::EOF => error!(eof, parser, "Identifier"),
		tt => error!(tt, parser, "Identifier"),
	}
}

fn parse_fixed_point(parser: &Parser, prefix: &str, max_bits: u8) -> miette::Result<u8> {
	let token = parser.peek(0);
	let token_str = token.to_string();
	let Some(bit_spec) = token_str.strip_prefix(prefix) else {
		return Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(token.range(), "here"),
			],
			"Parsed a Fixed-point type that doesn't start with '{prefix}'"
		}.with_source_code(parser.source.to_owned()));
	};

	let bits = if bit_spec.is_empty() {
		max_bits / 2
	} else if let Ok(bits) = bit_spec.parse::<u8>() {
		bits
	} else {
		return Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(token.range(), "here"),
			],
			"Unable to parse '{token}' into Fixed-point type"
		}.with_source_code(parser.source.to_owned()));
	};

	if bits > max_bits {
		return error!(token.tt, parser, format!("Bit specifier between 0..={max_bits}"));
	}
	Ok(bits)
}

fn value_type(
	parser: &mut Parser,
) -> miette::Result<ValueType> {
	let token = parser.peek(0).clone();
	let out = match token.tt {
		TokenType::U8  => ValueType::U8,
		TokenType::U16 => ValueType::U16,
		TokenType::U32 => ValueType::U32,
		TokenType::S8  => ValueType::S8,
		TokenType::S16 => ValueType::S16,
		TokenType::S32 => ValueType::S32,
		TokenType::F16(_) => parse_fixed_point(parser, "fw", 16).map(ValueType::F16)?,
		TokenType::F32(_) => parse_fixed_point(parser, "fl", 32).map(ValueType::F32)?,
		TokenType::Ident(_) => ValueType::UDT(token.to_string()),
		_ => return error!(token.tt, parser, "Value Type"),
	};
	parser.index += 1;
	Ok(out)
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Stmt<'a> {
	Rec {
		name: String,
		fields: Vec<TypedIdent>,
	},
	Fun {
		name: String,
		params: Vec<TypedIdent>,
		rtype: Option<ValueType>,
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
///         | expr '<|' expr
///         | expr '<=' expr
///         | expr '>' expr
///         | expr '>>' expr
///         | expr '|>' expr
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

fn prefix_binding_power(
	source: &str,
	token: Token,
) -> miette::Result<((),u8)> {
	use TokenType as TT;

	match token.tt {
		TT::Plus | TT::Minus => Ok(((),13)),
		TT::Dollar | TT::At => Ok(((),15)),
		TT::Bang => Ok(((),17)),
		tt => Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(token.range(), "here")
			],
			"Expected \"Unary Operator\", Found {tt:?}"
		}.with_source_code(source.to_owned())),
	}
}

fn infix_binding_power(tt: TokenType) -> Option<(u8,u8)> {
	use TokenType as TT;

	match tt {
		TT::Comma => Some((2,1)),
		TT::Amp2 | TT::Bar2 | TT::Carrot2 => Some((3,4)),
		TT::Amp1 | TT::Bar1 | TT::Carrot1 => Some((5,6)),
		TT::Eq2 | TT::BangEq |
		TT::RArrow1 | TT::RArrEq |
		TT::LArrow1 | TT::LArrEq => Some((7,8)),
		TT::Plus | TT::Minus => Some((9,10)),
		TT::Star | TT::Slash | TT::Percent | TT::SlashPer |
		TT::LArrow2 | TT::RArrow2 => Some((11,12)),
		TT::Dot => Some((20,19)),
		TT::OParen => Some((22,21)),

		TT::If | TT::Else |
		TT::Fun | TT::Rec | TT::Var |
		TT::While |

		TT::U8 | TT::U16 | TT::U32 |
		TT::S8 | TT::S16 | TT::S32 |
		TT::F16(_) | TT::F32(_) |

		TT::Ident(_) | TT::Number(_) |

		TT::At | TT::Bang |
		TT::Colon |
		TT::CParen |
		TT::Dollar |
		TT::Eq1 |
		TT::LArrBar | TT::RArrBar |
		TT::RetArrow |

		TT::EOF => None,
	}
}

fn expr<'a>(
	parser: &mut Parser<'a,'_>,
	min_bp: u8,
) -> miette::Result<S<'a>> {
	use TokenType as TT;

	let left_token = parser.peek(0).clone();
	let mut lhs = match left_token.tt {
		TT::Ident(_) |
		TT::Number(_) => S::Atom(left_token),
		TT::OParen => {
			parser.index += 1;
			let lhs = expr(parser, 0)?;
			if TT::CParen != parser.peek(0).tt {
				return error!(parser.peek(0).tt, parser, ")");
			}
			lhs
		}
		TT::Plus |
		TT::Minus |
		TT::Dollar |
		TT::At |
		TT::Bang => {
			let ((),r_bp) = prefix_binding_power(parser.source, left_token.clone())?;
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
			TT::Ident(_) | TT::Number(_) |
			TT::If | TT::Else | TT::While |
			TT::Fun | TT::Rec | TT::Var |
			TT::U8 | TT::U16 | TT::U32 |
			TT::S8 | TT::S16 | TT::S32 |
			TT::F16(_) | TT::F32(_) |
			TT::Colon | TT::CParen |
			TT::EOF) {
			break;
		};

		if op_token.tt == TT::OParen {
			parser.index += 1;
			if TT::CParen == parser.peek(0).tt {
				parser.index += 1;
				lhs = S::Cons(op_token, vec![lhs]);
				continue;
			}

			let rhs = expr(parser, 0)?;
			if TT::CParen != parser.peek(0).tt {
				return error!(parser.peek(0).tt, parser, ")");
			}
			parser.index += 1;
			lhs = S::Cons(op_token, vec![lhs,rhs]);
			continue;
		}

		if let Some((l_bp,r_bp)) = infix_binding_power(op_token.tt) {
			if l_bp < min_bp {
				break;
			}

			parser.index += 1;
			let rhs = expr(parser, r_bp)?;
			lhs = S::Cons(op_token, vec![lhs,rhs]);
			continue;
		}

		break;
	}

	Ok(lhs)
}

/// params := ( ident ':' value_type )*
fn params(
	parser: &mut Parser,
) -> miette::Result<Vec<TypedIdent>> {
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

/// fun := 'fun' ident '(' params ')' ('->' value_type)? '(' statement* expr? ')'
fn fun<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt<'a>> {
	match_token(parser, TokenType::Fun)?;
	let name = ident(parser)?;
	match_token(parser, TokenType::OParen)?;
	let params = params(parser)?;
	match_token(parser, TokenType::CParen)?;
	let rtype = match_token(parser, TokenType::RetArrow)
		.and_then(|_| value_type(parser))
		.ok();
	match_token(parser, TokenType::OParen)?;
	let mut body = Vec::new();
	while let Ok(stmt) = statement(parser) {
		body.push(stmt);
	}
	let output = expr(parser, 0).ok();
	match_token(parser, TokenType::CParen)?;
	Ok(Stmt::Fun { name, params, rtype, body, output })
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
		tt => error!(tt, parser, "Statement"),
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

#[cfg(test)]
mod test {
	use crate::tokens::TokenType as TT;
	use crate::parser::{S, Stmt, TypedIdent, ValueType as VT};

	fn atom(tt: TT) -> S {
		use crate::tokens::Token;
		S::Atom(Token::new(tt, 0))
	}

	fn num(s: &str) -> S {
		use crate::tokens::TokenType;
		atom(TokenType::Number(s))
	}

	fn ident(s: &str) -> S {
		use crate::tokens::TokenType;
		atom(TokenType::Ident(s))
	}

	fn cons<'a>(tt: TT<'a>, s: &[S<'a>]) -> S<'a> {
		S::Cons(
			crate::tokens::Token::new(tt, 0),
			s.into_iter().cloned().collect::<Vec<_>>(),
		)
	}

	fn expr_test(input: &str, s: S) -> miette::Result<()> {
		use crate::parser::{Parser, expr};
		use crate::lexer::eval;

		let mut parser = Parser {
			source: input,
			input: &eval(input)?,
			index: 0,
		};
		assert_eq!(expr(&mut parser, 0)?, s);
		Ok(())
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
		params: &[TypedIdent],
		rtype: Option<VT>,
		body: &[Stmt<'a>],
		output: Option<S<'a>>,
	) -> Stmt<'a> {
		Stmt::Fun {
			name: name.to_string(),
			params: params.to_vec(),
			rtype,
			body: body.to_vec(),
			output,
		}
	}

	fn rec<'a>(
		name: &'_ str,
		fields: &[TypedIdent],
	) -> Stmt<'a> {
		Stmt::Rec {
			name: name.to_string(),
			fields: fields.to_vec(),
		}
	}

	#[test]
	fn precedence() -> miette::Result<()> {
		expr_test("1 + 2 * 3", cons(TT::Plus, &[
			num("1"),
			cons(TT::Star, &[
				num("2"),
				num("3"),
			]),
		]))?;
		expr_test("1 * 2 + 3", cons(TT::Plus, &[
			cons(TT::Star, &[
				num("1"),
				num("2"),
			]),
			num("3"),
		]))
	}

	#[test]
	fn parentheses() -> miette::Result<()> {
		expr_test("1 * (2 + 3)", cons(TT::Star, &[
			num("1"),
			cons(TT::Plus, &[num("2"), num("3")]),
		]))
	}

	fn parse_test(
		input: &str,
		stmts: &[Stmt],
	) -> miette::Result<()> {
		use crate::parser;
		use crate::lexer;

		eprintln!("input: {input}");
		let tokens = lexer::eval(input)?;
		eprintln!("tokens: {tokens:?}");
		let ast = parser::eval(input, tokens)?;
		assert_eq!(ast, stmts.to_vec());
		Ok(())
	}

	#[test]
	fn empty_input() -> miette::Result<()> {
		parse_test("", &[])
	}

	#[test]
	fn var_stmt() -> miette::Result<()> {
		parse_test("var a = 0", &[
			var("a", None, num("0")),
		])
	}

	#[test]
	fn var_stmt_expr() -> miette::Result<()> {
		parse_test("var a = 3 * 2 + 1", &[
			var("a", None, cons(TT::Plus, &[
				cons(TT::Star, &[
					num("3"),
					num("2"),
				]),
				num("1"),
			])),
		])
	}

	#[test]
	fn var_stmt_vtype() -> miette::Result<()> {
		parse_test("var a: u8 = 0", &[
			var("a", Some(VT::U8), num("0")),
		])
	}

	#[test]
	fn var_stmt_udt_simple() -> miette::Result<()> {
		parse_test("var a = b", &[
			var("a", None, ident("b")),
		])
	}

	#[test]
	fn var_stmt_udt_fun_empty() -> miette::Result<()> {
		parse_test("var a = b()", &[
			var("a", None, cons(TT::OParen, &[ident("b")]))
		])
	}

	#[test]
	fn var_stmt_udt_fun_single() -> miette::Result<()> {
		parse_test("var a = b(c)", &[
			var("a", None, cons(TT::OParen, &[ident("b"), ident("c")]))
		])
	}

	#[test]
	fn var_stmt_udt_fun_multi() -> miette::Result<()> {
		parse_test("var a = b(c, d, e)", &[
			var("a", None, cons(TT::OParen, &[
				ident("b"),
				cons(TT::Comma, &[
					ident("c"),
					cons(TT::Comma, &[
						ident("d"),
						ident("e"),
					])
				]),
			]))
		])
	}

	#[test]
	fn fun_stmt() -> miette::Result<()> {
		parse_test("fun a () ()", &[
			fun("a", &[], None, &[], None),
		])
	}

	#[test]
	fn fun_stmt_params() -> miette::Result<()> {
		parse_test("fun a (b:u8 c:s16 d:fw6 e:fl10) ()", &[
			fun("a", &[
				("b".to_string(), VT::U8),
				("c".to_string(), VT::S16),
				("d".to_string(), VT::F16(6)),
				("e".to_string(), VT::F32(10)),
			], None, &[], None),
		])
	}

	#[test]
	fn fun_stmt_rtype_simple() -> miette::Result<()> {
		parse_test("fun a () -> u8 ()", &[
			fun("a", &[], Some(VT::U8), &[], None)
		])
	}

	#[test]
	fn fun_stmt_rtype_udt() -> miette::Result<()> {
		parse_test("fun a () -> b ()", &[
			fun("a", &[], Some(VT::UDT("b".to_string())), &[], None)
		])
	}

	#[test]
	fn fun_stmt_body() -> miette::Result<()> {
		parse_test("fun a () (
			var b = 1
			var c = 2
			b + c
		)", &[
			fun("a", &[], None, &[
				var("b", None, num("1")),
				var("c", None, num("2")),
			], Some(cons(TT::Plus, &[ident("b"), ident("c")])))
		])
	}

	#[test]
	fn rec_stmt() -> miette::Result<()> {
		parse_test("rec a ()", &[
			rec("a", &[]),
		])
	}

	#[test]
	fn field_access() -> miette::Result<()> {
		parse_test("var a = b.c.d", &[
			var("a", None, cons(TT::Dot, &[
				ident("b"),
				cons(TT::Dot, &[
					ident("c"),
					ident("d"),
				]),
			]))
		])
	}

	#[test]
	fn complex_test() -> miette::Result<()> {
		parse_test(
			"rec vec (
				x:fl
				y:fl
			)

			rec quat (
				s:fl
				v:fl
			)

			fun main() (
				var x:fl12 = 0.44
				var y:fl12 = 0.01

				var p = vec (x  , y  )
				var q = vec (1.5, 2.6)

				fun vmul(a:vec b:vec) -> quat (
					quat (
						a.x * b.x + a.y * b.y,
						a.x * b.y - b.x * a.y
					)
				)

				vmul(p, q)
			)", &[
				rec("vec", &[
					("x".to_string(), VT::F32(16)),
					("y".to_string(), VT::F32(16)),
				]),
				rec("quat", &[
					("s".to_string(), VT::F32(16)),
					("v".to_string(), VT::F32(16)),
				]),
				fun("main", &[], None, &[
					var("x", Some(VT::F32(12)), num("0.44")),
					var("y", Some(VT::F32(12)), num("0.01")),
					var("p", None, cons(TT::OParen, &[
						ident("vec"),
						cons(TT::Comma, &[
							ident("x"),
							ident("y"),
						]),
					])),
					var("q", None, cons(TT::OParen, &[
						ident("vec"),
						cons(TT::Comma, &[
							num("1.5"),
							num("2.6"),
						]),
					])),
					fun("vmul", &[
						("a".to_string(), VT::UDT("vec".to_string())),
						("b".to_string(), VT::UDT("vec".to_string())),
					], Some(VT::UDT("quat".to_string())), &[],
						Some(cons(TT::OParen, &[
							ident("quat"),
							cons(TT::Comma, &[
								cons(TT::Plus, &[
									cons(TT::Star, &[
										cons(TT::Dot, &[ident("a"), ident("x")]),
										cons(TT::Dot, &[ident("b"), ident("x")]),
									]),
									cons(TT::Star, &[
										cons(TT::Dot, &[ident("a"), ident("y")]),
										cons(TT::Dot, &[ident("b"), ident("y")]),
									]),
								]),
								cons(TT::Minus, &[
									cons(TT::Star, &[
										cons(TT::Dot, &[ident("a"), ident("x")]),
										cons(TT::Dot, &[ident("b"), ident("y")])
									]),
									cons(TT::Star, &[
										cons(TT::Dot, &[ident("b"), ident("x")]),
										cons(TT::Dot, &[ident("a"), ident("y")]),
									]),
								]),
							]),
						])),
					),
				], Some(cons(TT::OParen, &[
					ident("vmul"),
					cons(TT::Comma, &[
						ident("p"),
						ident("q"),
					]),
				]))),
			])
	}
}

