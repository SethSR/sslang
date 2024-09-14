
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

impl std::fmt::Display for S<'_> {
	fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
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

fn num(
	parser: &mut Parser,
) -> miette::Result<u64> {
	let token = parser.input[parser.index].clone();
	if token.tt != TokenType::Num {
		miette::bail!("{}", expected("Number"));
	}
	let out = token.to_string()
		.parse::<f64>()
		.into_diagnostic()
		.wrap_err("lexer should not allow invalid floating-point values")?;
	parser.index += 1;
	Ok(float_to_fixed(out))
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
	let token = parser.input[parser.index].clone();
	if token.tt != TokenType::Ident {
		miette::bail!("{}", expected("Identifier"));
	}
	let out = token.to_string().to_owned();
	parser.index += 1;
	Ok(out)
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
	if parser.index >= parser.input.len() {
		miette::bail!("Unexpected EOF");
	}

	let token = &parser.input[parser.index];
	if token.tt == tt {
		parser.index += 1;
		Ok(())
	} else {
		Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(token.range.clone(), "here")
			],
			"Expected {tt:?}",
		}.with_source_code(token.src.to_owned()))
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
	fn peek(&self) -> &Token<'a> {
		&self.input[self.index]
	}
}

enum Expr {}

#[derive(Debug, Clone)]
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

fn prefix_binding_power(op: Op) -> ((),u8) {
	match op {
		Op::Add | Op::Sub => ((),11),
		Op::Ref | Op::Deref => ((),13),
		Op::Not => ((),15),
		_ => panic!("bad unary op"),
	}
}

fn infix_binding_power(op: Op) -> (u8,u8) {
	match op {
		Op::AndL | Op::OrL | Op::XorL => (1,2),
		Op::AndB | Op::OrB | Op::XorB => (3,4),
		Op::CmpEq | Op::CmpNE | Op::CmpGT | Op::CmpGE | Op::CmpLT | Op::CmpLE => (5,6),
		Op::Add | Op::Sub => (7,8),
		Op::Mul | Op::Div | Op::Mod | Op::DivMod | Op::LShift | Op::RShift => (9,10),
		Op::FieldAccess => (18,17),
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

	let left_token = parser.peek().clone();
	let mut lhs = match left_token.tt {
		TT::Ident |
		TT::Num => S::Atom(left_token),
		TT::Plus => {
			let ((),r_bp) = prefix_binding_power(Op::Add);
			let rhs = expr(parser, r_bp)?;
			S::Cons(left_token, vec![rhs])
		}
		TT::Minus => {
			let ((),r_bp) = prefix_binding_power(Op::Sub);
			let rhs = expr(parser, r_bp)?;
			S::Cons(left_token, vec![rhs])
		}
		TT::Bang => {
			let ((),r_bp) = prefix_binding_power(Op::Not);
			let rhs = expr(parser, r_bp)?;
			S::Cons(left_token, vec![rhs])
		}
		_ => miette::bail! {
			"Expected Identifier, Function Call, or Literal"
		}
	};
	parser.index += 1;

	loop {
		let op_token = parser.peek().clone();
		let op = match op_token.tt {
			TT::Amp1 => Op::AndB,
			TT::Amp2 => Op::AndL,
			TT::At => Op::Deref,
			TT::Bar1 => Op::OrB,
			TT::Bar2 => Op::OrL,
			TT::Bang => Op::Not,
			TT::BangEq => Op::CmpNE,
			TT::Carrot1 => Op::XorB,
			TT::Carrot2 => Op::XorL,
			TT::Dollar => Op::Ref,
			TT::Dot => Op::FieldAccess,
			TT::Eq1 => Op::CmpEq,
			TT::Eq2 => Op::CmpEq,
			TT::LArrow1 => Op::CmpLT,
			TT::LArrow2 => Op::LShift,
			TT::LArrBar => Op::LRotate,
			TT::LArrEq => Op::CmpLE,
			TT::Minus => Op::Sub,
			TT::Percent => Op::Mod,
			TT::Plus => Op::Add,
			TT::RArrow1 => Op::CmpGT,
			TT::RArrow2 => Op::RShift,
			TT::RArrBar => Op::RRotate,
			TT::RArrEq => Op::CmpGE,
			TT::Slash => Op::Div,
			TT::SlashPer => Op::DivMod,
			TT::Star => Op::Mul,

			TT::Ident | TT::Num |
			TT::If | TT::Else | TT::Fun | TT::Rec | TT::Var | TT::While |
			TT::U8 | TT::U16 | TT::U32 | TT::S8 | TT::S16 | TT::S32 | TT::F16 | TT::F32 |
			TT::Colon | TT::CParen | TT::OParen |
			TT::EOF => break,
		};
		let (l_bp, r_bp) = infix_binding_power(op);

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
	while parser.peek().tt != TokenType::CParen {
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
	let body = expr(parser, 0)?;
	Ok(Stmt::Var { name, vtype, body })
}

/// statement := rec | fun | var
fn statement<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt<'a>> {
	match parser.peek().clone() {
		Token { tt: TokenType::Rec, ..} => rec(parser),
		Token { tt: TokenType::Fun, ..} => fun(parser),
		Token { tt: TokenType::Var, ..} => var(parser),
		token => Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(token.range, "here"),
			],
			"Expected Statement"
		}.with_source_code(token.src.to_owned())),
	}
}

/// program := statement*
fn program<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Vec<Stmt<'a>>> {
	let mut program = Vec::default();
	while parser.index < parser.input.len() {
		program.push(statement(parser)?);
	}
	Ok(program)
}

fn expected(s: &str) -> String {
	format!("{s} Expected")
}

/*
#[test]
fn let_expressions_set_unknown_value_types_to_S32(
) -> miette::Result<()> {
	let input = "let a 3";
	let ast = eval(input, crate::lexer::eval(input)?)?;
	assert_eq!(ast, vec![
		S::Let {
			ident: ("a".to_string(), ValueType::S32),
			value: Expression::Literal(3, None),
		}
	]);
	Ok(())
}

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

