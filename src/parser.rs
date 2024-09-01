
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

#[allow(dead_code)]
pub(crate) enum Op {
	Add,
	Sub,
	Mul,
	DivMod,
	Div,
	Mod,
	LShift,
	RShift,
	LRotate,
	RRotate,
	And,
	Or,
	Xor,
	Not,
	Neg,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Expression {
	/// literal := _number:64b_ value-type
	Literal(u64,Option<ValueType>),
	/// function-call := ident expr*
	FnCall(String, Vec<Expression>),
}

#[derive(Debug, PartialEq)]
pub(crate) enum SExpression<'a> {
	Expr(Token<'a>),
	List(Vec<SExpression<'a>>),
}

impl std::fmt::Display for SExpression<'_> {
	fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
		match self {
			SExpression::Expr(token) => write!(fmt, "{:?}", token.tt),
			SExpression::List(list) => {
				let out = list.iter().map(|i| format!("{i}")).collect::<Vec<String>>().join(" ");
				write!(fmt, "({out})")
			}
		}
	}
}

pub fn eval<'a>(
	input: Vec<Token<'a>>,
) -> miette::Result<SExpression<'a>> {
	if input.len() == 0 {
		miette::bail!("Empty input");
	}

	let mut index = 0;
	let ast = program(&input, &mut index)?;
	Ok(ast)
}

fn num(
	input: &[Token],
	index: &mut usize,
) -> miette::Result<u64> {
	let token = input[*index].clone();
	if token.tt != TokenType::Num {
		miette::bail!("{}", expected("Number"));
	}
	let out = token.get_token()
		.parse::<f64>()
		.into_diagnostic()
		.wrap_err("lexer should not allow invalid floating-point values")?;
	*index += 1;
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
	input: &[Token],
	index: &mut usize,
) -> miette::Result<String> {
	let token = input[*index].clone();
	if token.tt != TokenType::Ident {
		miette::bail!("{}", expected("Identifier"));
	}
	let out = token.get_token().to_owned();
	*index += 1;
	Ok(out)
}

fn value_type(
	input: &[Token],
	index: &mut usize,
) -> miette::Result<ValueType> {
	let token = input[*index].clone();
	let out = match token.tt {
		TokenType::U8  => ValueType::U8,
		TokenType::U16 => ValueType::U16,
		TokenType::U32 => ValueType::U32,
		TokenType::S8  => ValueType::S8,
		TokenType::S16 => ValueType::S16,
		TokenType::S32 => ValueType::S32,
		TokenType::F16 => {
			let Some(bit_spec) = token.get_token()
				.strip_prefix("fw")
			else {
				miette::bail!("Parsed a Fixed-point Word that doesn't start with 'fw'");
			};

			let Ok(bits) = bit_spec.parse::<u8>()
			else {
				miette::bail!("Unable to parse {} into Fixed-point Word type", token.get_token());
			};

			if bits > 16 {
				miette::bail!("{}",
					expected("Bit specifier between 0..=16"));
			}
			ValueType::F16(bits)
		}
		TokenType::F32 => {
			let Some(bit_spec) = token.get_token()
				.strip_prefix("fl")
			else {
				miette::bail!("Parsed a Fixed-point Long that doesn't start with 'fl'");
			};

			let Ok(bits) = bit_spec.parse::<u8>() else {
				miette::bail!("Unable to parse {} into Fixed-point Long type", token.get_token());
			};

			if bits > 32 {
				miette::bail!("{}",
					expected("Bit specifier between 0..=32"));
			}
			ValueType::F32(bits)
		}
		TokenType::Ident => ValueType::TypeName(token.get_token().to_owned()),
		_ => miette::bail!("Expected Value Type"),
	};
	*index += 1;
	Ok(out)
}

fn ident_typed_opt(
	input: &[Token],
	index: &mut usize,
) -> miette::Result<(String,Option<ValueType>)> {
	let id = ident(input, index)?;
	let val_type = value_type(input, index).ok();
	Ok((id, val_type))
}

fn match_token(
	input: &[Token],
	index: &mut usize,
	tt: TokenType,
) -> miette::Result<()> {
	if *index >= input.len() {
		miette::bail!("Unexpected EOF");
	}

	let token = &input[*index];
	if token.tt == tt {
		*index += 1;
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

fn minor_expr(
	input: &[Token],
	index: &mut usize,
) -> miette::Result<Expression> {
	let token = input[*index].clone();
	match token.tt {
		TokenType::Let |
		TokenType::Fn |
		TokenType::Rec |
		TokenType::U8 |
		TokenType::U16 |
		TokenType::U32 |
		TokenType::S8 |
		TokenType::S16 |
		TokenType::S32 |
		TokenType::F16 |
		TokenType::F32 => Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(token.range.clone(), "here"),
			],
			"Expected Expression",
		}.with_source_code(token.src.to_owned())),

		t @ TokenType::Plus |
		t @ TokenType::Minus |
		t @ TokenType::Star |
		t @ TokenType::Slash |
		t @ TokenType::Percent |
		t @ TokenType::LArrow |
		t @ TokenType::RArrow |
		t @ TokenType::Carrot |
		t @ TokenType::Bang |
		t @ TokenType::Ampersand |
		t @ TokenType::Pipe |
		t @ TokenType::Dot |
		t @ TokenType::CParen |
		t @ TokenType::Eq => Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(token.range.clone(), "here"),
			],
			"Unexpected {t:?}",
		}.with_source_code(token.src.to_owned())),

		TokenType::OParen => {
			match_token(input, index, TokenType::OParen)?;
			let out = expr(input, index)?;
			match_token(input, index, TokenType::CParen)?;
			Ok(out)
		}

		TokenType::Ident => {
			let mut s = token.get_token().to_owned();
			*index += 1;
			while match_token(input, index, TokenType::Dot).is_ok() {
				s.push('.');
				s.push_str(&ident(input, index)?);
			}
			Ok(Expression::FnCall(s, vec![]))
		}

		TokenType::Num => {
			let n = num(input, index)?;
			let val_type = value_type(input, index).ok();
			Ok(Expression::Literal(n, val_type))
		}
	}
}

fn gather_params(
	input: &[Token],
	index: &mut usize,
	count: usize,
) -> miette::Result<Vec<Expression>> {
	let mut out = Vec::default();
	for _ in 0..count {
		out.push(minor_expr(input, index)?);
	}
	while input[*index].tt != TokenType::CParen {
		out.push(minor_expr(input, index)?);
	}
	Ok(out)
}

fn expr(
	input: &[Token],
	index: &mut usize,
) -> miette::Result<Expression> {
	let token = input[*index].clone();
	match token.tt {
		TokenType::Let |
		TokenType::Fn |
		TokenType::Rec |
		TokenType::U8 |
		TokenType::U16 |
		TokenType::U32 |
		TokenType::S8 |
		TokenType::S16 |
		TokenType::S32 |
		TokenType::F16 |
		TokenType::F32 |
		TokenType::Dot |
		TokenType::OParen |
		TokenType::CParen => Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(token.range.clone(), "here"),
			],
			"Unexpected '{:?}'", token.tt,
		}.with_source_code(token.src.to_owned())),

		TokenType::Plus => {
			match_token(input, index, TokenType::Plus)?;
			Ok(Expression::FnCall("Add".to_string(),
				gather_params(input, index, 1)?))
		}
		TokenType::Minus => {
			match_token(input, index, TokenType::Minus)?;
			Ok(Expression::FnCall("Sub".to_string(),
				gather_params(input, index, 1)?))
		}
		TokenType::Star => {
			match_token(input, index, TokenType::Star)?;
			Ok(Expression::FnCall("Mul".to_string(),
				gather_params(input, index, 2)?))
		}
		TokenType::Slash => {
			match_token(input, index, TokenType::Slash)?;
			if match_token(input, index, TokenType::Percent).is_ok() {
				Ok(Expression::FnCall("DivMod".to_string(),
					gather_params(input, index, 2)?))
			} else {
				Ok(Expression::FnCall("Div".to_string(),
					gather_params(input, index, 2)?))
			}
		}
		TokenType::Percent => {
			match_token(input, index, TokenType::Percent)?;
			Ok(Expression::FnCall("Mod".to_string(),
				gather_params(input, index, 2)?))
		}

		TokenType::LArrow => {
			match_token(input, index, TokenType::LArrow)?;
			if match_token(input, index, TokenType::LArrow).is_ok() {
				Ok(Expression::FnCall("ShiftL".to_string(),
					gather_params(input, index, 2)?))
			} else if match_token(input, index, TokenType::RArrow).is_ok() {
				Ok(Expression::FnCall("NEq".to_string(),
					gather_params(input, index, 2)?))
			} else if match_token(input, index, TokenType::Eq).is_ok() {
				Ok(Expression::FnCall("LE".to_string(),
					gather_params(input, index, 2)?))
			} else {
				Ok(Expression::FnCall("LT".to_string(),
					gather_params(input, index, 2)?))
			}
		}
		TokenType::RArrow => {
			match_token(input, index, TokenType::RArrow)?;
			if match_token(input, index, TokenType::RArrow).is_ok() {
				Ok(Expression::FnCall("ShiftR".to_string(),
					gather_params(input, index, 2)?))
			} else if match_token(input, index, TokenType::Eq).is_ok() {
				Ok(Expression::FnCall("GE".to_string(),
					gather_params(input, index, 2)?))
			} else {
				Ok(Expression::FnCall("GT".to_string(),
					gather_params(input, index, 2)?))
			}
		}
		TokenType::Ampersand => {
			match_token(input, index, TokenType::Ampersand)?;
			if match_token(input, index, TokenType::Ampersand).is_ok() {
				Ok(Expression::FnCall("AndL".to_string(),
					gather_params(input, index, 2)?))
			} else {
				Ok(Expression::FnCall("AndB".to_string(),
					gather_params(input, index, 2)?))
			}
		}
		TokenType::Pipe => {
			match_token(input, index, TokenType::Pipe)?;
			if match_token(input, index, TokenType::Pipe).is_ok() {
				Ok(Expression::FnCall("OrL".to_string(),
					gather_params(input, index, 2)?))
			} else {
				Ok(Expression::FnCall("OrB".to_string(),
					gather_params(input, index, 2)?))
			}
		}
		TokenType::Carrot => {
			match_token(input, index, TokenType::Carrot)?;
			if match_token(input, index, TokenType::Carrot).is_ok() {
				Ok(Expression::FnCall("XorL".to_string(),
					gather_params(input, index, 2)?))
			} else {
				Ok(Expression::FnCall("XorB".to_string(),
					gather_params(input, index, 2)?))
			}
		}
		TokenType::Bang => {
			match_token(input, index, TokenType::Bang)?;
			let out = vec![minor_expr(input, index)?];
			Ok(Expression::FnCall("Not".to_string(), out))
		}
		TokenType::Eq => {
			match_token(input, index, TokenType::Eq)?;
			Ok(Expression::FnCall("Eq".to_string(),
				gather_params(input, index, 2)?))
		}
		TokenType::Ident => {
			let mut s = ident(input, index)?;
			while match_token(input, index, TokenType::Dot).is_ok() {
				s.push('.');
				s.push_str(&ident(input, index)?);
			}
			Ok(Expression::FnCall(s, gather_params(input, index, 0)?))
		}
		TokenType::Num => {
			let n = num(input, index)?;
			let val_type = value_type(input, index).ok();
			Ok(Expression::Literal(n, val_type))
		}
	}
}

fn ident_typed(
	input: &[Token],
	index: &mut usize,
) -> miette::Result<TypedIdent> {
	let id = ident(input, index)?;
	let val_type = value_type(input, index)?;
	Ok((id, val_type))
}

fn param_list(
	input: &[Token],
	index: &mut usize,
) -> miette::Result<Vec<TypedIdent>> {
	let mut params = Vec::default();
	while input[*index].tt != TokenType::CParen {
		match_token(input, index, TokenType::OParen)?;
		params.push(ident_typed(input, index)?);
		match_token(input, index, TokenType::CParen)?;
	}
	Ok(params)
}

fn list<'a>(
	input: &[Token<'a>],
	index: &mut usize,
) -> miette::Result<Vec<SExpression<'a>>> {
	let mut out = Vec::default();
	while *index < input.len() && input[*index].tt != TokenType::CParen {
		out.push(s_expr(input, index)?);
	}
	Ok(out)
}

fn s_expr<'a>(
	input: &[Token<'a>],
	index: &mut usize,
) -> miette::Result<SExpression<'a>> {
	if *index >= input.len() {
		miette::bail!("Unexpected EOF");
	}

	match input[*index].tt {
		TokenType::OParen => {
			match_token(input, index, TokenType::OParen)?;
			let mut list = list(input, index)?;
			match_token(input, index, TokenType::CParen)?;
			if list.len() == 1 {
				Ok(list.remove(0))
			} else {
				Ok(SExpression::List(list))
			}
		}

		_ => {
			*index += 1;
			Ok(SExpression::Expr(input[*index-1].clone()))
/*
			Err(miette::miette! {
				labels = vec![
					LabeledSpan::at(token.range, "here"),
				],
				"Expected SExpression"
			}.with_source_code(data.source.to_owned()))
*/
		}
	}
}

fn program<'a>(
	input: &[Token<'a>],
	index: &mut usize,
) -> miette::Result<SExpression<'a>> {
	let mut program = Vec::default();
	while *index < input.len() {
		program.push(s_expr(input, index)?);
	}
	if program.len() == 1 {
		Ok(program.remove(0))
	} else {
		Ok(SExpression::List(program))
	}
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
		SExpression::Let {
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
		SExpression::Expr(Token::new(TokenType::Let, 0..3)),
		SExpression::List(vec![
			SExpression::Expr(Token::new(TokenType::Ident, 5..6)),
			SExpression::Expr(Token::new(TokenType::U8, 7..9)),
		]),
		SExpression::Expr(Token::new(TokenType::Num, 12..13)),
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
		SExpression::Expr(Token::new(TokenType::Let, 0..3)),
		SExpression::Expr(Token::new(TokenType::Ident, 5..6)),
		SExpression::List(vec![
			SExpression::Expr(Token::new(TokenType::Ident, 9..10)),
			SExpression::Expr(Token::new(TokenType::Dot, 10..11)),
			SExpression::Expr(Token::new(TokenType::Ident, 11..12)),
			SExpression::Expr(Token::new(TokenType::Dot, 12..13)),
			SExpression::Expr(Token::new(TokenType::Ident, 13..14)),
		]),
	]);
	Ok(())
}
*/

