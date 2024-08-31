
use miette::{IntoDiagnostic, WrapErr};

use crate::lexer::{Token, TokenType};

pub(crate) type TypedIdent = (String, ValueType);

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, PartialEq)]
pub(crate) enum Expression {
	/// literal := _number:64b_ value-type
	Literal(u64,Option<ValueType>),
	/// function-call := ident expr*
	FnCall(String, Vec<Expression>),
	/// field-access := ident ('.' ident)+
	FieldAccess(String, Vec<String>),
}

#[derive(Debug, PartialEq)]
pub(crate) enum SExpression {
	Function {
		ident : String,
		params: Vec<TypedIdent>,
		body  : (Vec<SExpression>, Option<Expression>),
	},
	Value {
		ident: (String, Option<ValueType>),
		expr : Expression,
	},
	Record {
		ident : String,
		fields: Vec<TypedIdent>,
	},
}

struct Parser<'a,'b> where 'b: 'a {
	index: usize,
	input: &'a [Token<'b>],
}

impl<'a,'b> Parser<'a,'b> {
	fn new(input: &'a [Token<'b>]) -> Self {
		Self {
			index: 0,
			input,
		}
	}

	fn peek(&self) -> &Token {
		&self.input[self.index]
	}

	fn next_token(&mut self) {
		self.index += 1;
	}

	fn is_empty(&self) -> bool {
		self.index == self.input.len()
	}
}


pub fn eval(
	input: &[Token],
) -> miette::Result<Vec<SExpression>> {
	if input.len() == 0 {
		miette::bail!("Empty input");
	}

	program(Parser::new(input))
}

fn num(data: &'_ mut Parser) -> miette::Result<u64> {
	let token = data.peek();
	if token.get_type() != TokenType::Num {
		miette::bail!("{}", expected("Number"));
	}
	let out = token.get_token()
		.parse::<f64>()
		.into_diagnostic()
		.wrap_err("lexer should not allow invalid floating-point values")?;
	data.next_token();
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

fn ident(data: &'_ mut Parser) -> miette::Result<String> {
	let token = data.peek();
	if token.get_type() != TokenType::Ident {
		miette::bail!("{}", expected("Identifier"));
	}
	let out = token.get_token().to_owned();
	data.next_token();
	Ok(out)
}

fn value_type(
	data: &mut Parser,
) -> miette::Result<ValueType> {
	let token = data.peek();
	if !matches!(token.get_type(),
		TokenType::U8 | TokenType::U16 | TokenType::U32 |
		TokenType::S8 | TokenType::S16 | TokenType::S32 |
		TokenType::F16 | TokenType::F32)
	{
		miette::bail!("{}", expected("Value Type"));
	}

	match token.get_type() {
		TokenType::U8 => Ok(ValueType::U8),
		TokenType::U16 => Ok(ValueType::U16),
		TokenType::U32 => Ok(ValueType::U32),
		TokenType::S8 => Ok(ValueType::S8),
		TokenType::S16 => Ok(ValueType::S16),
		TokenType::S32 => Ok(ValueType::S32),
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

			data.next_token();
			Ok(ValueType::F16(bits))
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

			data.next_token();
			Ok(ValueType::F32(bits))
		}
		TokenType::Ident => Ok(ValueType::TypeName(token.get_token().to_owned())),
		_ => miette::bail!("Expected Value Type"),
	}
}

fn ident_typed_opt(
	data: &mut Parser,
) -> miette::Result<(String,Option<ValueType>)> {
	let id = ident(data)?;
	let val_type = value_type(data).ok();
	Ok((id, val_type))
}

fn match_token(
	data: &mut Parser,
	tt: TokenType,
) -> miette::Result<()> {
	if data.peek().get_type() == tt {
		Ok(data.next_token())
	} else {
		miette::bail!("{}",
			expected(&format!("{tt:?} @ {}", data.index)))
	}
}

fn minor_expr(
	data: &mut Parser,
) -> miette::Result<Expression> {
	let token = data.peek().clone();
	match token.get_type() {
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
		TokenType::F32 => miette::bail!("{}",
			expected("Expression")),

		t @ TokenType::Plus |
		t @ TokenType::Minus |
		t @ TokenType::Star |
		t @ TokenType::Slash |
		t @ TokenType::Percent |
		t @ TokenType::LArrow |
		t @ TokenType::RArrow |
		t @ TokenType::UArrow |
		t @ TokenType::Bang |
		t @ TokenType::Ampersand |
		t @ TokenType::Pipe |
		t @ TokenType::Dot |
		t @ TokenType::RParen |
		t @ TokenType::Eq => miette::bail!("Unexpected {t:?}"),

		TokenType::LParen => {
			match_token(data, TokenType::LParen)?;
			let out = expr(data)?;
			match_token(data, TokenType::RParen)?;
			Ok(out)
		}

		TokenType::Ident => {
			let s = token.get_token().to_owned();
			data.next_token();
			let mut fields = Vec::default();
			while match_token(data, TokenType::Dot).is_ok() {
				fields.push(ident(data)?);
			}
			if !fields.is_empty() {
				Ok(Expression::FieldAccess(s, fields))
			} else {
				Ok(Expression::FnCall(s, vec![]))
			}
		}

		TokenType::Num => {
			let n = num(data)?;
			let val_type = value_type(data).ok();
			Ok(Expression::Literal(n, val_type))
		}
	}
}

fn gather_params(
	data: &mut Parser,
	count: usize,
) -> miette::Result<Vec<Expression>> {
	let mut out = Vec::default();
	for _ in 0..count {
		out.push(minor_expr(data)?);
	}
	while data.peek().get_type() != TokenType::RParen {
		out.push(minor_expr(data)?);
	}
	Ok(out)
}

fn expr(data: &mut Parser) -> miette::Result<Expression> {
	let token = data.peek().clone();
	match token.get_type() {
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
		TokenType::F32 => miette::bail!("{}",
			expected("Expression")),

		TokenType::Dot => miette::bail!("Unexpected '.'"),
		TokenType::RParen => miette::bail!("Unexpected ')'"),

		TokenType::Plus => {
			match_token(data, TokenType::Plus)?;
			Ok(Expression::FnCall("Add".to_string(),
				gather_params(data, 1)?))
		}
		TokenType::Minus => {
			match_token(data, TokenType::Minus)?;
			Ok(Expression::FnCall("Sub".to_string(),
				gather_params(data, 1)?))
		}
		TokenType::Star => {
			match_token(data, TokenType::Star)?;
			Ok(Expression::FnCall("Mul".to_string(),
				gather_params(data, 2)?))
		}
		TokenType::Slash => {
			match_token(data, TokenType::Slash)?;
			if match_token(data, TokenType::Percent).is_ok() {
				Ok(Expression::FnCall("DivMod".to_string(),
					gather_params(data, 2)?))
			} else {
				Ok(Expression::FnCall("Div".to_string(),
					gather_params(data, 2)?))
			}
		}
		TokenType::Percent => {
			match_token(data, TokenType::Percent)?;
			Ok(Expression::FnCall("Mod".to_string(),
				gather_params(data, 2)?))
		}
		TokenType::LParen => {
			match_token(data, TokenType::LParen)?;
			let out = expr(data)?;
			match_token(data, TokenType::RParen)?;
			Ok(out)
		}
		TokenType::LArrow => {
			match_token(data, TokenType::LArrow)?;
			if match_token(data, TokenType::LArrow).is_ok() {
				Ok(Expression::FnCall("ShiftL".to_string(),
					gather_params(data, 2)?))
			} else if match_token(data, TokenType::RArrow).is_ok() {
				Ok(Expression::FnCall("NEq".to_string(),
					gather_params(data, 2)?))
			} else if match_token(data, TokenType::Eq).is_ok() {
				Ok(Expression::FnCall("LE".to_string(),
					gather_params(data, 2)?))
			} else {
				Ok(Expression::FnCall("LT".to_string(),
					gather_params(data, 2)?))
			}
		}
		TokenType::RArrow => {
			match_token(data, TokenType::RArrow)?;
			if match_token(data, TokenType::RArrow).is_ok() {
				Ok(Expression::FnCall("ShiftR".to_string(),
					gather_params(data, 2)?))
			} else if match_token(data, TokenType::Eq).is_ok() {
				Ok(Expression::FnCall("GE".to_string(),
					gather_params(data, 2)?))
			} else {
				Ok(Expression::FnCall("GT".to_string(),
					gather_params(data, 2)?))
			}
		}
		TokenType::Ampersand => {
			match_token(data, TokenType::Ampersand)?;
			if match_token(data, TokenType::Ampersand).is_ok() {
				Ok(Expression::FnCall("AndL".to_string(),
					gather_params(data, 2)?))
			} else {
				Ok(Expression::FnCall("AndB".to_string(),
					gather_params(data, 2)?))
			}
		}
		TokenType::Pipe => {
			match_token(data, TokenType::Pipe)?;
			if match_token(data, TokenType::Pipe).is_ok() {
				Ok(Expression::FnCall("OrL".to_string(),
					gather_params(data, 2)?))
			} else {
				Ok(Expression::FnCall("OrB".to_string(),
					gather_params(data, 2)?))
			}
		}
		TokenType::UArrow => {
			match_token(data, TokenType::UArrow)?;
			if match_token(data, TokenType::UArrow).is_ok() {
				Ok(Expression::FnCall("XorL".to_string(),
					gather_params(data, 2)?))
			} else {
				Ok(Expression::FnCall("XorB".to_string(),
					gather_params(data, 2)?))
			}
		}
		TokenType::Bang => {
			match_token(data, TokenType::Bang)?;
			let out = vec![minor_expr(data)?];
			Ok(Expression::FnCall("Not".to_string(), out))
		}
		TokenType::Eq => {
			match_token(data, TokenType::Eq)?;
			Ok(Expression::FnCall("Eq".to_string(),
				gather_params(data, 2)?))
		}
		TokenType::Ident => {
			let mut s = ident(data)?;
			while match_token(data, TokenType::Dot).is_ok() {
				s.push('.');
				s.push_str(&ident(data)?);
			}
			Ok(Expression::FnCall(s, gather_params(data, 0)?))
		}
		TokenType::Num => {
			let n = num(data)?;
			let val_type = value_type(data).ok();
			Ok(Expression::Literal(n, val_type))
		}
	}
}

fn ident_typed(
	data: &mut Parser,
) -> miette::Result<TypedIdent> {
	let id = ident(data)?;
	let val_type = value_type(data)?;
	Ok((id, val_type))
}

fn param_list(
	data: &mut Parser,
) -> miette::Result<Vec<TypedIdent>> {
	let mut params = Vec::default();
	while data.peek().get_type() != TokenType::RParen {
		match_token(data, TokenType::LParen)?;
		params.push(ident_typed(data)?);
		match_token(data, TokenType::RParen)?;
	}
	Ok(params)
}

fn body(
	data: &mut Parser,
) -> miette::Result<(Vec<SExpression>, Option<Expression>)> {
	let mut out1 = Vec::default();
	while data.peek().get_type() == TokenType::LParen {
		match_token(data, TokenType::LParen)?;
		out1.push(s_expr(data)?);
		match_token(data, TokenType::RParen)?;
	}
	let out2 = if data.peek().get_type() == TokenType::RParen {
		None
	} else {
		Some(expr(data)?)
	};
	Ok((out1, out2))
}

fn s_expr(
	data: &mut Parser,
) -> miette::Result<SExpression> {
	match data.peek().get_type() {
		TokenType::Let => {
			match_token(data, TokenType::Let)?;
			match_token(data, TokenType::LParen)?;
			let ident = ident_typed_opt(data)?;
			match_token(data, TokenType::RParen)?;
			match_token(data, TokenType::LParen)?;
			let expr = expr(data)?;
			match_token(data, TokenType::RParen)?;
			Ok(SExpression::Value { ident, expr })
		}
		TokenType::Fn => {
			match_token(data, TokenType::Fn)?;
			let ident = ident(data)?;
			match_token(data, TokenType::LParen)?;
			let params = param_list(data)?;
			match_token(data, TokenType::RParen)?;
			match_token(data, TokenType::LParen)?;
			let body = body(data)?;
			match_token(data, TokenType::RParen)?;
			Ok(SExpression::Function { ident, params, body })
		}
		TokenType::Rec => {
			match_token(data, TokenType::Rec)?;
			let ident = ident(data)?;
			match_token(data, TokenType::LParen)?;
			let fields = param_list(data)?;
			match_token(data, TokenType::RParen)?;
			Ok(SExpression::Record { ident, fields })
		}
		_ => miette::bail!("{}", expected("S-Expression")),
	}
}

fn program(
	mut data: Parser,
) -> miette::Result<Vec<SExpression>> {
	let mut program = Vec::default();
	while !data.is_empty() {
		program.push(s_expr(&mut data)?);
	}
	Ok(program)
}

fn expected(s: &str) -> String {
	format!("{s} Expected")
}

#[test]
fn let_expressions_bind_value_types(
) -> miette::Result<()> {
	let input = "let (a u8) (3)";
	let ast = eval(&crate::lexer::eval(input)?)?;
	assert_eq!(ast.len(), 1);
	assert_eq!(ast[0], SExpression::Value {
		ident: ("a".to_string(), Some(ValueType::U8)),
		expr: Expression::Literal(3, None),
	});
	Ok(())
}

#[test]
fn field_accessor_in_call_position_captures_all_accesses(
) -> miette::Result<()> {
	let input = "let (a) (b.c.d)";
	let lexemes = crate::lexer::eval(input)?;
	let ast = eval(&lexemes)?;
	assert_eq!(ast.len(), 1);
	assert_eq!(ast[0], SExpression::Value {
		ident: ("a".to_string(), None),
		expr: Expression::FnCall("b.c.d".to_string(), vec![]),
	});
	Ok(())
}

