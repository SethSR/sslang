
use std::fmt;
use std::ops::Range;

use miette::{IntoDiagnostic, LabeledSpan, WrapErr};
use tracing::{instrument, trace};

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

impl fmt::Display for ValueType {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		use ValueType as VT;

		match self {
			VT::U8 => write!(fmt, "u8"),
			VT::U16 => write!(fmt, "u16"),
			VT::U32 => write!(fmt, "u32"),
			VT::S8 => write!(fmt, "s8"),
			VT::S16 => write!(fmt, "s16"),
			VT::S32 => write!(fmt, "s32"),
			VT::F16(n) => write!(fmt, "fw{n}"),
			VT::F32(n) => write!(fmt, "fl{n}"),
			VT::UDT(s) => write!(fmt, "{s}"),
		}
	}
}

type TokenInfo = Range<usize>;

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Atom {
	Num(i64),
	Id(String),
	Block(Block),
	If(S, Block, Option<Block>),
}

impl fmt::Display for Atom {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Num(n) => write!(fmt, "{n}"),
			Self::Id(s) => write!(fmt, "{s}"),
			Self::Block(b) => write!(fmt, "{b}"),
			Self::If(cond, bt, Some(bf)) => write!(fmt, "(if {cond} {bt} else {bf})"),
			Self::If(cond, bt, None) => write!(fmt, "(if {cond} {bt})"),
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
	Accessor, // .
	Add,      // +
	AndB,     // &
	AndL,     // &&
	Assign,   // =
	CmpEq,    // ==
	CmpGE,    // >=
	CmpGT,    // >
	CmpLE,    // <=
	CmpLT,    // <
	CmpNE,    // !=
	Comma,    // ,
	Deref,    // @
	Div,      // /
	DivMod,   // /%
	FnCall,   // (
	LFShift,  // <|
	LShift,   // <<
	Mod,      // %
	Mul,      // *
	Not,      // !
	OrB,      // |
	OrL,      // ||
	RFShift,  // |>
	RShift,   // >>
	Ref,      // $
	Sub,      // -
	XorB,     // ^
	XorL,     // ^^
}

impl fmt::Display for Op {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		let op = match self {
			Op::Accessor => ".",
			Op::Add      => "+",
			Op::AndB     => "&",
			Op::AndL     => "&&",
			Op::Assign   => "=",
			Op::CmpEq    => "==",
			Op::CmpGE    => ">=",
			Op::CmpGT    => ">",
			Op::CmpLE    => "<=",
			Op::CmpLT    => "<",
			Op::CmpNE    => "!=",
			Op::Comma    => ",",
			Op::Deref    => "@",
			Op::Div      => "/",
			Op::DivMod   => "/%",
			Op::FnCall   => "call",
			Op::LFShift  => "<|",
			Op::LShift   => "<<",
			Op::Mod      => "%",
			Op::Mul      => "*",
			Op::Not      => "!",
			Op::OrB      => "|",
			Op::OrL      => "||",
			Op::RFShift  => "|>",
			Op::RShift   => ">>",
			Op::Ref      => "$",
			Op::Sub      => "-",
			Op::XorB     => "^",
			Op::XorL     => "^^",
		};
		write!(fmt, "{op}")
	}
}

impl TryFrom<TokenType<'_>> for Op {
	type Error = miette::Report;
	fn try_from(tt: TokenType) -> Result<Self, Self::Error> {
		match tt {
			TokenType::Amp1     => Ok(Op::AndB),
			TokenType::Amp2     => Ok(Op::AndL),
			TokenType::At       => Ok(Op::Deref),
			TokenType::Bang     => Ok(Op::Not),
			TokenType::BangEq   => Ok(Op::CmpNE),
			TokenType::Bar1     => Ok(Op::OrB),
			TokenType::Bar2     => Ok(Op::OrL),
			TokenType::Carrot1  => Ok(Op::XorB),
			TokenType::Carrot2  => Ok(Op::XorL),
			TokenType::Comma    => Ok(Op::Comma),
			TokenType::Dollar   => Ok(Op::Ref),
			TokenType::Dot      => Ok(Op::Accessor),
			TokenType::Eq1      => Ok(Op::Assign),
			TokenType::Eq2      => Ok(Op::CmpEq),
			TokenType::LArrow1  => Ok(Op::CmpLT),
			TokenType::LArrow2  => Ok(Op::LShift),
			TokenType::LArrBar  => Ok(Op::LFShift),
			TokenType::LArrEq   => Ok(Op::CmpLE),
			TokenType::Minus    => Ok(Op::Sub),
			TokenType::OParen   => Ok(Op::FnCall),
			TokenType::Percent  => Ok(Op::Mod),
			TokenType::Plus     => Ok(Op::Add),
			TokenType::RArrow1  => Ok(Op::CmpGT),
			TokenType::RArrow2  => Ok(Op::RShift),
			TokenType::RArrBar  => Ok(Op::RFShift),
			TokenType::RArrEq   => Ok(Op::CmpGE),
			TokenType::Slash    => Ok(Op::Div),
			TokenType::SlashPer => Ok(Op::DivMod),
			TokenType::Star     => Ok(Op::Mul),
			_ => Err(miette::miette! {
				"{tt:?} is not an operator"
			})
		}
	}
}

#[derive(Clone)]
pub(crate) enum S {
	Atom(Box<Atom>, TokenInfo),
	Cons(Op, TokenInfo, Vec<S>),
}

impl PartialEq for S {
	fn eq(&self, rhs: &Self) -> bool {
		match (self, rhs) {
			(Self::Atom(left_atom,_), Self::Atom(right_atom,_)) =>
				left_atom == right_atom,
			(Self::Cons(left_op,_,left_list), Self::Cons(right_op,_,right_list)) =>
				left_op == right_op && left_list == right_list,
			_ => false,
		}
	}
}

impl fmt::Debug for S {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		match self {
			S::Atom(token, info) => write!(fmt, "{token:?}[{info:?}]"),
			S::Cons(head, info, rest) => {
				let out = rest.iter().map(|i| format!("{i}")).collect::<Vec<String>>().join(" ");
				write!(fmt, "({head}[{info:?}] {out})")
			}
		}
	}
}

impl fmt::Display for S {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		match self {
			S::Atom(token, _) => write!(fmt, "{token}"),
			S::Cons(head, _, rest) => {
				let out = rest.iter().map(|i| format!("{i}")).collect::<Vec<String>>().join(" ");
				write!(fmt, "({head} {out})")
			}
		}
	}
}

pub fn eval<'a>(
	source: &'a str,
	input: Vec<Token<'a>>,
) -> miette::Result<Vec<Stmt>> {
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

#[derive(Debug)]
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
	($parser:expr, $msg:expr) => {
		Err(miette::miette! {
			labels = vec![
				LabeledSpan::at(
					$parser.peek(0).range(),
					"here")
			],
			"Expected {:?}, Found {:?}", $msg, $parser.peek(0),
		}.with_source_code($parser.source.to_owned()))
	}
}

#[instrument(skip(parser))]
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
		_ => error!(parser, "Number"),
	}
}

#[instrument]
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
			trace!("{}", parser.peek(0));
			parser.index += 1;
			Ok(s.to_string())
		}
		TokenType::EOF => error!(eof, parser, "Identifier"),
		_ => error!(parser, "Identifier"),
	}
}

#[instrument(skip(parser))]
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
		return error!(token, parser, format!("Bit specifier between 0..={max_bits}"));
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
	trace!("{token}");
	parser.index += 1;
	Ok(out)
}

fn match_token(
	parser: &mut Parser,
	tt: TokenType,
) -> miette::Result<()> {
	match parser.peek(0).tt {
		t if t != tt => if t == TokenType::EOF {
			error!(parser, tt)
		} else {
			error!(eof, parser, tt)
		}
		_ => {
			trace!("{}", parser.peek(0));
			Ok(parser.index += 1)
		}
	}
}

#[instrument(skip(parser))]
fn ident_typed(
	parser: &mut Parser,
) -> miette::Result<TypedIdent> {
	let id = ident(parser)?;
	let val_type = value_type(parser)?;
	Ok((id, val_type))
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Block(Vec<Stmt>, Option<S>);

impl fmt::Display for Block {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		write!(fmt, "[{}]", self.0.iter().map(|s| format!("{s}")).collect::<Vec<_>>().join(","))?;
		if let Some(s) = &self.1 {
			write!(fmt, "->{s}")?;
		}
		Ok(())
	}
}

impl Block {
	fn new(list: Vec<Stmt>, s: Option<S>) -> Self {
		Self(list, s)
	}

	#[allow(dead_code)]
	fn expr(s: S) -> Self {
		Self(vec![], Some(s))
	}

	#[allow(dead_code)]
	fn stmts(stmts: Vec<Stmt>) -> Self {
		Self(stmts, None)
	}
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Stmt {
	Rec {
		name: String,
		fields: Vec<TypedIdent>,
	},
	Fun {
		name: String,
		params: Vec<TypedIdent>,
		rtype: Option<ValueType>,
		body: Block,
	},
	Var {
		name: String,
		vtype: Option<ValueType>,
		body: S,
	},
	If {
		cond: S,
		bt: Block,
		bf: Option<Block>,
	},
	While {
		cond: S,
		body: Block,
	},
	Assign {
		referent: String,
		body: S,
	}
}

impl fmt::Display for Stmt {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		fn show(list: &[TypedIdent]) -> String {
				list.iter()
					.map(|(s,vt)| format!("{s}:{vt}"))
					.collect::<Vec<_>>()
					.join(",")
		}

		match self {
			Self::Rec { name, fields } => write!(fmt, "(Rec {name} {})", show(fields)),
			Self::Fun { name, params, rtype: Some(rt), body } => write!(fmt, "(Fn {name} ({}) -> {rt} {body})", show(params)),
			Self::Fun { name, params, rtype: None, body } => write!(fmt, "(Fn {name} ({}) {body})", show(params)),
			Self::Var { name, vtype: Some(vt), body } => write!(fmt, "(Var {name}: {vt} = {body})"),
			Self::Var { name, vtype: None, body } => write!(fmt, "(Var {name} = {body})"),
			Self::If { cond, bt, bf: Some(b) } => write!(fmt, "(If {cond} {bt} else {b})"),
			Self::If { cond, bt, bf: None } => write!(fmt, "(If {cond} {bt})"),
			Self::While { cond, body } => write!(fmt, "(While {cond} [{body}])"),
			Self::Assign { referent, body } => write!(fmt, "({referent} = {body})"),
		}
	}
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

/// args := (expr (',' expr)* ','?)?
#[instrument(skip(parser))]
fn args<'a>(parser: &mut Parser<'a,'_>) -> Option<Vec<S>> {
	let first = expr(parser, 0)
		.ok()?;
	let mut out = vec![first];
	while let Some(next) = match_token(parser, TokenType::Comma)
		.and_then(|_| expr(parser, 0))
		.ok()
	{
		out.push(next);
	}
	let _ = match_token(parser, TokenType::Comma);
	Some(out)
}

#[instrument(skip(parser))]
fn prefix_binding_power(
	parser: &mut Parser,
) -> miette::Result<((),u8)> {
	use TokenType as TT;

	match parser.peek(0).tt {
		TT::Plus | TT::Minus => Ok(((),11)),
		TT::Dollar | TT::At => Ok(((),13)),
		TT::Bang => Ok(((),15)),
		_ => error!(parser, "Expected 'Unary Operator'"),
	}
}

#[instrument]
fn infix_binding_power(tt: TokenType) -> Option<(u8,u8)> {
	use TokenType as TT;

	match tt {
		TT::Amp2 | TT::Bar2 | TT::Carrot2 => Some((1,2)),
		TT::Amp1 | TT::Bar1 | TT::Carrot1 => Some((3,4)),
		TT::Eq2 | TT::BangEq |
		TT::RArrow1 | TT::RArrEq |
		TT::LArrow1 | TT::LArrEq => Some((5,6)),
		TT::Plus | TT::Minus => Some((7,8)),
		TT::Star | TT::Slash | TT::Percent | TT::SlashPer |
		TT::LArrow2 | TT::RArrow2 => Some((9,10)),
		TT::Dot => Some((18,17)),
		TT::OParen => Some((20,19)),

		TT::If | TT::Else |
		TT::Fun | TT::Rec | TT::Var |
		TT::While |

		TT::U8 | TT::U16 | TT::U32 |
		TT::S8 | TT::S16 | TT::S32 |
		TT::F16(_) | TT::F32(_) |

		TT::Ident(_) | TT::Number(_) |

		TT::At | TT::Bang |
		TT::Colon | TT::Comma | TT::CBrace | TT::CParen |
		TT::Dollar |
		TT::Eq1 |
		TT::LArrBar |
		TT::OBrace |
		TT::RArrBar | TT::RetArrow |

		TT::EOF => None,
	}
}

#[instrument(skip(parser))]
fn expr<'a>(
	parser: &mut Parser<'a,'_>,
	min_bp: u8,
) -> miette::Result<S> {
	use TokenType as TT;

	let left_token = parser.peek(0).clone();
	let mut lhs = match left_token.tt {
		TT::Ident(s) => {
			parser.index += 1;
			S::Atom(Box::new(Atom::Id(s.to_owned())), left_token.range())
		}
		TT::Number(n) => {
			parser.index += 1;
			S::Atom(Box::new(Atom::Num(n.parse::<i64>().into_diagnostic()?)), left_token.range())
		}

		TT::If => {
			parser.index += 1;
			let start = left_token.range().start;
			let (cond, bt, bf) = expr_if(parser)?;
			let end = parser.peek(-1).range().end;
			S::Atom(Box::new(Atom::If(cond, bt, bf)), start..end)
		}

		TT::OParen => {
			parser.index += 1;
			let lhs = expr(parser, 0)?;
			if TT::CParen != parser.peek(0).tt {
				return error!(parser, ")");
			}
			parser.index += 1;
			lhs
		}

		TT::Plus |
		TT::Minus |
		TT::Dollar |
		TT::At |
		TT::Bang => {
			let ((),r_bp) = prefix_binding_power(parser)?;
			let rhs = expr(parser, r_bp)?;
			parser.index += 1;
			S::Cons(left_token.tt.try_into()?, left_token.range(), vec![rhs])
		}
		TT::EOF => return error!(eof, parser,
			"Identifier, Function Call, or Literal"),
		_ => return error!(parser,
			"Identifier, Function Call, or Literal"),
	};

	loop {
		let op_token = parser.peek(0).clone();
		if matches!(op_token.tt,
			TT::Ident(_) | TT::Number(_) |
			TT::If | TT::Else | TT::While |
			TT::Fun | TT::Rec | TT::Var |
			TT::U8 | TT::U16 | TT::U32 |
			TT::S8 | TT::S16 | TT::S32 |
			TT::F16(_) | TT::F32(_) |
			TT::OBrace |
			TT::Colon | TT::CBrace | TT::CParen |
			TT::EOF) {
			break;
		}

		if TT::OParen == op_token.tt {
			parser.index += 1;
			if TT::CParen == parser.peek(0).tt {
				parser.index += 1;
				lhs = S::Cons(Op::FnCall, op_token.range(), vec![lhs]);
				continue;
			}

			let Some(rhs) = args(parser) else {
				return error!(parser, "Argument List");
			};
			if TT::CParen != parser.peek(0).tt {
				return error!(parser, ")");
			}
			parser.index += 1;
			let mut args = vec![lhs];
			args.extend(rhs);
			lhs = S::Cons(Op::FnCall, op_token.range(), args);
			continue;
		}

		if let Some((l_bp,r_bp)) = infix_binding_power(op_token.tt) {
			if l_bp < min_bp {
				break;
			}

			parser.index += 1;
			let rhs = expr(parser, r_bp)?;
			lhs = S::Cons(op_token.tt.try_into()?, op_token.range(), vec![lhs,rhs]);
			continue;
		}

		break;
	}

	trace!("{lhs}");
	Ok(lhs)
}

/// params := ( ident ':' value_type )*
#[instrument(skip(parser))]
fn params(
	parser: &mut Parser,
) -> miette::Result<Vec<TypedIdent>> {
	let mut out = Vec::new();
	while let Ok(id) = ident(parser) {
		match_token(parser, TokenType::Colon)?;
		let vt = value_type(parser)?;
		out.push((id,vt));
	}
	Ok(out)
}

/// block := '{' stmt* expr? '}'
#[instrument(skip(parser))]
fn block(
	parser: &mut Parser,
) -> miette::Result<Block> {
	match_token(parser, TokenType::OBrace)?;
	let mut body = Vec::new();
	while let Ok(stmt) = statement(parser) {
		body.push(stmt);
	}
	let output = expr(parser, 0).ok();
	match_token(parser, TokenType::CBrace)?;
	Ok(Block::new(body, output))
}

/// rec := 'rec' ident '{' params '}'
#[instrument(skip(parser))]
fn stmt_rec(
	parser: &mut Parser,
) -> miette::Result<Stmt> {
	match_token(parser, TokenType::Rec)?;
	let name = ident(parser)?;
	match_token(parser, TokenType::OBrace)?;
	let fields = params(parser)?;
	match_token(parser, TokenType::CBrace)?;
	Ok(Stmt::Rec { name, fields })
}

/// fn := 'fn' ident '(' params ')' ('->' value_type)? block
#[instrument(skip(parser))]
fn stmt_fn(
	parser: &mut Parser,
) -> miette::Result<Stmt> {
	match_token(parser, TokenType::Fun)?;
	let name = ident(parser)?;
	match_token(parser, TokenType::OParen)?;
	let params = params(parser)?;
	match_token(parser, TokenType::CParen)?;
	let rtype = match_token(parser, TokenType::RetArrow)
		.and_then(|_| value_type(parser))
		.ok();
	let body = block(parser)?;
	Ok(Stmt::Fun { name, params, rtype, body })
}

/// var := 'var' ident (':' value_type)? '=' (block | expr)
#[instrument(skip(parser))]
fn stmt_var<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt> {
	match_token(parser, TokenType::Var)?;
	let name = ident(parser)?;
	let vtype = match_token(parser, TokenType::Colon)
		.and_then(|_| value_type(parser))
		.ok();
	match_token(parser, TokenType::Eq1)?;
	let start = parser.peek(0).range().start;
	let body = block(parser)
		.map(|b| S::Atom(Box::new(Atom::Block(b)), start..parser.peek(-1).range().end))
		.or_else(|_| expr(parser, 0))?;
	Ok(Stmt::Var { name, vtype, body })
}

/// expr_if := expr block ('else' block)?
#[instrument(skip(parser))]
fn expr_if<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<(S,Block,Option<Block>)> {
	let cond = expr(parser, 0)?;
	let bt = block(parser)?;
	let bf = match_token(parser, TokenType::Else)
		.and_then(|_| block(parser))
		.ok();
	Ok((cond, bt, bf))
}

/// if := 'if' expr_if
#[instrument(skip(parser))]
fn stmt_if<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt> {
	match_token(parser, TokenType::If)?;
	let (cond, bt, bf) = expr_if(parser)?;
	Ok(Stmt::If { cond, bt, bf })
}

/// while := 'while' expr block
#[instrument(skip(parser))]
fn stmt_while<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt> {
	match_token(parser, TokenType::While)?;
	let cond = expr(parser, 0)?;
	let body = block(parser)?;
	Ok(Stmt::While { cond, body })
}

/// assign := ident '=' (block | expr)
#[instrument(skip(parser))]
fn stmt_assign<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt> {
	if !matches!(parser.peek(0).tt, TokenType::Ident(_)) || parser.peek(1).tt != TokenType::Eq1 {
		return error!(parser, "Assignment");
	}
	let referent = ident(parser)?;
	parser.index += 1;
	let start = parser.peek(0).range().start;
	let body = block(parser)
		.map(|b| S::Atom(Box::new(Atom::Block(b)), start..parser.peek(-1).range().end))
		.or_else(|_| expr(parser, 0))?;
	Ok(Stmt::Assign { referent, body })
}

/// statement := rec | fn | var | if | while | ident
fn statement<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Stmt> {
	match parser.peek(0).tt {
		TokenType::Rec      => stmt_rec(parser),
		TokenType::Fun      => stmt_fn(parser),
		TokenType::Var      => stmt_var(parser),
		TokenType::If       => stmt_if(parser),
		TokenType::While    => stmt_while(parser),
		TokenType::Ident(_) => stmt_assign(parser),
		_ => error!(parser, "Statement"),
	}
}

/// program := statement*
fn program<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Vec<Stmt>> {
	let mut program = Vec::default();
	while parser.peek(0).tt != TokenType::EOF {
		program.push(statement(parser)?);
	}
	Ok(program)
}

#[cfg(test)]
mod test {
	use crate::parser::{Block, S, Stmt, TypedIdent, ValueType as VT};
	use super::{Atom, Op};

	fn atom(a: Atom) -> S {
		S::Atom(Box::new(a), 0..0)
	}

	fn num(n: i64) -> S {
		atom(Atom::Num(n))
	}

	fn ident(s: &str) -> S {
		atom(Atom::Id(s.to_owned()))
	}

	fn cons<'a>(op: Op, s: &[S]) -> S {
		S::Cons(op, 0..0, s.into_iter().cloned().collect::<Vec<_>>())
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

	fn var_s(
		name: &str,
		vtype: Option<VT>,
		body: S,
	) -> Stmt {
		Stmt::Var {
			name: name.to_string(),
			vtype,
			body,
		}
	}

	fn fn_s(
		name: &str,
		params: &[TypedIdent],
		rtype: Option<VT>,
		body: &[Stmt],
		output: Option<S>,
	) -> Stmt {
		Stmt::Fun {
			name: name.to_string(),
			params: params.to_vec(),
			rtype,
			body: Block::new(body.to_vec(),	output),
		}
	}

	fn rec_s(
		name: &str,
		fields: &[TypedIdent],
	) -> Stmt {
		Stmt::Rec {
			name: name.to_string(),
			fields: fields.to_vec(),
		}
	}

	fn if_s(
		cond: S,
		bt: Block,
		bf: Option<Block>,
	) -> Stmt {
		Stmt::If { cond, bt, bf }
	}

	fn while_s(
		cond: S,
		body: Block,
	) -> Stmt {
		Stmt::While { cond, body }
	}

	fn assign_s(
		referent: &str,
		body: S,
	) -> Stmt {
		Stmt::Assign {
			referent: referent.to_string(),
			body,
		}
	}

	#[test]
	fn precedence() -> miette::Result<()> {
		expr_test("1 + 2 * 3", cons(Op::Add, &[
			num(1),
			cons(Op::Mul, &[
				num(2),
				num(3),
			]),
		]))?;
		expr_test("1 * 2 + 3", cons(Op::Add, &[
			cons(Op::Mul, &[
				num(1),
				num(2),
			]),
			num(3),
		]))
	}

	#[test]
	fn parentheses() -> miette::Result<()> {
		expr_test("1 * (2 + 3)", cons(Op::Mul, &[
			num(1),
			cons(Op::Add, &[num(2), num(3)]),
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
			var_s("a", None, num(0)),
		])
	}

	#[test]
	fn var_stmt_expr() -> miette::Result<()> {
		parse_test("var a = 3 * 2 + 1", &[
			var_s("a", None, cons(Op::Add, &[
				cons(Op::Mul, &[
					num(3),
					num(2),
				]),
				num(1),
			])),
		])
	}

	#[test]
	fn var_stmt_vtype() -> miette::Result<()> {
		parse_test("var a: u8 = 0", &[
			var_s("a", Some(VT::U8), num(0)),
		])
	}

	#[test]
	fn var_stmt_udt_simple() -> miette::Result<()> {
		parse_test("var a = b", &[
			var_s("a", None, ident("b")),
		])
	}

	#[test]
	fn var_stmt_udt_fncall_empty() -> miette::Result<()> {
		parse_test("var a = b()", &[
			var_s("a", None, cons(Op::FnCall, &[ident("b")]))
		])
	}

	#[test]
	fn var_stmt_udt_fncall_single() -> miette::Result<()> {
		parse_test("var a = b(c)", &[
			var_s("a", None, cons(Op::FnCall, &[ident("b"), ident("c")]))
		])
	}

	#[test]
	fn var_stmt_udt_fncall_multi() -> miette::Result<()> {
		parse_test("var a = b(c, d + e)", &[
			var_s("a", None, cons(Op::FnCall, &[
				ident("b"),
				ident("c"),
				cons(Op::Add, &[ident("d"), ident("e")]),
			]))
		])
	}

	#[test]
	fn fn_stmt() -> miette::Result<()> {
		parse_test("fn a() {}", &[
			fn_s("a", &[], None, &[], None),
		])
	}

	#[test]
	fn fn_stmt_params() -> miette::Result<()> {
		parse_test("fn a(b:u8 c:s16 d:fw6 e:fl10) {}", &[
			fn_s("a", &[
				("b".to_string(), VT::U8),
				("c".to_string(), VT::S16),
				("d".to_string(), VT::F16(6)),
				("e".to_string(), VT::F32(10)),
			], None, &[], None),
		])
	}

	#[test]
	fn fn_stmt_rtype_simple() -> miette::Result<()> {
		parse_test("fn a() -> u8 {}", &[
			fn_s("a", &[], Some(VT::U8), &[], None)
		])
	}

	#[test]
	fn fn_stmt_rtype_udt() -> miette::Result<()> {
		parse_test("fn a() -> b {}", &[
			fn_s("a", &[], Some(VT::UDT("b".to_string())), &[], None)
		])
	}

	#[test]
	fn fn_stmt_body() -> miette::Result<()> {
		parse_test("fn a() {
			var b = 1
			var c = 2
			b + c
		}", &[
			fn_s("a", &[], None, &[
				var_s("b", None, num(1)),
				var_s("c", None, num(2)),
			], Some(cons(Op::Add, &[ident("b"), ident("c")])))
		])
	}

	#[test]
	fn rec_stmt() -> miette::Result<()> {
		parse_test("rec a{}", &[
			rec_s("a", &[]),
		])
	}

	#[test]
	fn rec_stmt_fields() -> miette::Result<()> {
		parse_test("rec vec{x:fl y:fl}", &[
			rec_s("vec", &[
				("x".to_string(), VT::F32(16)),
				("y".to_string(), VT::F32(16)),
			])
		])
	}

	#[test]
	fn field_access() -> miette::Result<()> {
		parse_test("var a = b.c.d", &[
			var_s("a", None, cons(Op::Accessor, &[
				ident("b"),
				cons(Op::Accessor, &[
					ident("c"),
					ident("d"),
				]),
			]))
		])
	}

	#[test]
	fn if_stmt() -> miette::Result<()> {
		parse_test("if a > b {a}", &[
			if_s(
				cons(Op::CmpGT, &[ident("a"), ident("b")]),
				Block::expr(ident("a")),
				None,
			)
		])
	}

	#[test]
	fn if_else_stmt() -> miette::Result<()> {
		parse_test("if a < b {a} else {b}", &[
			if_s(
				cons(Op::CmpLT, &[ident("a"), ident("b")]),
				Block::expr(ident("a")),
				Some(Block::expr(ident("b"))),
			)
		])
	}

	#[test]
	fn while_stmt() -> miette::Result<()> {
		parse_test("while a == b {b}", &[
			while_s(
				cons(Op::CmpEq, &[ident("a"), ident("b")]),
				Block::expr(ident("b")),
			)
		])
	}

	#[test]
	fn assign_stmt() -> miette::Result<()> {
		parse_test("a = 3", &[
			assign_s("a", num(3)),
		])
	}
}

