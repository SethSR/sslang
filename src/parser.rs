
use std::fmt;
use std::ops::Range;
use std::rc::Rc;

use miette::{IntoDiagnostic, LabeledSpan, WrapErr};
use tracing::{instrument, trace};

use crate::tokens::{Token, TokenType};

pub(crate) type TypedIdent = (Rc<str>, ValueType);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unsigned {
	Top, U8, U16, U32, Bot,
}

impl fmt::Display for Unsigned {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Top => write!(f, "U::Top"),
			Self::U8 => write!(f, "u8"),
			Self::U16 => write!(f, "u16"),
			Self::U32 => write!(f, "u32"),
			Self::Bot => write!(f, "U::Bot"),
		}
	}
}

impl Unsigned {
	fn meet(&self, rhs: &Self) -> Self {
		match (self, rhs) {
			(Self::Bot, _) | (_, Self::Bot) => Self::Bot,
			(Self::U32, _) | (_, Self::U32) => Self::U32,
			(Self::U16, _) | (_, Self::U16) => Self::U16,
			(Self::U8,  _) | (_, Self::U8)  => Self::U8,
			(Self::Top, Self::Top) => Self::Top,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Signed {
	Top, S8, S16, S32, Bot,
}

impl fmt::Display for Signed {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Top => write!(f, "S::Top"),
			Self::S8 => write!(f, "s8"),
			Self::S16 => write!(f, "s16"),
			Self::S32 => write!(f, "s32"),
			Self::Bot => write!(f, "S::Bot"),
		}
	}
}

impl Signed {
	fn meet(&self, rhs: &Self) -> Self {
		match (self, rhs) {
			(Self::Bot, _) | (_, Self::Bot) => Self::Bot,
			(Self::S32, _) | (_, Self::S32) => Self::S32,
			(Self::S16, _) | (_, Self::S16) => Self::S16,
			(Self::S8,  _) | (_, Self::S8)  => Self::S8,
			(Self::Top, Self::Top) => Self::Top,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Int {
	Top,
	Unsigned(Unsigned),
	Signed(Signed),
	Bot,
}

impl fmt::Display for Int {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Top => write!(f, "I::Top"),
			Self::Unsigned(u) => write!(f, "{u}"),
			Self::Signed(s) => write!(f, "{s}"),
			Self::Bot => write!(f, "I::Bot"),
		}
	}
}

impl Int {
	fn meet(&self, rhs: &Self) -> Self {
		match (self, rhs) {
			(Self::Bot, _) | (_, Self::Bot) |
			(Self::Signed(_), Self::Unsigned(_)) |
			(Self::Unsigned(_), Self::Signed(_)) => Self::Bot,

			(Self::Unsigned(a), Self::Unsigned(b)) => Self::Unsigned(a.meet(b)),
			(Self::Unsigned(int), Self::Top) => Self::Unsigned(*int),
			(Self::Top, Self::Unsigned(int)) => Self::Unsigned(*int),

			(Self::Signed(a), Self::Signed(b)) => Self::Signed(a.meet(b)),
			(Self::Signed(int), Self::Top) => Self::Signed(*int),
			(Self::Top, Self::Signed(int)) => Self::Signed(*int),

			(Self::Top, Self::Top) => Self::Top,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fix {
	Top,
	F16(u8), // 16-bit fixed point with <u8> integer bits
	F32(u8), // 32-bit fixed point with <u8> integer bits
	Bot,
}

impl Fix {
	fn meet(&self, _rhs: &Self) -> Self {
		todo!()
	}
}

// TODO - srenshaw - Change 'fw' and 'fd' to 'f' and 'd'?

impl fmt::Display for Fix {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Top => write!(f, "F::Top"),
			Self::F16(n) => write!(f, "fw{n}"),
			Self::F32(n) => write!(f, "fd{n}"),
			Self::Bot => write!(f, "F::Bot"),
		}
	}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ValueType {
	/// Bottom type
	Unit,
	Int(Int),
	Fix(Fix),
	UDT(String), // User Defined Type
	// Used for function calls, we'll resolve these later
	/// Any type - also used for unknown/unresolved types during compilation
	Any,
}

impl ValueType {
	pub(crate) fn to_u8() -> Self {
		Self::Int(Int::Unsigned(Unsigned::U8))
	}

	pub(crate) fn to_u16() -> Self {
		Self::Int(Int::Unsigned(Unsigned::U16))
	}

	pub(crate) fn to_u32() -> Self {
		Self::Int(Int::Unsigned(Unsigned::U32))
	}

	pub(crate) fn to_s8() -> Self {
		Self::Int(Int::Signed(Signed::S8))
	}

	pub(crate) fn to_s16() -> Self {
		Self::Int(Int::Signed(Signed::S16))
	}

	pub(crate) fn to_s32() -> Self {
		Self::Int(Int::Signed(Signed::S32))
	}

	pub(crate) fn to_f16(bits: u8) -> Self {
		Self::Fix(Fix::F16(bits))
	}

	pub(crate) fn to_f32(bits: u8) -> Self {
		Self::Fix(Fix::F32(bits))
	}

	pub(crate) fn meet(&self, rhs: &Self) -> Self {
		match (self, rhs) {
			(Self::Unit, _) | (_, Self::Unit) => Self::Unit,

			(Self::Int(a), Self::Int(b)) => Self::Int(a.meet(b)),
			(Self::Int(_), Self::Fix(_)) => Self::Unit,
			(Self::Int(_), Self::UDT(_)) => Self::Unit,
			(Self::Int(int), Self::Any) => Self::Int(*int),

			(Self::Fix(_), Self::Int(_)) => Self::Unit,
			(Self::Fix(a), Self::Fix(b)) => Self::Fix(a.meet(b)),
			(Self::Fix(_), Self::UDT(_)) => Self::Unit,
			(Self::Fix(fix), Self::Any) => Self::Fix(*fix),

			(Self::UDT(_), Self::Int(_)) => Self::Unit,
			(Self::UDT(_), Self::Fix(_)) => Self::Unit,
			(Self::UDT(_), Self::UDT(_)) => Self::Unit,
			(Self::UDT(udt), Self::Any) => Self::UDT(udt.clone()),

			(Self::Any, Self::Int(_)) => Self::Unit,
			(Self::Any, Self::Fix(_)) => Self::Unit,
			(Self::Any, Self::UDT(_)) => Self::Unit,
			(Self::Any, Self::Any) => Self::Any,
		}
	}
}

impl fmt::Display for ValueType {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		use ValueType as VT;

		match self {
			VT::Int(int) => write!(fmt, "{int}"),
			VT::Fix(fix) => write!(fmt, "{fix}"),
			VT::UDT(s) => write!(fmt, "{s}"),
			VT::Unit => write!(fmt, "()"),
			VT::Any => write!(fmt, "??"),
		}
	}
}

type TokenInfo = Range<usize>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOp {
	/// '@'
	Deref,
	/// '-'
	Neg,
	/// '!'
	Not,
	/// '+'
	Pos,
	/// '$'
	Ref,
}

impl TryFrom<TokenType> for UnaryOp {
	type Error = miette::Report;
	fn try_from(tt: TokenType) -> Result<Self, Self::Error> {
		match tt {
			TokenType::At       => Ok(UnaryOp::Deref),
			TokenType::Bang     => Ok(UnaryOp::Not),
			TokenType::Dollar   => Ok(UnaryOp::Ref),
			TokenType::Minus    => Ok(UnaryOp::Neg),
			TokenType::Plus     => Ok(UnaryOp::Pos),
			_ => Err(miette::miette! {
				"{tt:?} is not a unary operator"
			})
		}
	}
}

impl TryFrom<&TokenType> for UnaryOp {
	type Error = miette::Report;
	fn try_from(tt: &TokenType) -> Result<Self, Self::Error> {
		match tt {
			TokenType::At       => Ok(UnaryOp::Deref),
			TokenType::Bang     => Ok(UnaryOp::Not),
			TokenType::Dollar   => Ok(UnaryOp::Ref),
			TokenType::Minus    => Ok(UnaryOp::Neg),
			TokenType::Plus     => Ok(UnaryOp::Pos),
			_ => Err(miette::miette! {
				"{tt:?} is not a unary operator"
			})
		}
	}
}

impl fmt::Display for UnaryOp {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		let op = match self {
			UnaryOp::Deref => "@",
			UnaryOp::Neg   => "-",
			UnaryOp::Not   => "!",
			UnaryOp::Pos   => "+",
			UnaryOp::Ref   => "$",
		};
		write!(fmt, "{op}")
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
	/// '.'
	Accessor,
	/// '+'
	Add,
	/// '&'
	AndB,
	/// '&&'
	AndL,
	/// '='
	Assign,
	/// '=='
	CmpEq,
	/// '>='
	CmpGE,
	/// '>'
	CmpGT,
	/// '<='
	CmpLE,
	/// '<'
	CmpLT,
	/// '!='
	CmpNE,
	/// ','
	Comma,
	/// '/'
	Div,
	/// '/%'
	DivMod,
	/// '<|'
	LRot,
	/// '<<'
	LShift,
	/// '%'
	Mod,
	/// '*'
	Mul,
	/// '|'
	OrB,
	/// '||'
	OrL,
	/// '|>'
	RRot,
	/// '>>'
	RShift,
	/// '-'
	Sub,
	/// '^'
	XorB,
	/// '^^'
	XorL,
}

impl TryFrom<TokenType> for BinaryOp {
	type Error = miette::Report;
	fn try_from(tt: TokenType) -> Result<Self, Self::Error> {
		match tt {
			TokenType::Amp1     => Ok(BinaryOp::AndB),
			TokenType::Amp2     => Ok(BinaryOp::AndL),
			TokenType::BangEq   => Ok(BinaryOp::CmpNE),
			TokenType::Bar1     => Ok(BinaryOp::OrB),
			TokenType::Bar2     => Ok(BinaryOp::OrL),
			TokenType::Carrot1  => Ok(BinaryOp::XorB),
			TokenType::Carrot2  => Ok(BinaryOp::XorL),
			TokenType::Comma    => Ok(BinaryOp::Comma),
			TokenType::Dot      => Ok(BinaryOp::Accessor),
			TokenType::Eq1      => Ok(BinaryOp::Assign),
			TokenType::Eq2      => Ok(BinaryOp::CmpEq),
			TokenType::LArrow1  => Ok(BinaryOp::CmpLT),
			TokenType::LArrow2  => Ok(BinaryOp::LShift),
			TokenType::LArrBar  => Ok(BinaryOp::LRot),
			TokenType::LArrEq   => Ok(BinaryOp::CmpLE),
			TokenType::Minus    => Ok(BinaryOp::Sub),
			TokenType::Percent  => Ok(BinaryOp::Mod),
			TokenType::Plus     => Ok(BinaryOp::Add),
			TokenType::RArrow1  => Ok(BinaryOp::CmpGT),
			TokenType::RArrow2  => Ok(BinaryOp::RShift),
			TokenType::RArrBar  => Ok(BinaryOp::RRot),
			TokenType::RArrEq   => Ok(BinaryOp::CmpGE),
			TokenType::Slash    => Ok(BinaryOp::Div),
			TokenType::SlashPer => Ok(BinaryOp::DivMod),
			TokenType::Star     => Ok(BinaryOp::Mul),
			_ => Err(miette::miette! {
				"{tt:?} is not an operator"
			})
		}
	}
}

impl TryFrom<&TokenType> for BinaryOp {
	type Error = miette::Report;
	fn try_from(tt: &TokenType) -> Result<Self, Self::Error> {
		match tt {
			TokenType::Amp1     => Ok(BinaryOp::AndB),
			TokenType::Amp2     => Ok(BinaryOp::AndL),
			TokenType::BangEq   => Ok(BinaryOp::CmpNE),
			TokenType::Bar1     => Ok(BinaryOp::OrB),
			TokenType::Bar2     => Ok(BinaryOp::OrL),
			TokenType::Carrot1  => Ok(BinaryOp::XorB),
			TokenType::Carrot2  => Ok(BinaryOp::XorL),
			TokenType::Comma    => Ok(BinaryOp::Comma),
			TokenType::Dot      => Ok(BinaryOp::Accessor),
			TokenType::Eq1      => Ok(BinaryOp::Assign),
			TokenType::Eq2      => Ok(BinaryOp::CmpEq),
			TokenType::LArrow1  => Ok(BinaryOp::CmpLT),
			TokenType::LArrow2  => Ok(BinaryOp::LShift),
			TokenType::LArrBar  => Ok(BinaryOp::LRot),
			TokenType::LArrEq   => Ok(BinaryOp::CmpLE),
			TokenType::Minus    => Ok(BinaryOp::Sub),
			TokenType::Percent  => Ok(BinaryOp::Mod),
			TokenType::Plus     => Ok(BinaryOp::Add),
			TokenType::RArrow1  => Ok(BinaryOp::CmpGT),
			TokenType::RArrow2  => Ok(BinaryOp::RShift),
			TokenType::RArrBar  => Ok(BinaryOp::RRot),
			TokenType::RArrEq   => Ok(BinaryOp::CmpGE),
			TokenType::Slash    => Ok(BinaryOp::Div),
			TokenType::SlashPer => Ok(BinaryOp::DivMod),
			TokenType::Star     => Ok(BinaryOp::Mul),
			_ => Err(miette::miette! {
				"{tt:?} is not an operator"
			})
		}
	}
}

impl fmt::Display for BinaryOp {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		let op = match self {
			BinaryOp::Accessor => ".",
			BinaryOp::Add      => "+",
			BinaryOp::AndB     => "&",
			BinaryOp::AndL     => "&&",
			BinaryOp::Assign   => "=",
			BinaryOp::CmpEq    => "==",
			BinaryOp::CmpGE    => ">=",
			BinaryOp::CmpGT    => ">",
			BinaryOp::CmpLE    => "<=",
			BinaryOp::CmpLT    => "<",
			BinaryOp::CmpNE    => "!=",
			BinaryOp::Comma    => ",",
			BinaryOp::Div      => "/",
			BinaryOp::DivMod   => "/%",
			BinaryOp::LRot     => "<|",
			BinaryOp::LShift   => "<<",
			BinaryOp::Mod      => "%",
			BinaryOp::Mul      => "*",
			BinaryOp::OrB      => "|",
			BinaryOp::OrL      => "||",
			BinaryOp::RRot     => "|>",
			BinaryOp::RShift   => ">>",
			BinaryOp::Sub      => "-",
			BinaryOp::XorB     => "^",
			BinaryOp::XorL     => "^^",
		};
		write!(fmt, "{op}")
	}
}

// TODO - srenshaw - Move node-types from Expr into Node.
// TODO - srenshaw - Create a more robust type system. (Probably based on github.com/SeaOfNodes)

#[derive(Debug, Clone)]
pub(crate) struct Node {
	pub(crate) info: TokenInfo,
	pub(crate) kind: ValueType,
	pub(crate) expr: Box<Expr>,
}

impl PartialEq for Node {
	fn eq(&self, rhs: &Self) -> bool {
		self.expr == rhs.expr
	}
}

impl Node {
	pub(crate) fn new(expr: Expr, kind: ValueType, info: TokenInfo) -> Self {
		Self { info, kind, expr: expr.into() }
	}
}

impl fmt::Display for Node {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", self.expr)
	}
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
	Num(i64),
	Id(Rc<str>),
	Block(Vec<Node>),
	Rec {
		name: Rc<str>,
		fields: Vec<TypedIdent>,
	},
	Fun {
		name: Rc<str>,
		params: Vec<TypedIdent>,
		rtype: ValueType,
		body: Vec<Node>,
	},
	Var {
		name: Rc<str>,
		body: Node,
	},
	If {
		cond: Node,
		bt: Vec<Node>,
		bf: Vec<Node>,
	},
	While {
		cond: Node,
		body: Vec<Node>,
	},
	Assign {
		name: Rc<str>,
		body: Node,
	},
	Unary {
		op: UnaryOp,
		rhs: Node,
	},
	Binary {
		op: BinaryOp,
		lhs: Node,
		rhs: Node,
	},
	FnCall {
		name: Rc<str>,
		args: Vec<Node>,
	},
}

impl Node {
	pub(crate) fn new_block(b: Vec<Node>, info: TokenInfo) -> Self {
		Self::new(Expr::Block(b), ValueType::Unit, info)
	}

	pub(crate) fn new_rec(name: Rc<str>, fields: Vec<TypedIdent>, info: TokenInfo) -> Self {
		let udt = name.to_string();
		Self::new(Expr::Rec { name, fields }, ValueType::UDT(udt), info)
	}

	pub(crate) fn new_fun(
		name: Rc<str>,
		params: Vec<TypedIdent>,
		rtype: ValueType,
		body: Vec<Node>,
		info: TokenInfo,
	) -> Self {
		let kind = rtype.clone();
		Self::new(Expr::Fun { name, params, rtype, body }, kind, info)
	}

	pub(crate) fn new_var(
		name: Rc<str>,
		vtype: ValueType,
		body: Node,
		info: TokenInfo,
	) -> Self {
		Self::new(Expr::Var { name, body }, vtype, info)
	}

	pub(crate) fn new_if(cond: Node, bt: Vec<Node>, bf: Vec<Node>, info: TokenInfo) -> Self {
		let kind = match (bt.last(), bf.last()) {
			(Some(true_node), Some(false_node)) => true_node.kind.meet(&false_node.kind),
			_ => ValueType::Unit,
		};
		Self::new(Expr::If { cond, bt, bf }, kind, info)
	}

	pub(crate) fn new_while(cond: Node, body: Vec<Node>, info: TokenInfo) -> Self {
		Self::new(Expr::While { cond, body }, ValueType::Unit, info)
	}

	pub(crate) fn new_assign(name: Rc<str>, body: Node, info: TokenInfo) -> Self {
		let kind = body.kind.clone();
		Self::new(Expr::Assign { name, body }, kind, info)
	}

	pub(crate) fn new_unary(op: UnaryOp, rhs: Node, info: TokenInfo) -> Self {
		let kind = rhs.kind.clone();
		Self::new(Expr::Unary { op, rhs }, kind, info)
	}

	pub(crate) fn new_binary(op: BinaryOp, lhs: Node, rhs: Node, info: TokenInfo) -> Self {
		let kind = lhs.kind.meet(&rhs.kind);
		Self::new(Expr::Binary { op, lhs, rhs }, kind, info)
	}

	pub(crate) fn new_call(name: Rc<str>, args: Vec<Node>, info: TokenInfo) -> Self {
		Self::new(Expr::FnCall { name, args }, ValueType::Any, info)
	}
}

impl fmt::Display for Expr {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		fn show<T>(list: &[T], f: fn(&T) -> String) -> String {
				list.iter()
					.map(f)
					.collect::<Vec<_>>()
					.join(", ")
		}

		match self {
			Expr::Rec { name, fields }    => write!(fmt, "(rec {name} {})",
				show(fields, |(s,vt)| format!("{s}: {vt}"))),
			Expr::While { cond, body }    => write!(fmt, "(while {cond} {body:?})"),
			Expr::Assign { name, body }   => write!(fmt, "({name} = {body})"),
			Expr::Num(n)                  => write!(fmt, "{n}"),
			Expr::Id(s)                   => write!(fmt, "{s}"),
			Expr::Block(b)                => write!(fmt, "{b:?}"),
			Expr::Unary { op, rhs }       => write!(fmt, "({op} {rhs})"),
			Expr::Binary { op, lhs, rhs } => write!(fmt, "({op} {lhs} {rhs})"),
			Expr::Fun { name, params, rtype, body } => write!(fmt, "(fn {name} ({}) -> {rtype} {body:?})",
				show(params, |(s,vt)| format!("{s}: {vt}"))),
			Expr::Var { name, body } => write!(fmt, "(var {name} = {body}"),
			Expr::If { cond, bt, bf } => write!(fmt, "(if {cond} {bt:?} {bf:?})"),
			Expr::FnCall { name, args } => write!(fmt, "{name}({args:?})"),
		}
	}
}

pub fn eval(
	source: &str,
	input: Vec<Token>,
) -> miette::Result<Vec<Node>> {
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
struct Parser<'a,'b> {
	input: &'a [Token],
	source: &'b str,
	index: usize,
}

impl Parser<'_,'_> {
	fn peek(&self, offset: isize) -> &Token {
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
	match parser.peek(0).tt.clone() {
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
) -> miette::Result<Rc<str>> {
	match parser.peek(0).tt.clone() {
		TokenType::Ident(s) => {
			trace!("{}", parser.peek(0));
			parser.index += 1;
			Ok(s)
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
		TokenType::U8  => ValueType::to_u8(),
		TokenType::U16 => ValueType::to_u16(),
		TokenType::U32 => ValueType::to_u32(),
		TokenType::S8  => ValueType::to_s8(),
		TokenType::S16 => ValueType::to_s16(),
		TokenType::S32 => ValueType::to_s32(),
		TokenType::F16(_) => parse_fixed_point(parser, "fw", 16).map(ValueType::to_f16)?,
		TokenType::F32(_) => parse_fixed_point(parser, "fd", 32).map(ValueType::to_f32)?,
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
	match parser.peek(0).tt.clone() {
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

/// args := (expr (',' expr)* ','?)?
#[instrument(skip(parser))]
fn args<'a>(parser: &mut Parser<'a,'_>) -> Option<Vec<Node>> {
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
) -> miette::Result<u8> {
	use TokenType as TT;

	match parser.peek(0).tt {
		TT::Plus | TT::Minus => Ok(11),
		TT::Dollar | TT::At => Ok(13),
		TT::Bang => Ok(15),
		_ => error!(parser, "Expected 'Unary Operator'"),
	}
}

#[instrument]
fn infix_binding_power(tt: &TokenType) -> Option<(u8,u8)> {
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

/// statement := rec | fn | var | if | while | ident
fn statement<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Node> {
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

/// expr_if := expr block ('else' block)?
#[instrument(skip(parser))]
fn expr_if<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<(Node,Vec<Node>,Vec<Node>)> {
	let cond = expr(parser, 0)?;
	let bt = block(parser)?;
	let bf = match_token(parser, TokenType::Else)
		.and_then(|_| block(parser))
		.unwrap_or_default();
	Ok((cond, bt, bf))
}

/// if := 'if' expr_if
#[instrument(skip(parser))]
fn stmt_if<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Node> {
	let start = parser.peek(0).range().start;
	match_token(parser, TokenType::If)?;
	let (cond, bt, bf) = expr_if(parser)?;
	let end = parser.peek(-1).range().end;
	Ok(Node::new_if(cond, bt, bf, start..end))
}

#[instrument(skip(parser))]
fn expr<'a>(
	parser: &mut Parser<'a,'_>,
	min_bp: u8,
) -> miette::Result<Node> {
	use TokenType as TT;

	let left_token = parser.peek(0).clone();
	let mut lhs = match left_token.tt {
		TT::Rec => stmt_rec(parser)?,
		TT::Fun => stmt_fn(parser)?,
		TT::Var => stmt_var(parser)?,
		TT::While => stmt_while(parser)?,

		TT::Ident(ref s) => {
			parser.index += 1;
			Node::new(Expr::Id(s.to_owned()), ValueType::Any, left_token.range())
		}

		TT::Number(ref n) => {
			parser.index += 1;
			Node::new(
				Expr::Num(n.parse::<i64>().into_diagnostic()?),
				ValueType::Int(Int::Bot),
				left_token.range(),
			)
		}

		TT::If => {
			let start = left_token.range().start;
			parser.index += 1;
			let (cond, bt, bf) = expr_if(parser)?;
			let end = parser.peek(-1).range().end;
			Node::new_if(cond, bt, bf, start..end)
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
			let r_bp = prefix_binding_power(parser)?;
			parser.index += 1;
			let rhs = expr(parser, r_bp)?;
			Node::new_unary((&left_token.tt).try_into()?, rhs, left_token.range())
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
			let Expr::Id(name) = *lhs.expr else {
				return error!(parser, "Identifier");
			};

			parser.index += 1;
			if TT::CParen == parser.peek(0).tt {
				parser.index += 1;
				lhs = Node::new_call(name, vec![], lhs.info.start..op_token.range().end);
				continue;
			}

			let Some(args) = args(parser) else {
				return error!(parser, "Argument List");
			};
			if TT::CParen != parser.peek(0).tt {
				return error!(parser, ")");
			}
			parser.index += 1;
			lhs = Node::new_call(name, args, lhs.info.start..op_token.range().end);
			continue;
		}

		if let Some((l_bp,r_bp)) = infix_binding_power(&op_token.tt) {
			if l_bp < min_bp {
				break;
			}

			parser.index += 1;
			let op: BinaryOp = (&op_token.tt).try_into()?;
			let rhs = expr(parser, r_bp)?;
			lhs = Node::new_binary(op, lhs, rhs, op_token.range());
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
) -> miette::Result<Vec<Node>> {
	match_token(parser, TokenType::OBrace)?;
	let mut body = Vec::new();
	while let Ok(stmt) = statement(parser) {
		body.push(stmt);
	}
	if let Ok(output) = expr(parser, 0) {
		body.push(output);
	}
	match_token(parser, TokenType::CBrace)?;
	Ok(body)
}

/// rec := 'rec' ident '{' params '}'
#[instrument(skip(parser))]
fn stmt_rec(
	parser: &mut Parser,
) -> miette::Result<Node> {
	let start = parser.peek(0).range().start;
	match_token(parser, TokenType::Rec)?;
	let name = ident(parser)?;
	match_token(parser, TokenType::OBrace)?;
	let fields = params(parser)?;
	match_token(parser, TokenType::CBrace)?;
	let end = parser.peek(-1).range().end;
	Ok(Node::new_rec(name, fields, start..end))
}

/// fn := 'fn' ident '(' params ')' ('->' value_type)? block
#[instrument(skip(parser))]
fn stmt_fn(
	parser: &mut Parser,
) -> miette::Result<Node> {
	let start = parser.peek(0).range().start;
	match_token(parser, TokenType::Fun)?;
	let name = ident(parser)?;
	match_token(parser, TokenType::OParen)?;
	let params = params(parser)?;
	match_token(parser, TokenType::CParen)?;
	let rtype = match_token(parser, TokenType::RetArrow)
		.and_then(|_| value_type(parser))
		.unwrap_or(ValueType::Unit);
	let body = block(parser)?;
	let end = parser.peek(-1).range().end;
	Ok(Node::new_fun(name, params, rtype, body, start..end))
}

/// var := 'var' ident (':' value_type)? '=' (block | expr)
#[instrument(skip(parser))]
fn stmt_var<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Node> {
	let start = parser.peek(0).range().start;
	match_token(parser, TokenType::Var)?;
	let name = ident(parser)?;
	let vtype = match_token(parser, TokenType::Colon)
		.and_then(|_| value_type(parser))
		.unwrap_or(ValueType::Unit);
	match_token(parser, TokenType::Eq1)?;
	let body_start = parser.peek(0).range().start;
	let body = block(parser)
		.map(|b| Node::new_block(b, body_start..parser.peek(-1).range().end))
		.or_else(|_| expr(parser, 0))?;
	let end = parser.peek(-1).range().end;
	Ok(Node::new_var(name, vtype, body, start..end))
}

/// while := 'while' expr block
#[instrument(skip(parser))]
fn stmt_while<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Node> {
	let start = parser.peek(0).range().start;
	match_token(parser, TokenType::While)?;
	let cond = expr(parser, 0)?;
	let body = block(parser)?;
	let end = parser.peek(-1).range().end;
	Ok(Node::new_while(cond, body, start..end))
}

/// assign := ident '=' (block | expr)
#[instrument(skip(parser))]
fn stmt_assign<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Node> {
	let start = parser.peek(0).range().start;
	if !matches!(parser.peek(0).tt, TokenType::Ident(_)) || parser.peek(1).tt != TokenType::Eq1 {
		return error!(parser, "Assignment");
	}
	let name = ident(parser)?;
	parser.index += 1;
	let body_start = parser.peek(0).range().start;
	let body = block(parser)
		.map(|b| Node::new_block(b, body_start..parser.peek(-1).range().end))
		.or_else(|_| expr(parser, 0))?;
	let end = parser.peek(-1).range().end;
	Ok(Node::new_assign(name, body, start..end))
}

/// program := statement*
fn program<'a>(
	parser: &mut Parser<'a,'_>,
) -> miette::Result<Vec<Node>> {
	let mut program = Vec::default();
	while parser.peek(0).tt != TokenType::EOF {
		program.push(statement(parser)?);
	}
	Ok(program)
}

#[cfg(test)]
mod test {
	use crate::parser::{
		BinaryOp,
		Expr,
		Node,
		TypedIdent,
		UnaryOp,
		ValueType as VT,
	};

	fn num(n: i64) -> Node {
		Node {
			kind: VT::Any,
			expr: Expr::Num(n).into(),
			info: 0..0,
		}
	}

	fn ident(s: &str) -> Node {
		Node {
			kind: VT::Any,
			expr: Expr::Id(s.into()).into(),
			info: 0..0,
		}
	}

	fn unary(op: UnaryOp, rhs: Node) -> Node {
		Node {
			kind: rhs.kind.clone(),
			expr: Expr::Unary { op, rhs }.into(),
			info: 0..0,
		}
	}

	fn binary(op: BinaryOp, lhs: Node, rhs: Node) -> Node {
		Node {
			kind: lhs.kind.meet(&rhs.kind),
			expr: Expr::Binary { op, lhs, rhs }.into(),
			info: 0..0,
		}
	}

	fn fn_call(name: &str, s: &[Node]) -> Node {
		Node {
			kind: VT::Any,
			expr: Expr::FnCall {
				name: name.into(),
				args: s.to_vec(),
			}.into(),
			info: 0..0,
		}
	}

	fn expr_test(input: &str, s: Node) -> miette::Result<()> {
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
		vtype: VT,
		body: Node,
	) -> Node {
		Node {
			kind: vtype,
			expr: Expr::Var {
				name: name.into(),
				body,
			}.into(),
			info: 0..0,
		}
	}

	fn fn_s(
		name: &str,
		params: &[TypedIdent],
		rtype: VT,
		body: &[Node],
	) -> Node {
		Node {
			kind: VT::Any,
			expr: Expr::Fun {
				name: name.into(),
				params: params.to_vec(),
				rtype,
				body: body.to_vec()
			}.into(),
			info: 0..0,
		}
	}

	fn rec_s(
		name: &str,
		fields: &[TypedIdent],
	) -> Node {
		Node {
			kind: VT::Unit,
			expr: Expr::Rec {
				name: name.into(),
				fields: fields.to_vec(),
			}.into(),
			info: 0..0,
		}
	}

	fn if_s(
		cond: Node,
		bt: &[Node],
		bf: &[Node],
	) -> Node {
		Node {
			kind: bt.last().zip(bf.last()).map(|(t,f)| t.kind.meet(&f.kind)).unwrap_or(VT::Unit),
			expr: Expr::If {
				cond,
				bt: bt.to_vec(),
				bf: bf.to_vec(),
			}.into(),
			info: 0..0,
		}
	}

	fn while_s(
		cond: Node,
		body: &[Node],
	) -> Node {
		Node {
			kind: VT::Unit,
			expr: Expr::While {
				cond,
				body: body.to_vec(),
			}.into(),
			info: 0..0,
		}
	}

	fn assign_s(
		name: &str,
		body: Node,
	) -> Node {
		Node {
			kind: body.kind.clone(),
			expr: Expr::Assign {
				name: name.into(),
				body,
			}.into(),
			info: 0..0,
		}
	}

	#[test]
	fn unary_op_deref() -> miette::Result<()> {
		expr_test("@a", unary(UnaryOp::Deref, ident("a")))
	}

	#[test]
	fn unary_op_neg() -> miette::Result<()> {
		expr_test("-3", unary(UnaryOp::Neg, num(3)))
	}

	#[test]
	fn unary_op_not() -> miette::Result<()> {
		expr_test("!3", unary(UnaryOp::Not, num(3)))
	}

	#[test]
	fn unary_op_pos() -> miette::Result<()> {
		expr_test("+3", unary(UnaryOp::Pos, num(3)))
	}

	#[test]
	fn unary_op_ref() -> miette::Result<()> {
		expr_test("$a", unary(UnaryOp::Ref, ident("a")))
	}

	#[test]
	fn precedence() -> miette::Result<()> {
		expr_test("1 + 2 * 3", binary(BinaryOp::Add,
			num(1),
			binary(BinaryOp::Mul, num(2), num(3)),
		))?;
		expr_test("1 * 2 + 3", binary(BinaryOp::Add,
			binary(BinaryOp::Mul, num(1), num(2)),
			num(3),
		))
	}

	#[test]
	fn parentheses() -> miette::Result<()> {
		expr_test("1 * (2 + 3)", binary(BinaryOp::Mul,
			num(1),
			binary(BinaryOp::Add, num(2), num(3)),
		))
	}

	fn parse_test(
		input: &str,
		stmts: &[Node],
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
			var_s("a", VT::Unit, num(0)),
		])
	}

	#[test]
	fn var_stmt_expr() -> miette::Result<()> {
		parse_test("var a = 3 * 2 + 1", &[
			var_s("a", VT::Unit, binary(BinaryOp::Add,
				binary(BinaryOp::Mul, num(3), num(2)),
				num(1),
			)),
		])
	}

	#[test]
	fn var_stmt_vtype() -> miette::Result<()> {
		parse_test("var a: u8 = 0", &[
			var_s("a", VT::to_u8(), num(0)),
		])
	}

	#[test]
	fn var_stmt_udt_simple() -> miette::Result<()> {
		parse_test("var a = b", &[
			var_s("a", VT::Unit, ident("b")),
		])
	}

	#[test]
	fn var_stmt_udt_fncall_empty() -> miette::Result<()> {
		parse_test("var a = b()", &[
			var_s("a", VT::Unit, fn_call("b", &[]))
		])
	}

	#[test]
	fn var_stmt_udt_fncall_single() -> miette::Result<()> {
		parse_test("var a = b(c)", &[
			var_s("a", VT::Unit, fn_call("b", &[ident("c")]))
		])
	}

	#[test]
	fn var_stmt_udt_fncall_multi() -> miette::Result<()> {
		parse_test("var a = b(c, d + e)", &[
			var_s("a", VT::Unit, fn_call("b", &[
				ident("c"),
				binary(BinaryOp::Add, ident("d"), ident("e")),
			]))
		])
	}

	#[test]
	fn fn_stmt() -> miette::Result<()> {
		parse_test("fn a() {}", &[
			fn_s("a", &[], VT::Unit, &[]),
		])
	}

	#[test]
	fn fn_stmt_params() -> miette::Result<()> {
		parse_test("fn a(b:u8 c:s16 d:fw6 e:fd10) {}", &[
			fn_s("a", &[
				("b".into(), VT::to_u8()),
				("c".into(), VT::to_s16()),
				("d".into(), VT::to_f16(6)),
				("e".into(), VT::to_f32(10)),
			], VT::Unit, &[]),
		])
	}

	#[test]
	fn fn_stmt_rtype_simple() -> miette::Result<()> {
		parse_test("fn a() -> u8 {}", &[
			fn_s("a", &[], VT::to_u8(), &[])
		])
	}

	#[test]
	fn fn_stmt_rtype_udt() -> miette::Result<()> {
		parse_test("fn a() -> b {}", &[
			fn_s("a", &[], VT::UDT("b".to_string()), &[])
		])
	}

	#[test]
	fn fn_stmt_body() -> miette::Result<()> {
		parse_test("fn a() {
			var b = 1
			var c = 2
			b + c
		}", &[
			fn_s("a", &[], VT::Unit, &[
				var_s("b", VT::Unit, num(1)),
				var_s("c", VT::Unit, num(2)),
				binary(BinaryOp::Add, ident("b"), ident("c")),
			]),
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
		parse_test("rec vec{x:fd y:fd}", &[
			rec_s("vec", &[
				("x".into(), VT::to_f32(16)),
				("y".into(), VT::to_f32(16)),
			])
		])
	}

	#[test]
	fn field_access() -> miette::Result<()> {
		parse_test("var a = b.c.d", &[
			var_s("a", VT::Unit, binary(BinaryOp::Accessor,
				ident("b"),
				binary(BinaryOp::Accessor, ident("c"), ident("d")),
			))
		])
	}

	#[test]
	fn if_stmt() -> miette::Result<()> {
		parse_test("if a > b {a}", &[
			if_s(
				binary(BinaryOp::CmpGT, ident("a"), ident("b")),
				&[ident("a")],
				&[],
			)
		])
	}

	#[test]
	fn if_else_stmt() -> miette::Result<()> {
		parse_test("if a < b {a} else {b}", &[
			if_s(
				binary(BinaryOp::CmpLT, ident("a"), ident("b")),
				&[ident("a")],
				&[ident("b")],
			)
		])
	}

	#[test]
	fn while_stmt() -> miette::Result<()> {
		parse_test("while a == b {b}", &[
			while_s(
				binary(BinaryOp::CmpEq, ident("a"), ident("b")),
				&[ident("b")],
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

