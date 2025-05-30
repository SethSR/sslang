
use std::rc::Rc;

use miette::{LabeledSpan, IntoDiagnostic, WrapErr};
use tracing::debug;

use crate::tokens::{Token, TokenType};

use super::node::{NodeId, NodeStore};
use super::{
	BinaryOp,
	Expr,
	Fix,
	Int,
	TypedIdent,
	ValueType,
};

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

#[derive(Debug)]
pub(super) struct Parser<'a,'b> {
	input: &'a [Token],
	source: &'b str,
	index: usize,

	pub(super) nodes: NodeStore,

	// DEBUG
	dbg_depth: usize,
}

impl<'a,'b> Parser<'a,'b> {
	pub fn new(source: &'b str, input: &'a [Token]) -> Self {
		Self {
			source,
			input: &input,
			index: 0,

			nodes: NodeStore::default(),

			dbg_depth: 2,
		}
	}

	pub fn peek(&self, offset: isize) -> &Token {
		&self.input[self.index.saturating_add_signed(offset)]
	}

	/// program := expr*
	pub fn program(
		&mut self,
	) -> miette::Result<NodeId> {
		let mut program = Vec::default();
		while self.peek(0).tt != TokenType::EOF {
			program.push(self.expr(0)?);
		}
		Ok(self.nodes.new_block(program, 0..self.source.len()))
	}
}

fn float_to_fixed(n: f64) -> i64 {
	(n * (1u64 << 32) as f64) as i64
}

#[test]
fn convert_float_to_fixed() {
	assert_eq!(0, float_to_fixed(0.0));
	assert_eq!(0x00000000_80000000, float_to_fixed(0.5));
	assert_eq!(0x00000005_4CCCCCCC, float_to_fixed(5.3));
	assert_eq!(0x00000006_66666666, float_to_fixed(6.4));
}

fn infix_binding_power(tt: &TokenType) -> Option<(u8,u8)> {
	use TokenType as TT;

	match tt {
		// Assignment Operator
		// HACK - srenshaw - The values were chosen semi-randomly. Should probably come up with better
		// ones.
		TT::Eq1 => Some((100,0)),

		// Boolean Operators
		TT::Amp2 | TT::Bar2 | TT::Carrot2 => Some((1,2)),

		// Bitwise Operators
		TT::Amp1 | TT::Bar1 | TT::Carrot1 => Some((3,4)),

		// Boolean Comparison Operators
		TT::Eq2 | TT::BangEq |
		TT::RArrow1 | TT::RArrEq |
		TT::LArrow1 | TT::LArrEq => Some((5,6)),

		// Add-Sub Operators
		TT::Plus | TT::Minus => Some((7,8)),

		// Mul-Div Operators
		TT::Star | TT::Slash | TT::Percent | TT::SlashPer |
		// Shift Operators
		TT::LArrow2 | TT::RArrow2 => Some((9,10)),
		// Rotation Operators
		TT::LArrBar | TT::RArrBar => Some((11,12)),

		// Access Operator
		TT::Dot => Some((20,19)),

		// Function-call Operator
		TT::OParen => Some((22,21)),

		// Keywords
		TT::If | TT::Else |
		TT::Fun | TT::Rec | TT::Var |
		TT::While | TT::True | TT::False |
		// Type Tokens
		TT::Bool |
		TT::U8 | TT::U16 | TT::U32 |
		TT::S8 | TT::S16 | TT::S32 |
		TT::F16(_) | TT::F32(_) |
		TT::Ident(_) | TT::Integer(_) | TT::Fixed(_) |
		// Unary Op Tokens
		TT::At | TT::Bang |
		TT::Dollar |
		// Syntax & Punctuation
		TT::Colon | TT::Comma | TT::CBrace | TT::CParen | TT::OBrace | TT::RetArrow |
		// End-of-file
		TT::EOF => None,
	}
}

fn prefix_binding_power(tt: &TokenType) -> Option<u8> {
	use TokenType as TT;

	match tt {
		TT::Plus | TT::Minus => Some(13),
		TT::Dollar | TT::At => Some(15),
		TT::Bang => Some(17),
		_ => None,
	}
}

fn log(s: &str, depth: usize) {
	debug!("{:>1$}{s}", "| ", depth);
}

fn log_item<T: std::fmt::Display>(item: T, depth: usize) {
	debug!("{:>1$}{item}", "> ", depth);
}

impl Parser<'_,'_> {
	pub(super) fn num(&mut self) -> miette::Result<i64> {
		log("Num", self.dbg_depth);
		self.dbg_depth += 2;
		let out = match self.peek(0).tt.clone() {
			TokenType::Integer(s) => {
				self.index += 1;
				Ok(s.parse::<i64>()
					.into_diagnostic()?)
			}
			TokenType::Fixed(s) => {
				self.index += 1;
				Ok(float_to_fixed(s
					.chars()
					.filter(|c| *c != '_')
					.collect::<String>()
					.parse::<f64>()
					.into_diagnostic()
					.wrap_err("lexer should not allow invalid fixed-point values")?))
			}
			TokenType::EOF => error!(eof, self, "Number"),
			_ => error!(self, "Number"),
		};
		self.dbg_depth -= 2;
		out
	}

	pub(super) fn ident(&mut self) -> miette::Result<Rc<str>> {
		log("Ident", self.dbg_depth);
		self.dbg_depth += 2;
		let out = match self.peek(0).tt.clone() {
			TokenType::Ident(s) => {
				log_item(&s, self.dbg_depth);
				self.index += 1;
				Ok(s)
			}
			TokenType::EOF => error!(eof, self, "Identifier"),
			_ => error!(self, "Identifier"),
		};
		self.dbg_depth -= 2;
		out
	}

	pub(super) fn parse_fixed_point(&self, prefix: &str, max_bits: u8) -> miette::Result<u8> {
		let token = self.peek(0);
		let token_str = token.to_string();
		let Some(bit_spec) = token_str.strip_prefix(prefix) else {
			return Err(miette::miette! {
				labels = vec![
					LabeledSpan::at(token.range(), "here"),
				],
				"Parsed a Fixed-point type that doesn't start with '{prefix}'"
			}.with_source_code(self.source.to_owned()));
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
			}.with_source_code(self.source.to_owned()));
		};

		if bits > max_bits {
			return error!(token, self, format!("Bit specifier between 0..={max_bits}"));
		}
		Ok(bits)
	}

	pub(super) fn value_type(&mut self) -> miette::Result<ValueType> {
		log("ValueType", self.dbg_depth);
		self.dbg_depth += 2;
		let token = self.peek(0).clone();
		let result = match token.tt {
			TokenType::U8  => Ok(ValueType::to_u8()),
			TokenType::U16 => Ok(ValueType::to_u16()),
			TokenType::U32 => Ok(ValueType::to_u32()),
			TokenType::S8  => Ok(ValueType::to_s8()),
			TokenType::S16 => Ok(ValueType::to_s16()),
			TokenType::S32 => Ok(ValueType::to_s32()),
			TokenType::F16(_) => self.parse_fixed_point("fw", 16).map(ValueType::to_f16),
			TokenType::F32(_) => self.parse_fixed_point("fd", 32).map(ValueType::to_f32),
			TokenType::Ident(ref s) => Ok(ValueType::UDT(Rc::clone(s))),
			_ => error!(token.tt, self, "Value Type"),
		};
		self.index += 1;
		log_item(&token, self.dbg_depth);
		self.dbg_depth -= 2;
		result
	}

	pub(super) fn match_token(&mut self, tt: TokenType) -> miette::Result<()> {
		match self.peek(0).tt.clone() {
			t if t != tt => if t == TokenType::EOF {
				error!(eof, self, tt)
			} else {
				error!(self, tt)
			}
			_ => Ok(self.index += 1),
		}
	}

	pub(super) fn ident_typed(&mut self) -> miette::Result<TypedIdent> {
		log("TypedIdent", self.dbg_depth);
		self.dbg_depth += 2;
		let result = self.ident()
			.and_then(|id| self.match_token(TokenType::Colon).map(|_| id))
			.and_then(|id| self.value_type().map(|vt| (id, vt)));
		self.dbg_depth -= 2;
		result
	}

	/// args := (expr (',' expr)* ','?)?
	pub(super) fn args(&mut self) -> Option<Vec<NodeId>> {
		log("Args", self.dbg_depth);
		self.dbg_depth += 2;
		let first = self.expr(0)
			.ok()?;
		let mut out = vec![first];
		while let Some(next) = self.match_token(TokenType::Comma)
			.and_then(|_| self.expr(0))
			.ok()
		{
			out.push(next);
		}
		let _ = self.match_token(TokenType::Comma);
		self.dbg_depth -= 2;
		Some(out)
	}

	/// if := 'if' expr block ('else' block)?
	pub(super) fn stmt_if(&mut self) -> miette::Result<NodeId> {
		debug!("{:1$}If", "", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		self.match_token(TokenType::If)?;
		let cond = self.expr(0)?;
		let bt = self.block()?;
		let bf = self.match_token(TokenType::Else)
			.and_then(|_| self.block())
			.unwrap_or_default();
		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;
		Ok(self.nodes.new_if(cond, bt, bf, start..end))
	}

	pub(super) fn expr(&mut self, min_bp: u8) -> miette::Result<NodeId> {
		log("Expr", self.dbg_depth);
		self.dbg_depth += 2;
		use TokenType as TT;

		let left_token = self.peek(0).clone();
		let mut lhs: NodeId = match left_token.tt {
			TT::Rec => self.stmt_rec()?,
			TT::Fun => self.stmt_fn()?,
			TT::If => self.stmt_if()?,
			TT::Var => self.stmt_var()?,
			TT::While => self.stmt_while()?,

			TT::True => {
				self.index += 1;
				self.nodes.new_bool(true, left_token.range())
			}
			TT::False => {
				self.index += 1;
				self.nodes.new_bool(false, left_token.range())
			}

			TT::Ident(ref s) => {
				// HACK - srenshaw - We probably need a more robust way to distinguish between Record
				// initialization, "ident -> block" sequences, and assignement.
				if self.peek(1).tt == TT::OBrace && self.peek(3).tt == TT::Colon {
					self.expr_rec_init()?
				} else {
					self.index += 1;
					self.nodes.new_id(Rc::clone(s), ValueType::Any, left_token.range())
				}
			}

			TT::Integer(_) => {
				let num = self.num()?;
				self.nodes.new_num(num, ValueType::Int(Int::Bot), left_token.range())
			}

			TT::Fixed(_) => {
				let num = self.num()?;
				self.nodes.new_num(num, ValueType::Fix(Fix::Bot), left_token.range())
			}

			TT::OParen => {
				self.index += 1;
				let lhs = self.expr(0)?;
				if TT::CParen != self.peek(0).tt {
					return error!(self, ")");
				}
				self.index += 1;
				lhs
			}

			TT::Plus |
			TT::Minus |
			TT::Dollar |
			TT::At |
			TT::Bang => {
				let Some(r_bp) = prefix_binding_power(&self.peek(0).tt) else {
					return error!(self, "Expected 'Unary Operator'");
				};
				self.index += 1;
				let rhs = self.expr(r_bp)?;
				self.nodes.new_unary((&left_token.tt).try_into()?, rhs, left_token.range())
					.map_err(|err| err.with_source_code(self.source.to_string()))?
			}

			TT::EOF => return error!(eof, self,
				"Identifier, Function Call, or Literal"),

			_ => return error!(self,
				"Identifier, Function Call, or Literal"),
		};

		loop {
			let op_token = self.peek(0).clone();
			if matches!(op_token.tt,
				TT::Ident(_) | TT::Integer(_) | TT::Fixed(_) |
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
				let lhs_node = self.nodes.get(lhs)?.clone();
				let Expr::Id(name) = lhs_node.expr else {
					return error!(self, "Identifier");
				};

				self.index += 1;
				if TT::CParen == self.peek(0).tt {
					self.index += 1;
					lhs = self.nodes.new_call(name, vec![], lhs_node.info.start..op_token.range().end);
					continue;
				}

				let Some(args) = self.args() else {
					return error!(self, "Argument List");
				};
				if TT::CParen != self.peek(0).tt {
					return error!(self, ")");
				}
				self.index += 1;
				lhs = self.nodes.new_call(name, args, lhs_node.info.start..op_token.range().end);
				continue;
			}

			if let Some((l_bp,r_bp)) = infix_binding_power(&op_token.tt) {
				if l_bp < min_bp {
					break;
				}

				self.index += 1;
				let op: BinaryOp = (&op_token.tt).try_into()?;
				let rhs = self.expr(r_bp)?;
				lhs = self.nodes.new_binary(op, lhs, rhs, op_token.range())?;
				continue;
			}

			break;
		}

		log_item(self.nodes.get(lhs)?, self.dbg_depth);
		self.dbg_depth -= 2;
		Ok(lhs)
	}

	/// field_init := ident ':' (block | expr)
	fn field_init(&mut self) -> miette::Result<(Rc<str>, NodeId)> {
		log("FieldInit", self.dbg_depth);
		self.dbg_depth += 2;
		let id = self.ident()?;
		self.match_token(TokenType::Colon)?;
		let body = self.block()
			.or_else(|_| self.expr(0))?;
		self.dbg_depth -= 2;
		Ok((id, body))
	}

	/// expr_rec_init := ident '{' field_init* '}'
	pub(super) fn expr_rec_init(&mut self) -> miette::Result<NodeId> {
		log("RecInit", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		let id = self.ident()?;
		self.match_token(TokenType::OBrace)?;
		let mut fields = vec![];
		if let Ok(first) = self.field_init() {
			fields.push(first);

			while let Ok(field) = self.match_token(TokenType::Comma)
				.and_then(|_| self.field_init())
			{
				fields.push(field);
			}
			let _ = self.match_token(TokenType::Comma);
		}
		self.match_token(TokenType::CBrace)?;
		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;
		Ok(self.nodes.new_rec_init(id, fields, start..end))
	}

	/// params := ( typed_ident (',' typed_ident)* ','? )?
	pub(super) fn params(&mut self) -> Vec<TypedIdent> {
		log("Params", self.dbg_depth);
		self.dbg_depth += 2;

		let Ok(first) = self.ident_typed() else {
			self.dbg_depth -= 2;
			return vec![];
		};

		let mut out = vec![first];
		while let Ok(id) = self.match_token(TokenType::Comma)
			.and_then(|_| self.ident_typed())
		{
			out.push(id);
		}

		let _ = self.match_token(TokenType::Comma);
		self.dbg_depth -= 2;
		out
	}

	/// block := '{' expr* '}'
	pub(super) fn block(&mut self) -> miette::Result<NodeId> {
		log("Block", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		match self.match_token(TokenType::OBrace) {
			Ok(_) => {},
			Err(e) => {
				self.dbg_depth -= 2;
				return Err(e);
			}
		}
		let mut body = Vec::new();
		while let Ok(expr) = self.expr(0) {
			body.push(expr);
		}
		let result = self.match_token(TokenType::CBrace)
			.map(|_| {
				let end = self.peek(-1).range().end;
				self.nodes.new_block(body, start..end)
			});
		self.dbg_depth -= 2;
		result
	}

	/// rec := 'rec' ident '{' params '}'
	pub(super) fn stmt_rec(&mut self) -> miette::Result<NodeId> {
		log("Record", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		self.match_token(TokenType::Rec)?;
		let name = self.ident()?;
		self.match_token(TokenType::OBrace)?;
		let fields = self.params();
		self.match_token(TokenType::CBrace)?;
		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;
		Ok(self.nodes.new_rec(name, fields, start..end))
	}

	/// fn := 'fn' ident '(' params ')' ('->' value_type)? block
	pub(super) fn stmt_fn(&mut self) -> miette::Result<NodeId> {
		log("Function", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		self.match_token(TokenType::Fun)?;
		let name = self.ident()?;
		self.match_token(TokenType::OParen)?;
		let params = self.params();
		self.match_token(TokenType::CParen)?;
		let rtype = self.match_token(TokenType::RetArrow)
			.and_then(|_| self.value_type())
			.unwrap_or(ValueType::Unit);
		let body = self.block()?;
		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;
		Ok(self.nodes.new_fun(name, params, rtype, body, start..end))
	}

	/// var := 'var' ident (':' value_type)? '=' (block | expr)
	pub(super) fn stmt_var(&mut self) -> miette::Result<NodeId> {
		log("Variable", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		self.match_token(TokenType::Var)?;
		let name = self.ident()?;
		let vtype = self.match_token(TokenType::Colon)
			.and_then(|_| self.value_type())
			.unwrap_or(ValueType::Unit);
		self.match_token(TokenType::Eq1)?;
		let body = self.block()
			.or_else(|_| self.expr(0))?;
		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;
		Ok(self.nodes.new_var(name, vtype, body, start..end))
	}

	/// while := 'while' expr block
	pub(super) fn stmt_while(&mut self) -> miette::Result<NodeId> {
		log("While", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		self.match_token(TokenType::While)?;
		let cond = self.expr(0)?;
		let body = self.block()?;
		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;
		Ok(self.nodes.new_while(cond, body, start..end))
	}
}

