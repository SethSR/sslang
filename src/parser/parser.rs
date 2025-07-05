
use std::collections::{HashMap, HashSet};
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
	TokenInfo,
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

pub(crate) type Scope = HashMap<Rc<str>, NodeId>;

#[derive(Debug, Default, Clone)]
pub(crate) struct ScopeTracker(Vec<Scope>);

#[derive(Debug)]
pub(super) struct Parser<'a,'b> {
	input: &'a [Token],
	source: &'b str,
	index: usize,

	pub(super) scopes: ScopeTracker,

	// TODO - srenshaw - All of these fields will probably be moved into a Scope structure.
	pub(super) nodes: NodeStore,
	pub(super) records: HashSet<Rc<str>>,
	pub(super) functions: HashSet<Rc<str>>,

	// DEBUG
	dbg_depth: usize,
}

impl<'a,'b> Parser<'a,'b> {
	pub fn new(source: &'b str, input: &'a [Token]) -> Self {
		Self {
			source,
			input,
			index: 0,

			scopes: ScopeTracker::default(),

			nodes: NodeStore::default(),
			records: HashSet::default(),
			functions: HashSet::default(),

			dbg_depth: 2,
		}
	}

	pub fn peek(&self, offset: isize) -> &Token {
		&self.input[self.index.saturating_add_signed(offset)]
	}

	/// program := expr*
	pub fn program(&mut self) -> miette::Result<NodeId> {
		let mut program = Vec::default();
		self.scopes.push();
		while self.peek(0).tt != TokenType::Eof {
			let nx = self.expr(0)?;
			let node = self.nodes.get(nx)?;
			match &node.expr {
				Expr::Var { name, ..} |
				Expr::Fun { name, ..} => {
					self.scopes.insert(name, nx);
				}
				_ => {}
			}
			program.push(nx);
		}
		let scope = self.scopes.pop()
			.expect("empty scope-list in `parser::program`");
		let nx = self.nodes.new_block(program, scope, 0..self.source.len());
		Ok(nx)
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
		TT::Eof => None,
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
			TokenType::Eof => error!(eof, self, "Number"),
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
			TokenType::Eof => error!(eof, self, "Identifier"),
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
			TokenType::Ident(ref s) => Ok(ValueType::Udt(Rc::clone(s))),
			_ => error!(token.tt, self, "Value Type"),
		};
		self.index += 1;
		log_item(&token, self.dbg_depth);
		self.dbg_depth -= 2;
		result
	}

	pub(super) fn match_token(&mut self, tt: TokenType) -> miette::Result<()> {
		match self.peek(0).tt.clone() {
			t if t != tt => if t == TokenType::Eof {
				error!(eof, self, tt)
			} else {
				error!(self, tt)
			}
			_ => {
				self.index += 1;
				Ok(())
			}
		}
	}

	pub(super) fn ident_typed(&mut self) -> miette::Result<(Rc<str>, ValueType, TokenInfo)> {
		log("TypedIdent", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		let result = self.ident()
			.and_then(|id| self.match_token(TokenType::Colon).map(|_| id))
			.and_then(|id| self.value_type().map(|vt| {
				let end = self.peek(-1).range().end;
				(Rc::clone(&id), vt, start..end)
			}));
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
		while let Ok(next) = self.match_token(TokenType::Comma)
			.and_then(|_| self.expr(0))
		{
			out.push(next);
		}
		let _ = self.match_token(TokenType::Comma);
		self.dbg_depth -= 2;
		Some(out)
	}

	/// if := 'if' expr block ('else' block)?
	pub(super) fn stmt_if(&mut self) -> miette::Result<NodeId> {
		log("If", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		self.match_token(TokenType::If)?;
		let cond = self.expr(0)?;

		let f_scopes = self.scopes.clone();

		let bt = {
			self.scopes.push();
			let (body, info) = self.block()?;
			let scope = self.scopes.pop()
				.expect("empty scope-list in `parser::stmt_if::true_block`");
			self.nodes.new_block(body, scope, info)
		};
		let t_scopes = self.scopes.clone();

		self.scopes = f_scopes;
		let bf = if self.match_token(TokenType::Else).is_ok() {
			self.scopes.push();
			let (body, info) = self.block()?;
			let scope = self.scopes.pop()
				.expect("empty scope-list in `parser::stmt_if::false_block`");
			Some(self.nodes.new_block(body, scope, info))
		} else {
			None
		};

		self.scopes = ScopeTracker::merge(
			&mut self.nodes,
			t_scopes,
			self.scopes.clone())?;

		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;
		Ok(self.nodes.new_if(cond, bt, bf, start..end))
	}

	fn expr_call(&mut self, lhs: NodeId, op_token: Token) -> miette::Result<NodeId> {
		let lhs_node = self.nodes.get(lhs)?.clone();
		let name = match lhs_node.expr {
			Expr::Id(name) => name,
			_ => return error!(self, "Identifier"),
		};

		self.index += 1;
		let args = self.args().unwrap_or_default();

		if TokenType::CParen != self.peek(0).tt {
			return error!(self, ")");
		}
		self.index += 1;

		if !self.functions.contains(&name) {
			if let Some(rx) = self.scopes.find(&name) {
				let def = self.nodes.get(rx)?;
				if let Expr::Fun { params,..} = &def.expr {
					assert_eq!(args.len(), params.len(),
						"mismatched argument and parameter lists");
				}
			}
		} else {
			return Err(miette::miette! {
				code = self.source,
				labels = [
					LabeledSpan::at(op_token.range(), "here"),
				],
				"Call to unknown function: '{name}'",
			});
		}
		Ok(self.nodes.new_call(name, args, lhs_node.info.start..op_token.range().end))
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
				// initialization, "ident -> block" sequences, and assignment.
				if self.peek(1).tt == TT::OBrace && self.peek(3).tt == TT::Colon {
					self.expr_rec_init()?
				} else {
					self.index += 1;
					if let Some(nx) = self.scopes.find(s) {
						nx
					} else {
						let nx = self.nodes.new_id(Rc::clone(s), ValueType::Any, left_token.range());
						self.scopes.insert(s, nx);
						nx
					}
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

			TT::Eof => return error!(eof, self,
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
				TT::Eof) {
				break;
			}

			if TT::OParen == op_token.tt {
				lhs = self.expr_call(lhs, op_token)?;
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
		let body = if self.peek(1).tt == TokenType::OBrace {
			self.scopes.push();
			let (body, info) = self.block()?;
			let scope = self.scopes.pop()
				.expect("empty scope-list in `parser::field_init`");
			self.nodes.new_block(body, scope, info)
		} else {
			self.expr(0)?
		};
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
	pub(super) fn params(&mut self) -> Vec<(Rc<str>, ValueType, TokenInfo)> {
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
	pub(super) fn block(&mut self) -> miette::Result<(Vec<NodeId>, TokenInfo)> {
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
				(body, start..end)
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
		let fields = self.params()
			.into_iter()
			.map(|(fname, ftype, finfo)| self.nodes.new_id(fname, ftype, finfo))
			.collect();
		self.match_token(TokenType::CBrace)?;
		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;

		if self.records.contains(&name) {
			return Err(miette::miette! {
				labels = vec![
					LabeledSpan::at(start..end, "here"),
				],
				"A Record with this name is already defined."
			});
		}
		self.records.insert(Rc::clone(&name));

		let nx = self.nodes.new_rec(Rc::clone(&name), fields, start..end);
		self.scopes.insert(&name, nx);
		Ok(nx)
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

		let (params, body) = {
			self.scopes.push();
			let params = params.iter()
				.map(|(pname, ptype, pinfo)| {
					let px = self.nodes.new_var(Rc::clone(pname), ptype.clone(), None, pinfo.clone());
					self.scopes.insert(pname, px)
				})
				.collect();
			let (body,info) = self.block()?;
			let scope = self.scopes.pop()
				.expect("empty scope-list in `parser::stmt_fn`");
			(params, self.nodes.new_block(body, scope, info))
		};

		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;

		if self.functions.contains(&name) {
			return Err(miette::miette! {
				labels = vec![
					LabeledSpan::at(start..end, "here"),
				],
				"A Function with this name is already defined."
			});
		}
		self.functions.insert(Rc::clone(&name));

		Ok(self.nodes.new_fun(Rc::clone(&name), params, rtype, body, start..end))
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
		let body = if self.peek(0).tt == TokenType::OBrace {
			self.scopes.push();
			let (body,info) = self.block()?;
			let scope = self.scopes.pop()
				.expect("empty scope-list in `parser::stmt_var`");
			self.nodes.new_block(body, scope, info)
		} else {
			self.expr(0)?
		};
		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;

		self.scopes.insert(&name, body);

		if self.nodes.get(body).map(|n| n.expr.is_const(&self.nodes))
			.unwrap_or_default()
		{
			Ok(body)
		} else {
			Ok(self.nodes.new_var(name, vtype, Some(body), start..end))
		}
	}

	/// while := 'while' expr block
	pub(super) fn stmt_while(&mut self) -> miette::Result<NodeId> {
		log("While", self.dbg_depth);
		self.dbg_depth += 2;
		let start = self.peek(0).range().start;
		self.match_token(TokenType::While)?;
		let cond = self.expr(0)?;
		let body = {
			self.scopes.push();
			let (body, info) = self.block()?;
			let scope = self.scopes.pop()
				.expect("empty scope-list in `parser::stmt_while`");
			self.nodes.new_block(body, scope, info)
		};
		let end = self.peek(-1).range().end;
		self.dbg_depth -= 2;
		Ok(self.nodes.new_while(cond, body, start..end))
	}
}

impl ScopeTracker {
	/// Used in type-checking (and potentially elsewhere) to re-add block scopes.
	pub fn add(&mut self, scope: Scope) {
		self.0.push(scope)
	}

	fn insert(&mut self, name: &Rc<str>, nx: NodeId) -> NodeId {
		let index = self.0.len();
		if let Some(scope) = self.0.last_mut() {
			println!("add {name} : {nx} to scope {}", index-1);
			scope.insert(Rc::clone(name), nx);
			nx
		} else {
			panic!("add {name} : {nx} with no base scope");
		}
	}

	fn push(&mut self) {
		println!("pushing a scope | prev-top {:?}", self.0.last());
		self.0.push(Scope::default());
	}

	fn pop(&mut self) -> Option<Scope> {
		let last = self.0.pop();
		println!("popping a scope | prev-top {last:?}");
		last
	}

	pub fn find(&self, id: &Rc<str>) -> Option<NodeId> {
		for scope in self.0.iter().rev() {
			if let Some(&nx) = scope.get(id) {
				return Some(nx);
			}
		}
		None
	}

	fn merge(
		nodes: &mut NodeStore,
		mut t_scopes: ScopeTracker,
		mut f_scopes: ScopeTracker,
	) -> miette::Result<ScopeTracker> {
		println!("merging scopes");

		let mut scopes = vec![];

		let mut t_iter = t_scopes.0.iter_mut();
		let mut f_iter = f_scopes.0.iter_mut();
		loop {
			let mut scope = Scope::default();

			match (t_iter.next(), f_iter.next()) {
				(Some(t_scope), Some(f_scope)) => {
					for (name, tnx) in t_scope.iter() {
						let nx = f_scope.remove(name)
							.filter(|fnx| tnx != fnx)
							.map(|fnx| nodes.new_phi(*tnx, fnx))
							.unwrap_or(Ok(*tnx))?;
						scope.insert(Rc::clone(name), nx);
					}

					for (name, fnx) in f_scope {
						let nx = t_scope.remove(name)
							.filter(|tnx| tnx != fnx)
							.map(|tnx| nodes.new_phi(tnx, *fnx))
							.unwrap_or(Ok(*fnx))?;
						scope.insert(Rc::clone(name), nx);
					}
				}

				// Just copy over missing nodes
				(Some(o_scope), None) |
				(None, Some(o_scope)) => {
					for (name, nx) in o_scope {
						scope.insert(Rc::clone(name), *nx);
					}
				}

				(None, None) => break,
			}

			scopes.push(scope);
		}

		Ok(ScopeTracker(scopes))
	}
}

#[test]
fn test_merge() {
	let mut n = NodeStore::default();
	let n0 = n.new_num(0, ValueType::Any, 0..0);
	let n1 = n.new_num(0, ValueType::Any, 0..0);
	let n2 = n.new_num(0, ValueType::Any, 0..0);
	let n3 = n.new_num(0, ValueType::Any, 0..0);
	let n4 = n3 + 1;

	let mut s1 = ScopeTracker::default();
	s1.push();
	s1.insert(&"a".into(), n0);
	s1.push();
	s1.insert(&"b".into(), n1);
	s1.insert(&"c".into(), n2);
	s1.push();

	let mut s2 = ScopeTracker::default();
	s2.push();
	s2.push();
	s2.insert(&"a".into(), n1);
	s2.insert(&"b".into(), n3);
	s2.insert(&"c".into(), n2);

	let s3 = ScopeTracker::merge(&mut n, s1, s2)
		.expect("unable to merge s1 and s2");
	assert_eq!(s3.0.len(), 3);

	let phi = n.get(n4)
		.unwrap_or_else(|_| panic!("no phi node for id 'b'\n\nNodes: {n:?}"));
	assert_eq!(phi.expr, Expr::Phi { lhs: n1, rhs: n3 });

	let mut temp1 = Scope::default();
	temp1.insert("a".into(), n0);
	assert_eq!(s3.0[0], temp1);
	let mut temp2 = Scope::default();
	temp2.insert("a".into(), n1);
	temp2.insert("b".into(), n4);
	temp2.insert("c".into(), n2);
	assert_eq!(s3.0[1], temp2);
}

