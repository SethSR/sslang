
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use miette::{LabeledSpan, IntoDiagnostic, WrapErr};
use tracing::debug;

use crate::tokens::{Token, TokenType};

use super::{BinaryOp, Error, Expr, Fix, Int, TokenInfo, ValueType};
use super::error::{Context, Result, error, report};
use super::node::{NodeId, NodeStore};

fn eof_error(parser: &Parser, msg: &str) -> Error {
	report(&parser.source, parser.peek(-1).range(), "after here",
		&format!("Expected {msg}, Found EoF"))
}

fn expected(parser: &Parser, msg: &str) -> Error {
	Error::Parse(miette::miette! {
		labels = [
			LabeledSpan::at(parser.peek(-1).range(), "between here"),
			LabeledSpan::at(parser.peek(0).range(), "and here"),
		],
		"Expected {msg}, found {}", parser.peek(0)
	}.with_source_code(parser.source.to_string()))
}

pub(crate) type Scope = HashMap<Rc<str>, NodeId>;

#[derive(Debug, Default, Clone)]
pub(crate) struct ScopeTracker(Vec<Scope>);

#[derive(Debug)]
pub(crate) enum StackOp {
	Block(TokenType),
	Expr,
	Rec,
	RecEnd,
	Fun,
	FunRet,
	FunEnd,
	Param(TokenType),
}

#[derive(Debug)]
enum StackValue {
	NodeId(NodeId),
	Block(NodeId),
	Ident(Rc<str>, u16),
	Param(NodeId),
	Type(ValueType),

	// Placeholders
	Rec,
	Fun,
}

#[derive(Debug)]
pub(crate) struct Parser {
	input: Box<[Token]>,
	source: Rc<str>,
	index: usize,

	pub(crate) scopes: ScopeTracker,

	// TODO - srenshaw - All of these fields will probably be moved into a Scope structure.
	pub(crate) nodes: NodeStore,
	pub(crate) records: HashSet<Rc<str>>,
	pub(crate) functions: HashSet<Rc<str>>,

	// Stack
	pub(crate) stack: Vec<StackOp>,
	values: Vec<StackValue>,
	pub(crate) ast: Vec<NodeId>,

	// DEBUG
	dbg_depth: usize,
	dbg_ctx: Context,
}

impl Parser {
	pub fn new(source: &str, input: &[Token]) -> Self {
		let mut scopes = ScopeTracker::default();

		// create the expression block for the main procedure
		scopes.push();

		Self {
			source: source.into(),
			input: input.into(),
			index: 0,

			scopes,

			nodes: NodeStore::default(),
			records: HashSet::default(),
			functions: HashSet::default(),

			stack: vec![StackOp::Block(TokenType::Eof)],
			values: vec![],
			ast: vec![],

			dbg_depth: 2,
			dbg_ctx: Context::default(),
		}
	}

	const EOF: Token = Token { tt: TokenType::Eof, start: 0 };
	pub fn peek(&self, offset: isize) -> &Token {
		self.input.get(self.index.saturating_add_signed(offset))
			.unwrap_or(&Self::EOF)
	}
}

pub(crate) enum StepResult {
	Ok(String),
	Err(Error),
	Fatal(Error),
	Done,
}

impl From<&str> for StepResult {
	fn from(s: &str) -> Self {
		Self::Ok(s.into())
	}
}

impl From<String> for StepResult {
	fn from(s: String) -> Self {
		Self::Ok(s)
	}
}

impl From<Error> for StepResult {
	fn from(e: Error) -> Self {
		Self::Err(e)
	}
}

impl Parser {
	pub(crate) fn step(&mut self) -> StepResult {
		match self.stack.pop() {
			Some(StackOp::Expr) => {
				use StackValue as SV;

				let mut values = vec![];
				if let StepResult::Err(e) = self.expr2(&mut values) {
					return e.into();
				}

				match values.pop() {
					Some(SV::NodeId(nx)) => {
						let node = match self.nodes.get(nx) {
							Ok(node) => node,
							Err(e) => return Error::Parse(e).into(),
						};
						match &node.expr {
							Expr::Var { name, ..} |
							Expr::Fun { name, ..} => {
								self.scopes.insert(name, nx);
							}
							_ => {}
						}
						self.ast.push(nx);
						format!("Pushed top-level expression: '{node}'").into()
					}

					Some(SV::Rec) => {
						"RECORD - placeholder until we can remove hybrid expression parser".into()
					}
					Some(SV::Fun) => {
						"FUNCTION - placeholder until we can remove hybrid expression parser".into()
					}

					Some(sv @ SV::Block(..)) |
					Some(sv @ SV::Type(..)) |
					Some(sv @ SV::Ident(..)) |
					Some(sv @ SV::Param(..)) => {
						StepResult::Fatal(error(&self.source, self.peek(0).range(),
							&format!("Unexpected stack-value: '{sv:?}'")))
					}

					None => StepResult::Fatal(error(&self.source, self.peek(0).range(),
						"Empty value stack")),
				}
			}

			Some(StackOp::Rec) => {
				match self.rec_start() {
					Ok(result) => result,
					Err(e) => e.into(),
				}
			}

			Some(StackOp::RecEnd) => match self.rec_end() {
				Ok(result) => result,
				Err(e) => e.into(),
			}

			Some(StackOp::Param(closing_token)) => match self.param(closing_token) {
				Ok(result) => result,
				Err(e) => e.into(),
			}

			Some(StackOp::Block(closing_token)) => match self.block2(closing_token) {
				Ok(result) => result,
				Err(e) => e.into(),
			}

			Some(StackOp::Fun) => match self.fun_start() {
				Ok(result) => result,
				Err(e) => e.into(),
			}

			Some(StackOp::FunRet) => match self.fun_return() {
				Ok(result) => result,
				Err(e) => e.into(),
			}

			Some(StackOp::FunEnd) => match self.fun_end() {
				Ok(result) => result,
				Err(e) => e.into(),
			}

			None => StepResult::Done,
		}
	}

	pub fn finish(mut self) -> Result<super::Output> {
		let start = match self.values.pop() {
			Some(StackValue::Block(start)) => start,
			value => return Err(self.dbg_ctx.with_msg(&format!("found value '{value:?}' instead of final block"))),
		};

		Ok(super::Output {
			start,
			store: self.nodes,
			records: self.records,
			functions: self.functions,
			scopes: self.scopes,
		})
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

impl Parser {
	pub(super) fn num(&mut self) -> Result<i64> {
		with_ctx!(self, "Num", {
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
				TokenType::Eof => Err(eof_error(self, "Number")),
				_ => Err(expected(self, "Number")),
			};
			self.dbg_depth -= 2;
			out
		})
	}

	pub(super) fn ident(&mut self) -> Result<Rc<str>> {
		with_ctx!(self, "Ident", {
			log("Ident", self.dbg_depth);
			self.dbg_depth += 2;
			let out = match self.peek(0).tt.clone() {
				TokenType::Ident(s) => {
					log_item(&s, self.dbg_depth);
					self.index += 1;
					Ok(s)
				}
				TokenType::Eof => Err(eof_error(self, "Identifier")),
				_ => Err(expected(self, "Identifier")),
			};
			self.dbg_depth -= 2;
			out
		})
	}

	pub(super) fn parse_fixed_point(&mut self, prefix: &str, max_bits: u8) -> Result<u8> {
		with_ctx!(self, "parse_fixed_point", {
			let token = self.peek(0);
			let token_str = token.to_string();
			let Some(bit_spec) = token_str.strip_prefix(prefix) else {
				let msg = format!("Parsed as fixed-point type that doesn't start with '{prefix}'");
				return Err(error(&self.source, token.range(), &msg));
			};

			let bits = if bit_spec.is_empty() {
				max_bits / 2
			} else if let Ok(bits) = bit_spec.parse::<u8>() {
				bits
			} else {
				let msg = format!("Unable to parse '{token}' into fixed-point type");
				return Err(error(&self.source, token.range(), &msg));
			};

			if bits > max_bits {
				return Err(expected(self, &format!("Bit specifier between 0..={max_bits}")));
			}

			Ok(bits)
		})
	}

	pub(super) fn value_type(&mut self) -> Result<ValueType> {
		with_ctx!(self, "value_type", {
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
				_ => Err(expected(self, "Value Type")),
			};
			self.index += 1;
			log_item(&token, self.dbg_depth);
			self.dbg_depth -= 2;
			result
		})
	}

	pub(super) fn match_token(&mut self, tt: TokenType) -> Result<()> {
		with_ctx!(self, "match_token", {
			match self.peek(0).tt.clone() {
				t if t != tt => if t == TokenType::Eof {
					Err(report(&self.source, self.peek(-1).range(), "after here",
						&format!("Expected {tt:?}. Found EoF")))
				} else {
					Err(expected(self, &format!("{tt:?}")))
				}
				_ => {
					self.index += 1;
					Ok(())
				}
			}
		})
	}

	pub(super) fn ident_typed(&mut self) -> Result<(Rc<str>, ValueType, TokenInfo)> {
		with_ctx!(self, "ident_typed", {
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
		})
	}

	/// args := (expr (',' expr)* ','?)?
	pub(super) fn args(&mut self) -> Option<Vec<NodeId>> {
		with_ctx!(self, "args", {
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
		})
	}

	/// if := 'if' expr block ('else' block)?
	pub(super) fn stmt_if(&mut self) -> Result<NodeId> {
		with_ctx!(self, "stmt_if", {
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
					.ok_or_else(|| self.dbg_ctx.with_msg("empty scope-list in `parser::stmt_if::true_block`"))?;
				self.nodes.new_block(body, scope, info)
			};
			let t_scopes = self.scopes.clone();

			self.scopes = f_scopes;
			let bf = if self.match_token(TokenType::Else).is_ok() {
				self.scopes.push();
				let (body, info) = self.block()?;
				let scope = self.scopes.pop()
					.ok_or_else(|| self.dbg_ctx.with_msg("empty scope-list in `parser::stmt_if::false_block`"))?;
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
		})
	}

	fn expr_call(&mut self, lhs: NodeId, info: TokenInfo) -> Result<NodeId> {
		with_ctx!(self, "expr_call", {
			let lhs_node = self.nodes.get(lhs)?.clone();
			let name = match lhs_node.expr {
				Expr::Id(name) => name,
				_ => return Err(expected(self, "Identifier")),
			};

			self.index += 1;
			let args = self.args().unwrap_or_default();

			if TokenType::CParen != self.peek(0).tt {
				return Err(expected(self, ")"));
			}
			self.index += 1;

			if self.functions.contains(&name) {
				if let Some(rx) = self.scopes.find(&name) {
					let def = self.nodes.get(rx)?;
					if let Expr::Fun { params,..} = &def.expr {
						assert_eq!(args.len(), params.len(),
							"mismatched argument and parameter lists");
					}
				}
			} else {
				return Err(error(&self.source, info,
					&format!("Call to unknown function '{name}'"),
				));
			}
			Ok(self.nodes.new_call(&name, args, lhs_node.info.start..info.end))
		})
	}

	fn expr2(&mut self, values: &mut Vec<StackValue>) -> StepResult {
		use TokenType as TT;

		let token = self.peek(0);
		match token.tt {
			TT::Rec => {
				// TODO - srenshaw - Remove this placeholder, once we can remove the hybrid expression
				// method.
				values.push(StackValue::Rec);
				self.stack.push(StackOp::Rec);
				"Begin parsing RECORD declaration".into()
			}
			TT::Fun => {
				// TODO - srenshaw - Remove this placeholder, once we can remove the hybrid expression
				// method.
				values.push(StackValue::Fun);
				self.stack.push(StackOp::Fun);
				"Parsed FUNCTION declaration".into()
			}
			TT::If => {
				let nx = match self.stmt_if() {
					Ok(nx) => nx,
					Err(e) => return e.into(),
				};
				// TODO - srenshaw - Add an end marker, so when we're pulling things out of the value stack
				// to add the IF node, we know when to stop.
				//
				// values.push(StackValue::If);
				values.push(StackValue::NodeId(nx));
				"Parsed IF expression".into()
			}
			TT::Var => {
				let nx = match self.stmt_var() {
					Ok(nx) => nx,
					Err(e) => return e.into(),
				};
				// TODO - srenshaw - Add an end marker, so when we're pulling things out of the value stack
				// to add the VARIABLE node, we know when to stop.
				//
				// values.push(StackValue::Var);
				values.push(StackValue::NodeId(nx));
				"Parsed VARIABLE expression".into()
			}
			TT::While => {
				let nx = match self.stmt_while() {
					Ok(nx) => nx,
					Err(e) => return e.into(),
				};
				// TODO - srenshaw - Add an end marker, so when we're pulling things out of the value stack
				// to add the WHILE node, we know when to stop.
				//
				// values.push(StackValue::While);
				values.push(StackValue::NodeId(nx));
				"Parsed WHILE expression".into()
			}
			TT::Ident(ref s) => {
				let s = s.clone();

				// HACK - srenshaw - We probably need a more robust way to distinguish between Record
				// initialization, "ident -> block" sequences, function-calls, and assignment.
				if self.peek(1).tt == TokenType::OParen {
					let info = token.range();
					let id = self.nodes.new_id(&s, ValueType::Any, info.clone());
					self.index += 1;
					let nx = match self.expr_call(id, info) {
						Ok(nx) => nx,
						Err(e) => return e.into(),
					};
					values.push(StackValue::NodeId(nx));
					"Parsed FUNCTION-CALL expression".into()
				} else if self.peek(1).tt == TokenType::OBrace && self.peek(3).tt == TokenType::Colon {
					let nx = match self.expr_rec_init() {
						Ok(nx) => nx,
						Err(e) => return e.into(),
					};
					values.push(StackValue::NodeId(nx));
					"Parsed RECORD initializer expression".into()
				} else if let Some(nx) = self.scopes.find(&s) {
					values.push(StackValue::NodeId(nx));
					format!("Parsed scoped IDENTIFIER '{s}' [{nx}]").into()
				} else {
					let nx = self.nodes.new_id(&s, ValueType::Any, token.range());
					values.push(StackValue::NodeId(nx));
					self.scopes.insert(&s, nx);
					format!("Parsed new IDENTIFIER '{s}' [{nx}]").into()
				}
			}
			_ => {
				match self.expr(0) {
					Ok(nx) => {
						values.push(StackValue::NodeId(nx));
						"Parsed expression".into()
					}
					Err(e) => e.into(),
				}
			}
		}
	}

	pub(super) fn expr(&mut self, min_bp: u8) -> Result<NodeId> {
		with_ctx!(self, "expr", {
			log("Expr", self.dbg_depth);
			self.dbg_depth += 2;
			use TokenType as TT;

			let left_token = self.peek(0);
			let mut lhs: NodeId = match left_token.tt {
				TT::If => self.stmt_if()?,
				TT::Fun => self.stmt_fn()?,
				TT::Rec => self.stmt_rec()?,
				TT::Var => self.stmt_var()?,
				TT::While => self.stmt_while()?,

				TT::True => {
					let token = left_token.clone();
					self.index += 1;
					self.nodes.new_bool(true, token.range())
				}
				TT::False => {
					let token = left_token.clone();
					self.index += 1;
					self.nodes.new_bool(false, token.range())
				}

				TT::Ident(ref s) => {
					if self.peek(1).tt == TT::OBrace && self.peek(3).tt == TT::Colon {
						self.expr_rec_init()?
					} else {
						let token = left_token.clone();
						let s = Rc::clone(s);
						self.index += 1;
						if let Some(nx) = self.scopes.find(&s) {
							nx
						} else {
							let nx = self.nodes.new_id(&s, ValueType::Any, token.range());
							self.scopes.insert(&s, nx)
						}
					}
				}

				TT::Integer(_) => {
					let token = left_token.clone();
					let num = self.num()?;
					self.nodes.new_num(num, ValueType::Int(Int::Bot), token.range())
				}

				TT::Fixed(_) => {
					let token = left_token.clone();
					let num = self.num()?;
					self.nodes.new_num(num, ValueType::Fix(Fix::Bot), token.range())
				}

				TT::OParen => {
					self.index += 1;
					let lhs = self.expr(0)?;
					if TT::CParen != self.peek(0).tt {
						return Err(expected(self, ")"));
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
						return Err(expected(self, "Unary Operator"));
					};
					let token = left_token.clone();
					self.index += 1;
					let rhs = self.expr(r_bp)?;
					self.nodes.new_unary((&token.tt).try_into()?, rhs, token.range())
						.map_err(|err| err.with_source_code(self.source.to_string()))?
				}

				TT::Eof => return Err(eof_error(self, "Identifier, Function Call, or Literal")),

				_ => return Err(expected(self, "Identifier, Function Call, or Literal")),
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
					lhs = self.expr_call(lhs, op_token.range())?;
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
		})
	}

	/// field_init := ident ':' (block | expr)
	fn field_init(&mut self) -> Result<(Rc<str>, NodeId)> {
		with_ctx!(self, "field_init", {
			log("FieldInit", self.dbg_depth);
			self.dbg_depth += 2;
			let id = self.ident()?;
			self.match_token(TokenType::Colon)?;
			let body = if self.peek(1).tt == TokenType::OBrace {
				self.scopes.push();
				let (body, info) = self.block()?;
				let scope = self.scopes.pop()
					.ok_or_else(|| self.dbg_ctx.with_msg("empty scope-list in `parser::field_init`"))?;
				self.nodes.new_block(body, scope, info)
			} else {
				self.expr(0)?
			};
			self.dbg_depth -= 2;
			Ok((id, body))
		})
	}

	/// expr_rec_init := ident '{' field_init* '}'
	pub(super) fn expr_rec_init(&mut self) -> Result<NodeId> {
		with_ctx!(self, "expr_rec_init", {
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
			Ok(self.nodes.new_rec_init(&id, fields, start..end))
		})
	}

	fn block2(&mut self, closing_token: TokenType) -> Result<StepResult> {
		if self.peek(0).tt == closing_token {
			let mut body = vec![];
			while let Some(StackValue::NodeId(nx)) = self.values.last() {
				body.push(*nx);
				self.values.pop();
			}
			return if let Some(scope) = self.scopes.pop() {
				let nx = self.nodes.new_block(body, scope, 0..0);
				self.values.push(StackValue::Block(nx));
				Ok("Finished parsing Block".into())
			} else {
				Err(self.dbg_ctx.with_msg("empty scope-list in `parser::block2`"))
			};
		}

		// We'll need to return here after we try to parse an expression
		self.stack.push(StackOp::Block(closing_token));

		// This is the expression we'll attempt to parse
		self.stack.push(StackOp::Expr);

		Ok("Parsing Expression".into())
	}

	fn param(&mut self, closing_token: TokenType) -> Result<StepResult> {
		if self.peek(0).tt == closing_token {
			// NOTE - srenshaw - Don't consume the closing token, as that will be handled by the
			// return-site.
			return Ok("Finished parsing Parameters".into());
		}

		let start = self.peek(0).range().start;
		let fname = self.ident()?;

		self.match_token(TokenType::Colon)?;

		let vt = self.value_type()?;
		let end = self.peek(-1).range().end;

		if self.match_token(TokenType::Comma).is_ok() {
			self.stack.push(StackOp::Param(closing_token));
		}

		let nx = self.nodes.new_id(&fname, vt.clone(), start..end);
		self.values.push(StackValue::Param(nx));

		Ok(format!("Parsed Parameter '{fname}: {vt}'").into())
	}

	/// params := ( typed_ident (',' typed_ident)* ','? )?
	pub(super) fn params(&mut self) -> Vec<(Rc<str>, ValueType, TokenInfo)> {
		with_ctx!(self, "params", {
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
		})
	}

	/// block := '{' expr* '}'
	pub(super) fn block(&mut self) -> Result<(Vec<NodeId>, TokenInfo)> {
		with_ctx!(self, "block", {
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
		})
	}

	fn rec_start(&mut self) -> Result<StepResult> {
		let start = self.peek(0).start;
		self.match_token(TokenType::Rec)?;

		let name = self.ident()?;
		self.match_token(TokenType::OBrace)?;

		self.values.push(StackValue::Ident(Rc::clone(&name), start));
		self.stack.push(StackOp::RecEnd);
		self.stack.push(StackOp::Param(TokenType::CBrace));
		Ok(format!("Parsed header for RECORD '{name}'").into())
	}

	fn rec_end(&mut self) -> Result<StepResult> {
		let end = self.peek(0).range().end;
		self.match_token(TokenType::CBrace)?;
		let mut params = vec![];
		loop {
			match self.values.pop() {
				Some(StackValue::Param(nx)) => {
					params.push(nx);
				}
				Some(StackValue::Ident(name, start)) => {
					let start = start as usize;
					let nx = self.nodes.new_rec(&name, params, start..end);
					self.scopes.insert(&name, nx);
					self.records.insert(Rc::clone(&name));
					self.ast.push(nx);
					break Ok(format!("Parsed RECORD '{name}' [{nx}]").into());
				}

				Some(StackValue::Block(..)) => break Err(self.dbg_ctx.with_msg("unexpected BLOCK while parsing RECORD")),
				Some(StackValue::NodeId(..)) => break Err(self.dbg_ctx.with_msg("unexpected node id while parsing RECORD")),
				Some(StackValue::Type(..)) => break Err(self.dbg_ctx.with_msg("unexpected value-type while parsing RECORD")),
				None => break Err(self.dbg_ctx.with_msg("empty value stack while parsing RECORD")),

				// Placeholders
				Some(StackValue::Rec) => {}
				Some(StackValue::Fun) => {}
			}
		}
	}

	/// rec := 'rec' ident '{' params '}'
	pub(super) fn stmt_rec(&mut self) -> Result<NodeId> {
		with_ctx!(self, "stmt_rec", {
			log("Record", self.dbg_depth);
			self.dbg_depth += 2;
			let start = self.peek(0).range().start;
			self.match_token(TokenType::Rec)?;
			let name = self.ident()?;
			self.match_token(TokenType::OBrace)?;
			let fields = self.params()
				.into_iter()
				.map(|(fname, ftype, finfo)| self.nodes.new_id(&fname, ftype, finfo))
				.collect();
			self.match_token(TokenType::CBrace)?;
			let end = self.peek(-1).range().end;
			self.dbg_depth -= 2;

			if self.records.contains(&name) {
				return Err(error(&self.source, start..end,
					"A record with this name is already defined"));
			}
			self.records.insert(Rc::clone(&name));

			let nx = self.nodes.new_rec(&name, fields, start..end);
			self.scopes.insert(&name, nx);
			Ok(nx)
		})
	}

	fn fun_start(&mut self) -> Result<StepResult> {
		let start = self.peek(0).start;
		self.match_token(TokenType::Fun)?;
		let name = self.ident()?;
		self.match_token(TokenType::OParen)?;

		self.values.push(StackValue::Ident(Rc::clone(&name), start));
		self.stack.push(StackOp::FunRet);
		self.stack.push(StackOp::Param(TokenType::CParen));

		Ok(format!("Parsed header for FUNCTION '{name}'").into())
	}

	fn fun_return(&mut self) -> Result<StepResult> {
		self.match_token(TokenType::CParen)?;
		let return_type = self.match_token(TokenType::RetArrow)
			.and_then(|_| self.value_type())
			.unwrap_or(ValueType::Unit);
		self.match_token(TokenType::OBrace)?;

		let out = format!("Parsed return-type for FUNCTION '{return_type}'");
		self.values.push(StackValue::Type(return_type));
		self.stack.push(StackOp::FunEnd);
		self.scopes.push();
		self.stack.push(StackOp::Block(TokenType::CBrace));

		Ok(out.into())
	}

	fn fun_end(&mut self) -> Result<StepResult> {
		let end = self.peek(0).range().end;
		self.match_token(TokenType::CBrace)?;

		let Some(StackValue::Block(body)) = self.values.pop() else {
			todo!("expected BLOCK value");
		};

		let Some(StackValue::Type(rtype)) = self.values.pop() else {
			todo!("expected TYPE value");
		};

		let mut params = vec![];
		while let Some(StackValue::Param(nx)) = self.values.last() {
			params.push(*nx);
			self.values.pop();
		}

		let Some(StackValue::Ident(name, start)) = self.values.pop() else {
			todo!("expected IDENTIFIER value");
		};

		let start = start as usize;
		let nx = self.nodes.new_fun(&name, params, rtype, body, start..end);

		self.scopes.insert(&name, nx);
		self.functions.insert(Rc::clone(&name));
		self.ast.push(nx);

		Ok(format!("Parsed FUNCTION '{name}' [{nx}]").into())
	}

	/// fn := 'fn' ident '(' params ')' ('->' value_type)? block
	pub(super) fn stmt_fn(&mut self) -> Result<NodeId> {
		with_ctx!(self, "stmt_fn", {
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
						let px = self.nodes.new_var(pname, ptype.clone(), None, pinfo.clone());
						self.scopes.insert(pname, px)
					})
					.collect();
				let (body,info) = self.block()?;
				let scope = self.scopes.pop()
					.ok_or_else(|| self.dbg_ctx.with_msg("empty scope-list in `parser::stmt_fn`"))?;
				(params, self.nodes.new_block(body, scope, info))
			};

			let end = self.peek(-1).range().end;
			self.dbg_depth -= 2;

			if self.functions.contains(&name) {
				return Err(error(&self.source, start..end,
					"A function with this name is already defined"));
			}
			self.functions.insert(Rc::clone(&name));

			Ok(self.nodes.new_fun(&name, params, rtype, body, start..end))
		})
	}

	/// var := 'var' ident (':' value_type)? '=' (block | expr)
	pub(super) fn stmt_var(&mut self) -> Result<NodeId> {
		with_ctx!(self, "stmt_var", {
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
					.ok_or_else(|| self.dbg_ctx.with_msg("empty scope-list in `parser::stmt_var`"))?;
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
				Ok(self.nodes.new_var(&name, vtype, Some(body), start..end))
			}
		})
	}

	/// while := 'while' expr block
	pub(super) fn stmt_while(&mut self) -> Result<NodeId> {
		with_ctx!(self, "stmt_while", {
			log("While", self.dbg_depth);
			self.dbg_depth += 2;
			let start = self.peek(0).range().start;
			self.match_token(TokenType::While)?;
			let cond = self.expr(0)?;
			let body = {
				self.scopes.push();
				let (body, info) = self.block()?;
				let scope = self.scopes.pop()
					.ok_or_else(|| self.dbg_ctx.with_msg("empty scope-list in `parser::stmt_while`"))?;
				self.nodes.new_block(body, scope, info)
			};
			let end = self.peek(-1).range().end;
			self.dbg_depth -= 2;
			Ok(self.nodes.new_while(cond, body, start..end))
		})
	}
}

impl ScopeTracker {
	/// Used in type-checking (and potentially elsewhere) to re-add block scopes.
	pub fn add(&mut self, scope: Scope) {
		self.0.push(scope)
	}

	fn insert(&mut self, name: &Rc<str>, nx: NodeId) -> NodeId {
		// let index = self.0.len();
		if let Some(scope) = self.0.last_mut() {
			// println!("add {name} : {nx} to scope {}", index-1);
			scope.insert(Rc::clone(name), nx);
			nx
		} else {
			panic!("add {name} : {nx} with no base scope");
		}
	}

	fn push(&mut self) {
		// println!("pushing a scope | prev-top {:?}", self.0.last());
		self.0.push(Scope::default());
	}

	fn pop(&mut self) -> Option<Scope> {
		// let last = self.0.pop();
		// println!("popping a scope | prev-top {last:?}");
		// last
		self.0.pop()
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
	) -> Result<ScopeTracker> {
		// println!("merging scopes");

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

