
use std::collections::HashMap;
use std::rc::Rc;

use miette::{LabeledSpan, IntoDiagnostic, WrapErr};

use crate::tokens::{Token, TokenType};

use super::{BinaryOp, Expr, Fix, Int, UnaryOp, ValueType};
use super::error::Error;
use super::node::{NodeId, NodeStore};

pub(super) type Result<T> = std::result::Result<T, Error>;

fn eof_error(parser: &Parser, msg: &str) -> Error {
	Error::report(&parser.source, parser.peek(-1).range(), "after here",
		&format!("Expected {msg}, Found EoF"))
}

fn error(parser: &Parser, msg: &str) -> Error {
	Error::report(&parser.source, parser.peek(0).range(), "here", msg)
}

fn expected(parser: &Parser, msg: &str) -> Error {
	Error::Report(miette::miette! {
		labels = [
			LabeledSpan::at(parser.peek(-1).range(), "between here"),
			LabeledSpan::at(parser.peek(0).range(), "and here"),
		],
		"Expected {msg}, found {}", parser.peek(0)
	}.with_source_code(parser.source.to_string()))
}

pub(crate) type Scope = HashMap<Rc<str>, (ValueType, NodeId)>;

#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct ScopeTracker(Vec<Scope>);

// struct FunDef {
// 	params: Vec<(Rc<str>, ValueType)>,
// 	rtype: ValueType,
// }

// type FunDefStorage = HashMap<Rc<str>, FunDef>;

// #[derive(Debug)]
// struct RecDef {
// 	fields: Vec<(Rc<str>, ValueType)>,
// }

// type RecDefStorage = HashMap<Rc<str>, RecDef>;

#[derive(Debug)]
pub(crate) enum StackOp {
	Param(TokenType),
	Block(TokenType, usize),
	Ident(Rc<str>),
	Expr(u8),
	ExprEnd(u8),
	BinaryOp(BinaryOp),
	UnaryOp(UnaryOp),

	// RecDef,
	// RecDefEnd,
	// RecInit,
	// RecInitParams(usize),
	// RecInitEnd(usize),
	// Fun,
	// FunRet,
	// FunEnd,
	// Call,
	// CallArg,
	// CallEnd,
	Var,
	VarEnd,
	While,
	WhileBody,
	WhileEnd,
	If,
	IfThen,
	IfElse,
	IfEnd,
}

#[derive(Debug, Clone)]
pub(crate) enum StackValue {
	NodeId(NodeId),
	Ident(Rc<str>),
	InfoStart(u16),
	Info(u16,u16),
	Type(ValueType),
	Scope(ScopeTracker),
}

#[derive(Debug)]
pub(crate) struct Parser {
	pub(crate) input: Box<[Token]>,
	pub(crate) source: Rc<str>,
	pub(crate) index: usize,

	pub(crate) scopes: ScopeTracker,

	// TODO - srenshaw - All of these fields will probably be moved into a Scope structure.
	pub(crate) nodes: NodeStore,
	// pub(crate) fun_defs: HashSet<Rc<str>>, //FunDefStorage,
	// pub(crate) rec_defs: RecDefStorage,

	// Stack
	pub(crate) stack: Vec<StackOp>,
	pub(crate) values: Vec<StackValue>,
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
			// rec_defs: RecDefStorage::default(),
			// fun_defs: HashSet::default(),

			stack: vec![StackOp::Block(TokenType::Eof, 0)],
			values: vec![],
		}
	}

	const EOF: Token = Token { tt: TokenType::Eof, start: 0 };
	pub fn peek(&self, offset: isize) -> &Token {
		self.input.get(self.index.saturating_add_signed(offset))
			.unwrap_or(&Self::EOF)
	}
}

pub(crate) enum Step {
	Next(String),
	Done,
}

impl From<&str> for Step {
	fn from(s: &str) -> Self {
		Self::Next(s.into())
	}
}

impl From<String> for Step {
	fn from(s: String) -> Self {
		Self::Next(s)
	}
}

impl Parser {
	#[inline]
	pub(crate) fn step_and_continue(
		&mut self,
		continue_on_error: bool,
	) -> bool {
		self.step_with_action(
			|s,_| {  println!("{s}"); true },
			|s,_| {  println!("{s}"); false },
			|s,_| { eprintln!("{s}"); continue_on_error },
			&mut String::new(),
		)
	}

	#[inline]
	pub(crate) fn step_with_action<T>(
		&mut self,
		on_next: impl FnOnce(String, &mut T) -> bool,
		on_done: impl FnOnce(String, &mut T) -> bool,
		on_erep: impl FnOnce(String, &mut T) -> bool,
		out: &mut T,
	) -> bool {
		match self.step() {
			Ok(Step::Next(s)) => {
				on_next(format!("[step] {s}"), out)
			}
			Ok(Step::Done) => {
				on_done(format!("[done] FIN"), out)
			}
			Err(Error::Report(e)) => {
				on_erep(format!("[erep] {e}"), out)
			}
		}
	}

	pub(crate) fn step(&mut self) -> Result<Step> {
		match self.stack.pop() {
			Some(StackOp::Param(closing_token)) => self.param(closing_token),

			Some(StackOp::Block(closing_token, count)) => self.block(closing_token, count),

			Some(StackOp::Ident(s)) => self.ident(s),

			Some(StackOp::Expr(min_bp)) => self.expr_start(min_bp),
			Some(StackOp::ExprEnd(min_bp)) => self.expr_end(min_bp),

			// Some(StackOp::RecDef) => self.rec_start(),
			// Some(StackOp::RecDefEnd) => self.rec_end(),

			// Some(StackOp::RecInit) => self.rec_init_start(),
			// Some(StackOp::RecInitParams(count)) => self.rec_init_params(count),
			// Some(StackOp::RecInitEnd(count)) => self.rec_init_end(count),

			// Some(StackOp::Fun) => self.fun_start(),
			// Some(StackOp::FunRet) => self.fun_return(),
			// Some(StackOp::FunEnd) => self.fun_end(),

			// Some(StackOp::Call) => self.call_start(),
			// Some(StackOp::CallArg) => self.call_arg(),
			// Some(StackOp::CallEnd) => self.call_end(),

			Some(StackOp::Var) => self.var_start(),
			Some(StackOp::VarEnd) => self.var_end(),

			Some(StackOp::While) => self.while_start(),
			Some(StackOp::WhileBody) => self.while_body(),
			Some(StackOp::WhileEnd) => self.while_end(),

			Some(StackOp::If) => self.if_start(),
			Some(StackOp::IfThen) => self.if_then(),
			Some(StackOp::IfElse) => self.if_else(),
			Some(StackOp::IfEnd) => self.if_end(),

			Some(StackOp::BinaryOp(op)) => self.binary_op(op),

			Some(StackOp::UnaryOp(op)) => self.unary_op(op),

			None => Ok(Step::Done),
		}
	}

	pub fn finish(mut self) -> std::result::Result<super::Output, Error> {
		let start = match self.values.pop() {
			Some(StackValue::NodeId(start)) => start,
			value => return Err(error(&self, &format!("found value '{value:?}' instead of final block"))),
		};

		Ok(super::Output {
			start,
			store: self.nodes,
			// records: self.rec_defs.keys().cloned().collect(),
			// functions: self.fun_defs,
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
		TT::Colon | TT::Comma | TT::CBrace | TT::CParen | TT::OBrace | TT::Semicolon | TT::RetArrow |
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

impl Parser {
	pub(super) fn num(&mut self) -> Result<i64> {
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
		out
	}

	pub(super) fn parse_ident(&mut self) -> Result<Rc<str>> {
		let out = match self.peek(0).tt.clone() {
			TokenType::Ident(s) => {
				self.index += 1;
				Ok(s)
			}
			TokenType::Eof => Err(eof_error(self, "Identifier")),
			_ => Err(expected(self, "Identifier")),
		};
		out
	}

	pub(super) fn parse_fixed_point(&mut self, prefix: &str, max_bits: u8) -> Result<u8> {
		let token = self.peek(0);
		let token_str = token.to_string();
		let Some(bit_spec) = token_str.strip_prefix(prefix) else {
			let msg = format!("Parsed as fixed-point type that doesn't start with '{prefix}'");
			return Err(error(self, &msg));
		};

		let bits = if bit_spec.is_empty() {
			max_bits / 2
		} else if let Ok(bits) = bit_spec.parse::<u8>() {
			bits
		} else {
			let msg = format!("Unable to parse '{token}' into fixed-point type");
			return Err(error(self, &msg));
		};

		if bits > max_bits {
			return Err(expected(self, &format!("Bit specifier between 0..={max_bits}")));
		}

		Ok(bits)
	}

	pub(super) fn value_type(&mut self) -> Result<ValueType> {
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
		result
	}

	pub(super) fn match_token(&mut self, tt: TokenType) -> Result<()> {
		match self.peek(0).tt.clone() {
			t if t != tt => if t == TokenType::Eof {
				Err(Error::report(&self.source, self.peek(-1).range(), "after here",
					&format!("Expected {tt:?}. Found EoF")))
			} else {
				Err(expected(self, &format!("{tt:?}")))
			}
			_ => {
				self.index += 1;
				Ok(())
			}
		}
	}

	fn if_start(&mut self) -> Result<Step> {
		let start = self.peek(-1).start;
		self.values.push(StackValue::InfoStart(start));
		self.stack.push(StackOp::IfThen);
		self.stack.push(StackOp::Expr(0));
		Ok("Parsing IF expression".into())
	}

	fn if_then(&mut self) -> Result<Step> {
		self.match_token(TokenType::OBrace)?;
		self.values.push(StackValue::Scope(self.scopes.clone()));
		self.stack.push(StackOp::IfElse);
		self.stack.push(StackOp::Block(TokenType::CBrace, 0));
		Ok("Parsed header for IF expression".into())
	}

	fn if_else(&mut self) -> Result<Step> {
		let end = self.peek(0).range().end;
		self.match_token(TokenType::CBrace)?;

		if self.match_token(TokenType::Else).is_ok() {
			self.match_token(TokenType::OBrace)?;

			let (then_body, f_scopes) = self.get_if_tail("if_else")?;

			let t_scopes = self.scopes.clone();
			self.scopes = f_scopes;

			self.values.push(StackValue::NodeId(then_body));
			self.values.push(StackValue::Scope(t_scopes));
			self.stack.push(StackOp::IfEnd);
			self.stack.push(StackOp::Block(TokenType::CBrace, 0));
			Ok("Parsed then-branch for IF expression".into())
		} else {
			let (then_body, mut f_scopes) = self.get_if_tail("if_else")?;
			f_scopes.push();

			let mut t_scopes = self.scopes.clone();
			let (start, cond, t_scope) = self.get_if_header("if_else", then_body)?;

			t_scopes.add(t_scope);

			self.scopes = ScopeTracker::merge(
				&mut self.nodes,
				t_scopes,
				f_scopes,
			)?;

			let nx = self.nodes.new_if(cond, then_body, None, start as usize..end);
			self.values.push(StackValue::NodeId(nx));
			Ok("Parsed IF expression".into())
		}
	}

	fn if_end(&mut self) -> Result<Step> {
		let end = self.peek(0).range().end;
		self.match_token(TokenType::CBrace)?;

		let (else_body, mut t_scopes) = self.get_if_tail("if_end")?;

		let Some(StackValue::NodeId(then_body)) = self.values.pop() else {
			return Err(error(self, "Expected NODE-ID value in 'if_end'"));
		};

		let (start, cond, f_scope) = self.get_if_header("if_end", else_body)?;

		let then_node = self.nodes.get(then_body);
		if then_node.expr() != &Expr::Block {
			return Err(error(self, "Expected BLOCK expression in 'if_end'"));
		};

		t_scopes.add(then_node.scope().clone());
		self.scopes.add(f_scope);

		self.scopes = ScopeTracker::merge(
			&mut self.nodes,
			t_scopes,
			self.scopes.clone(),
		)?;

		let nx = self.nodes.new_if(cond, then_body, Some(else_body), start as usize..end);
		self.values.push(StackValue::NodeId(nx));
		Ok("Parsed IF_ELSE expression".into())
	}

	fn get_if_tail(&mut self, parser_name: &str) -> Result<(NodeId, ScopeTracker)> {
		let Some(StackValue::NodeId(body)) = self.values.pop() else {
			return Err(error(self, &format!("Expected NODE-ID value in '{parser_name}'")));
		};

		let Some(StackValue::Scope(scopes)) = self.values.pop() else {
			return Err(error(self, &format!("Expected SCOPE value in '{parser_name}'")));
		};

		Ok((body, scopes))
	}

	fn get_if_header(&mut self, parser_name: &str, body: NodeId) -> Result<(u16, NodeId, Scope)> {
		let Some(StackValue::NodeId(cond)) = self.values.pop() else {
			return Err(error(self, &format!("Expected NODE-ID value in '{parser_name}'")));
		};

		let Some(StackValue::InfoStart(start)) = self.values.pop() else {
			return Err(error(self, &format!("Expected INFO value in '{parser_name}'")));
		};

		let node = self.nodes.get(body);
		let scope = node.scope();

		Ok((start, cond, scope.clone()))
	}

	pub(crate) fn expr_start(&mut self, min_bp: u8) -> Result<Step> {
		use TokenType as TT;

		self.stack.push(StackOp::ExprEnd(min_bp));

		let token = self.peek(0);
		match token.tt {
			// TT::Rec => {
			// 	self.index += 1;
			// 	self.stack.push(StackOp::RecDef);
			// 	Ok("Found RECORD declaration".into())
			// }
			// TT::Fun => {
			// 	self.index += 1;
			// 	self.stack.push(StackOp::Fun);
			// 	Ok("Found FUNCTION declaration".into())
			// }
			TT::If => {
				self.index += 1;
				self.stack.push(StackOp::If);
				Ok("Found IF expression".into())
			}
			TT::Var => {
				self.index += 1;
				self.stack.push(StackOp::Var);
				Ok("Found VARIABLE declaration".into())
			}
			TT::While => {
				self.index += 1;
				self.stack.push(StackOp::While);
				Ok("Found WHILE expression".into())
			}

			TT::Ident(ref s) => {
				let s = s.clone();
				self.index += 1;
				let out = format!("Found IDENTIFIER '{s}'");
				self.stack.push(StackOp::Ident(s));
				Ok(out.into())
			}

			TT::Integer(_) => {
				let info = token.range();
				let num = self.num()?;
				let nx = self.nodes.new_num(num, ValueType::Int(Int::Bot), info);
				self.values.push(StackValue::NodeId(nx));
				Ok(format!("Parsed INTEGER '{num}' [{nx:?}]").into())
			}

			TT::Fixed(_) => {
				let info = token.range();
				let num = self.num()?;
				let nx = self.nodes.new_num(num, ValueType::Fix(Fix::Bot), info);
				self.values.push(StackValue::NodeId(nx));
				Ok(format!("Parsed FIXED-POINT '{num}' [{nx:?}]").into())
			}

			TT::True => {
				let info = token.range();
				self.index += 1;
				let nx = self.nodes.new_bool(true, info);
				self.values.push(StackValue::NodeId(nx));
				Ok(format!("Parsed TRUE [{nx:?}]").into())
			}
			TT::False => {
				let info = token.range();
				self.index += 1;
				let nx = self.nodes.new_bool(false, info);
				self.values.push(StackValue::NodeId(nx));
				Ok(format!("Parsed FALSE [{nx:?}]").into())
			}

			TT::Plus |
			TT::Minus |
			TT::Dollar |
			TT::At |
			TT::Bang => {
				let Some(r_bp) = prefix_binding_power(&token.tt) else {
					return Err(expected(self, "Unary Operator ['+', '-', '$', '@', '!']"));
				};
				let unary_op : UnaryOp = token.tt.clone().try_into()?;
				self.index += 1;
				self.stack.push(StackOp::UnaryOp(unary_op));
				self.stack.push(StackOp::Expr(r_bp));
				Ok(format!("Parsed UNARY OP '{unary_op}'").into())
			}

			TT::Semicolon => {
				// End expression
				Ok("Found ';'".into())
			}
			TT::OBrace => {
				// Open block
				self.index += 1;
				self.stack.push(StackOp::Block(TokenType::CBrace, 0));
				Ok("Found '{'".into())
			}
			TT::CBrace => {
				// End expression - return to parsing block
				Ok("Found '}'".into())
			}
			TT::OParen => {
				// New expression
				self.index += 1;
				self.stack.push(StackOp::Expr(0));
				Ok("Found '('".into())
			}

			/* ERROR values */
			TT::CParen => {
				Err(expected(self, "Expression value ['if', 'while', 'rec', 'fn', 'val', 'var', identifier, number ]"))
			}

			_ => panic!("Found TOKEN '{token}'\n{self:?}"),
		}
	}

	fn expr_end(&mut self, min_bp: u8) -> Result<Step> {
		use TokenType as TT;

		let token = self.peek(0).clone();
		match token.tt {
			TT::Ident(_) | TT::Integer(_) | TT::Fixed(_) |
			TT::If | TT::Else | TT::While |
			TT::Fun | TT::Rec | TT::Var |
			TT::U8 | TT::U16 | TT::U32 |
			TT::S8 | TT::S16 | TT::S32 |
			TT::F16(_) | TT::F32(_) |
			TT::OBrace | TT::CBrace |
			TT::Colon | TT::Comma |
			TT::Eof => Ok("End of expression".into()),

			TT::CParen |
			TT::Semicolon => {
				self.index += 1;
				Ok("End of expression".into())
			}

			tt => {
				if let Some((l_bp, r_bp)) = infix_binding_power(&tt) {
					if l_bp < min_bp {
						return Ok("End of sub-expression".into());
					}

					let op: BinaryOp = tt.try_into()?;
					self.index += 1;
					self.stack.push(StackOp::ExprEnd(min_bp));
					self.stack.push(StackOp::BinaryOp(op));
					self.stack.push(StackOp::Expr(r_bp));
					Ok("Continuing expression parsing - binop".into())
				} else {
					panic!("unexpected token '{tt:?}'");
				}
			}
		}
	}

	fn ident(&mut self, id: Rc<str>) -> Result<Step> {
		// let start = self.peek(-1).start;

		// if self.peek(0).tt == TokenType::OParen {
		// 	self.values.push(StackValue::InfoStart(start as u16));
		// 	self.values.push(StackValue::Ident(id));
		// 	self.stack.push(StackOp::Call);
		// 	Ok("Starting FUNCTION-CALL expression".into())
		// } else if self.peek(0).tt == TokenType::OBrace {
		// 	self.values.push(StackValue::InfoStart(start as u16));
		// 	self.values.push(StackValue::Ident(id));
		// 	self.stack.push(StackOp::RecInit);
		// 	Ok("Starting RECORD initializer expression".into())
		// } else {
			let nx = self.nodes.new_id(Rc::clone(&id), self.peek(-1).range());
			self.values.push(StackValue::NodeId(nx));
			Ok(format!("Parsed IDENTIFIER '{id}' [{nx:?}]").into())
		// }
	}

	fn block(&mut self, closing_token: TokenType, count: usize) -> Result<Step> {
		let inc_count = self.peek(0).tt != TokenType::Var;

		// NOTE - srenshaw - Don't consume the closing token, as that will be
		// handled by the return-site.
		if self.peek(0).tt == closing_token {
			let mut body = None;
			for i in 0..count {
				let Some(StackValue::NodeId(nx)) = self.values.last() else {
					return Err(error(self, &format!("expected {count} items in block, found {i}")));
				};
				body = Some(*nx);
				self.values.pop();
			}

			if let Some(scope) = self.scopes.pop() {
				let nx = self.nodes.new_block(body, scope, 0..0);
				self.values.push(StackValue::NodeId(nx));
				Ok("Finished parsing Block".into())
			} else {
				Err(error(self, "empty scope-list in `parser::block2`"))
			}
		} else {
			// We'll need to return here after we try to parse an expression
			self.stack.push(StackOp::Block(closing_token,
				if inc_count { count + 1 } else { count }));

			// This is the expression we'll attempt to parse
			self.stack.push(StackOp::Expr(0));

			Ok("Parsing Expression".into())
		}
	}

	fn param(&mut self, closing_token: TokenType) -> Result<Step> {
		if self.peek(0).tt == closing_token {
			// NOTE - srenshaw - Don't consume the closing token, as that will be handled by the
			// return-site.
			return Ok("Finished parsing Parameters".into());
		}

		let start = self.peek(0).range().start;
		let fname = self.parse_ident()?;

		self.match_token(TokenType::Colon)?;

		let vt = self.value_type()?;
		let end = self.peek(-1).range().end;

		if self.match_token(TokenType::Comma).is_ok() {
			self.stack.push(StackOp::Param(closing_token));
		}

		self.values.push(StackValue::Info(start as u16, end as u16));
		self.values.push(StackValue::Type(vt.clone()));
		self.values.push(StackValue::Ident(fname.clone()));

		Ok(format!("Parsed Parameter '{fname}: {vt}'").into())
	}

	// fn rec_start(&mut self) -> Result<Step> {
	// 	let start = self.peek(-1).start;
	//
	// 	let name = self.parse_ident()?;
	// 	self.match_token(TokenType::OBrace)?;
	//
	// 	let out = format!("Parsed header for RECORD '{name}'");
	// 	self.values.push(StackValue::Ident(name));
	// 	self.values.push(StackValue::InfoStart(start));
	// 	self.stack.push(StackOp::RecDefEnd);
	// 	self.stack.push(StackOp::Param(TokenType::CBrace));
	// 	Ok(out.into())
	// }

	// fn rec_end(&mut self) -> Result<Step> {
	// 	let end = self.peek(0).range().end;
	// 	self.match_token(TokenType::CBrace)?;
	//
	// 	let mut params = vec![];
	// 	while let Some(StackValue::Ident(name)) = self.values.last() {
	// 		let name = Rc::clone(name);
	// 		self.values.pop();
	// 		let Some(StackValue::Type(vtype)) = self.values.pop() else {
	// 			return Err(error(self, "expected TYPE for param in 'rec_end'"));
	// 		};
	// 		let Some(StackValue::Info(_start, _end)) = self.values.pop() else {
	// 			return Err(error(self, "expected INFO for param in 'rec_end'"));
	// 		};
	// 		params.push((name, vtype));
	// 	}
	//
	// 	let Some(StackValue::InfoStart(start)) = self.values.pop() else {
	// 		return Err(error(self, "Expected INFO value in 'rec_end'"));
	// 	};
	//
	// 	let Some(StackValue::Ident(name)) = self.values.pop() else {
	// 		return Err(error(self, "Expected IDENTIFIER value in 'rec_end'"));
	// 	};
	//
	// 	let nx = self.nodes.new_rec(&name, params, start as usize..end);
	// 	let rd = RecDef { fields: params };
	//
	// 	let out = format!("Parsed RECORD '{name}' [{nx:?}]");
	// 	self.scopes.insert(&name, nx);
	// 	self.rec_defs.insert(name, rd);
	// 	self.ast.push(nx);
	// 	Ok(out.into())
	// }

	// fn rec_init_start(&mut self) -> Result<Step> {
	// 	self.match_token(TokenType::OBrace)?;
	// 	if self.match_token(TokenType::CBrace).is_ok() {
	// 		self.stack.push(StackOp::RecInitEnd(0));
	// 		Ok("Parsed header for REC-INIT".into())
	// 	} else {
	// 		self.rec_init_param(0)
	// 	}
	// }

	// fn rec_init_params(&mut self, count: usize) -> Result<Step> {
	// 	if self.match_token(TokenType::Comma).is_ok() {
	// 		self.rec_init_param(count)
	// 	} else {
	// 		self.match_token(TokenType::CBrace)?;
	// 		self.stack.push(StackOp::RecInitEnd(count));
	// 		Ok("End of REC-INIT params".into())
	// 	}
	// }

	// fn rec_init_param(&mut self, count: usize) -> Result<Step> {
	// 	let name = self.parse_ident()?;
	// 	self.match_token(TokenType::Colon)?;
	// 	self.values.push(StackValue::Ident(name));
	// 	self.stack.push(StackOp::RecInitParams(count + 1));
	// 	self.stack.push(StackOp::Expr(0));
	// 	Ok("Parsed REC-INIT param".into())
	// }

	// fn rec_init_end(&mut self, count: usize) -> Result<Step> {
	// 	let end = self.peek(-1).range().end;
	//
	// 	let mut params = vec![];
	// 	for i in 0..count {
	// 		let Some(StackValue::NodeId(expr)) = self.values.pop() else {
	// 			return Err(error(self, &format!("expected {count} expressions in block, found {i}")));
	// 		};
	//
	// 		let Some(StackValue::Ident(name)) = self.values.pop() else {
	// 			return Err(error(self, &format!("expected {count} names in block, found {i}")));
	// 		};
	//
	// 		params.push((name, expr));
	// 	}
	//
	// 	let Some(StackValue::Ident(name)) = self.values.pop() else {
	// 		todo!()
	// 	};
	//
	// 	let Some(StackValue::InfoStart(start)) = self.values.pop() else {
	// 		todo!()
	// 	};
	//
	// 	for (field_name, field_expr) in &params {
	// 		self.scopes.insert(&format!("{name}-{field_name}").into(), *field_expr);
	// 	}
	//
	// 	let nx = self.nodes.new_rec_init(&name, params, start as usize..end);
	// 	self.values.push(StackValue::NodeId(nx));
	// 	Ok("Parsed REC-INIT expression".into())
	// }

	// fn fun_start(&mut self) -> Result<Step> {
	// 	let start = self.peek(-1).start;
	//
	// 	let name = self.parse_ident()?;
	// 	self.match_token(TokenType::OParen)?;
	//
	// 	let out = format!("Parsed header for FUNCTION '{name}'");
	// 	self.values.push(StackValue::Ident(name));
	// 	self.values.push(StackValue::InfoStart(start));
	// 	self.stack.push(StackOp::FunRet);
	// 	self.stack.push(StackOp::Param(TokenType::CParen));
	// 	Ok(out.into())
	// }

	// fn fun_return(&mut self) -> Result<Step> {
	// 	self.match_token(TokenType::CParen)?;
	// 	let return_type = self.match_token(TokenType::RetArrow)
	// 		.and_then(|_| self.value_type())
	// 		.unwrap_or(ValueType::Unit);
	// 	self.match_token(TokenType::OBrace)?;
	//
	// 	let out = format!("Parsed return-type for FUNCTION '{return_type}'");
	// 	self.values.push(StackValue::Type(return_type));
	// 	self.stack.push(StackOp::FunEnd);
	// 	// Start a new scope for the function body
	// 	self.scopes.push();
	// 	self.stack.push(StackOp::Block(TokenType::CBrace, 0));
	//
	// 	Ok(out.into())
	// }

	// fn fun_end(&mut self) -> Result<Step> {
	// 	let end = self.peek(0).range().end;
	// 	self.match_token(TokenType::CBrace)?;
	//
	// 	let Some(StackValue::NodeId(body)) = self.values.pop() else {
	// 		return Err(error(self, "expected NODE_ID value in 'fun_end'"));
	// 	};
	//
	// 	let Some(StackValue::Type(rtype)) = self.values.pop() else {
	// 		return Err(error(self, "expected TYPE value in 'fun_end'"));
	// 	};
	//
	// 	let mut params = vec![];
	// 	while let Some(StackValue::Ident(name)) = self.values.last() {
	// 		let name = Rc::clone(name);
	// 		self.values.pop();
	// 		let Some(StackValue::Type(vtype)) = self.values.pop() else {
	// 			return Err(error(self, "expected TYPE for param in 'fun_end'"));
	// 		};
	// 		let Some(StackValue::Info(_start, _end)) = self.values.pop() else {
	// 			return Err(error(self, "expected INFO for param in 'fun_end'"));
	// 		};
	// 		params.push((name, vtype));
	// 	}
	//
	// 	let Some(StackValue::InfoStart(start)) = self.values.pop() else {
	// 		return Err(error(self, "expected INFO value in 'fun_end'"));
	// 	};
	//
	// 	let Some(StackValue::Ident(name)) = self.values.pop() else {
	// 		return Err(error(self, "expected IDENTIFIER value in 'fun_end'"));
	// 	};
	//
	// 	let start = start as usize;
	// 	let nx = self.nodes.new_fun(&name, params, rtype, body, start..end);
	//
	// 	let out = format!("Parsed FUNCTION '{name}' [{nx:?}]");
	// 	self.scopes.insert(&name, nx);
	// 	self.fun_defs.insert(name);
	// 	self.ast.push(nx);
	// 	Ok(out.into())
	// }

	// fn call_start(&mut self) -> Result<Step> {
	// 	self.match_token(TokenType::OParen)?;
	// 	if self.match_token(TokenType::CParen).is_ok() {
	// 		// no function arguments, jump to call-end
	// 		self.stack.push(StackOp::CallEnd);
	// 	} else {
	// 		self.stack.push(StackOp::CallArg);
	// 		self.stack.push(StackOp::Expr(0));
	// 	}
	// 	Ok("parsed header for FN-CALL".into())
	// }

	// fn call_arg(&mut self) -> Result<Step> {
	// 	if self.match_token(TokenType::Comma).is_ok() {
	// 		self.stack.push(StackOp::CallArg);
	// 		self.stack.push(StackOp::Expr(0));
	// 		Ok("parsed FN-CALL arg".into())
	// 	} else {
	// 		self.stack.push(StackOp::CallEnd);
	// 		Ok("finished FN-CALL args".into())
	// 	}
	// }

	// fn call_end(&mut self) -> Result<Step> {
	// 	let end = self.peek(-1).range().end;
	//
	// 	let mut args = vec![];
	// 	while let Some(StackValue::NodeId(nx)) = self.values.last() {
	// 		args.push(*nx);
	// 		self.values.pop();
	// 	}
	//
	// 	let Some(StackValue::Ident(name)) = self.values.pop() else {
	// 		return Err(error(self, "missing name for FN-CALL"));
	// 	};
	//
	// 	let Some(StackValue::InfoStart(start)) = self.values.pop() else {
	// 		return Err(error(self, "missing info for FN-CALL"));
	// 	};
	//
	// 	let nx = self.nodes.new_call(&name, args, start as usize..end);
	// 	self.values.push(StackValue::NodeId(nx));
	// 	Ok(format!("Finished parsing FN-CALL '{name}' [{nx:?}]").into())
	// }

	fn var_start(&mut self) -> Result<Step> {
		let start = self.peek(-1).start;

		let name = self.parse_ident()?;
		let vtype = self.match_token(TokenType::Colon)
			.and_then(|_| self.value_type())
			.unwrap_or(ValueType::Any);
		self.match_token(TokenType::Eq1)?;

		let out = format!("Parsed header for VARIABLE '{name}'");
		// Push the header marker.
		self.values.push(StackValue::InfoStart(start));
		self.values.push(StackValue::Ident(name));
		self.values.push(StackValue::Type(vtype));

		if self.match_token(TokenType::OBrace).is_ok() {
			// Finish parsing the variable after we're done with the block.
			self.stack.push(StackOp::VarEnd);
			// Start a new scope for the variable block.
			self.scopes.push();
			self.stack.push(StackOp::Block(TokenType::CBrace, 0));
		} else {
			// Finish parsing the variable after we're done with the body.
			self.stack.push(StackOp::VarEnd);
			self.stack.push(StackOp::Expr(0));
		}

		Ok(out.into())
	}

	fn var_end(&mut self) -> Result<Step> {
		let _end = self.peek(-1).range().end;

		let Some(StackValue::NodeId(body)) = self.values.pop() else {
			return Err(error(self, "Expected NODE_ID value in 'var_end'"));
		};

		let Some(StackValue::Type(vtype)) = self.values.pop() else {
			return Err(error(self, "Expected TYPE value in 'var_end'"));
		};

		let Some(StackValue::Ident(name)) = self.values.pop() else {
			return Err(error(self, "Expected IDENTIFIER value in 'var_end'"));
		};

		let Some(StackValue::InfoStart(_start)) = self.values.pop() else {
			return Err(error(self, "expected INFO value in 'var_end'"));
		};

		// Add to the current scope.
		self.scopes.insert(&name, vtype, body);

		Ok(format!("Parsed VARIABLE '{name}' [{body:?}]").into())
	}

	fn while_start(&mut self) -> Result<Step> {
		let start = self.peek(-1).start;
		self.values.push(StackValue::InfoStart(start));
		self.stack.push(StackOp::WhileBody);
		self.stack.push(StackOp::Expr(0));
		Ok("Parsed header for WHILE loop".into())
	}

	fn while_body(&mut self) -> Result<Step> {
		self.match_token(TokenType::OBrace)?;
		self.stack.push(StackOp::WhileEnd);
		self.scopes.push();
		self.stack.push(StackOp::Block(TokenType::CBrace, 0));
		Ok("Parsed condition for WHILE loop".into())
	}

	fn while_end(&mut self) -> Result<Step> {
		let end = self.peek(0).range().end;
		self.match_token(TokenType::CBrace)?;

		let Some(StackValue::NodeId(body)) = self.values.pop() else {
			return Err(error(self, "Expected BODY value in 'while_end'"));
		};

		let Some(StackValue::NodeId(cond)) = self.values.pop() else {
			return Err(error(self, "Expected COND value in 'while_end'"));
		};

		let Some(StackValue::InfoStart(start)) = self.values.pop() else {
			return Err(error(self, "Expected INFO value in 'while_end'"));
		};

		let nx = self.nodes.new_while(cond, body, start as usize..end);
		self.values.push(StackValue::NodeId(nx));
		Ok("Parsed WHILE loop".into())
	}

	fn binary_op(&mut self, op: BinaryOp) -> Result<Step> {
		let Some(StackValue::NodeId(right)) = self.values.pop() else {
			return Err(error(self, "missing right-operand for BINARY-OP"));
		};
		let Some(StackValue::NodeId(left)) = self.values.pop() else {
			return Err(error(self, "missing left-operand for BINARY-OP"));
		};

		let rnode = self.nodes.get(right);
		let lnode = self.nodes.get(left);

		if op == BinaryOp::Accessor {
			match (lnode.expr(), rnode.expr()) {
				(Expr::RecInit, Expr::Id) => {
					let Some((_,nx)) = self.scopes.find(&format!("{}-{}", lnode.name(), rnode.name()).into()) else {
						return Err(error(self, &format!("failed ACCESS '{}.{}'", lnode.name(), rnode.name())));
					};
					self.values.push(StackValue::NodeId(nx));
					return Ok(format!("Parsed ACCESS for '{}.{}'", lnode.name(), rnode.name()).into());
				}
				(Expr::Id, rexpr) => {
					return Err(error(self, &format!("failed ACCESS '{}.{rexpr:?}'", lnode.name())));
				}
				(lexpr, Expr::Id) => {
					return Err(error(self, &format!("failed ACCESS '{lexpr:?}.{}'", rnode.name())));
				}
				(lexpr, rexpr) => {
					return Err(error(self, &format!("failed ACCESS '{lexpr:?}.{rexpr:?}'")));
				}
			}
		}

		let end = rnode.info().end;
		let start = lnode.info().start;
		let nx = self.nodes.new_binary(op, left, right, start..end)
			.expect("unable to create new BINARY-OP");
		self.values.push(StackValue::NodeId(nx));
		Ok("Parsed BINARY-OP".into())
	}

	fn unary_op(&mut self, op: UnaryOp) -> Result<Step> {
		let Some(StackValue::NodeId(right)) = self.values.pop() else {
			return Err(error(self, "missing right-operand for UNARY-OP"));
		};
		let end = self.nodes.get(right).info().end;
		// TODO - srenshaw - save start location, so we can retrieve it here
		let nx = self.nodes.new_unary(op, right, 0..end)
			.expect("unable to create new UNARY-OP");
		self.values.push(StackValue::NodeId(nx));
		Ok("Parsed UNARY-OP".into())
	}
}

impl ScopeTracker {
	/// Used in type-checking (and potentially elsewhere) to re-add block scopes.
	pub(crate) fn add(&mut self, scope: Scope) {
		self.0.push(scope)
	}

	pub(super) fn insert(&mut self, name: &Rc<str>, vt: ValueType, nx: NodeId) -> NodeId {
		// let index = self.0.len();
		if let Some(scope) = self.0.last_mut() {
			// println!("add {name} : {nx} to scope {}", index-1);
			scope.insert(Rc::clone(name), (vt, nx));
			nx
		} else {
			panic!("add {name} : {vt} = {nx:?} with no base scope");
		}
	}

	pub(super) fn push(&mut self) {
		// println!("pushing a scope | prev-top {:?}", self.0.last());
		self.0.push(Scope::default());
	}

	pub(super) fn pop(&mut self) -> Option<Scope> {
		// let last = self.0.pop();
		// println!("popping a scope | prev-top {last:?}");
		// last
		self.0.pop()
	}

	pub fn find(&self, id: &Rc<str>) -> Option<(ValueType, NodeId)> {
		for scope in self.0.iter().rev() {
			if let Some(nx) = scope.get(id) {
				return Some(nx.clone());
			}
		}
		None
	}

	fn merge(
		nodes: &mut NodeStore,
		mut t_scopes: ScopeTracker,
		mut f_scopes: ScopeTracker,
	) -> Result<ScopeTracker> {
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
							.map(|fnx| nodes.new_phi(tnx.1, fnx.1))
							.unwrap_or(Ok(tnx.1))?;
						scope.insert(Rc::clone(name), (tnx.0.clone(), nx));
					}

					for (name, fnx) in f_scope {
						let nx = t_scope.remove(name)
							.filter(|tnx| tnx != fnx)
							.map(|tnx| nodes.new_phi(tnx.1, fnx.1))
							.unwrap_or(Ok(fnx.1))?;
						scope.insert(Rc::clone(name), (fnx.0.clone(), nx));
					}
				}

				// Just copy over missing nodes
				(Some(o_scope), None) |
				(None, Some(o_scope)) => {
					for (name, nx) in o_scope {
						scope.insert(Rc::clone(name), nx.clone());
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

	let mut s1 = ScopeTracker::default();
	s1.push();
	s1.insert(&"a".into(), ValueType::Any, n0);
	s1.push();
	s1.insert(&"b".into(), ValueType::Any, n1);
	s1.insert(&"c".into(), ValueType::Any, n2);
	s1.push();

	let mut s2 = ScopeTracker::default();
	s2.push();
	s2.push();
	s2.insert(&"a".into(), ValueType::Any, n1);
	s2.insert(&"b".into(), ValueType::Any, n3);
	s2.insert(&"c".into(), ValueType::Any, n2);

	let s3 = ScopeTracker::merge(&mut n, s1, s2)
		.expect("unable to merge s1 and s2");
	assert_eq!(s3.0.len(), 3);

	let (_,n4) = s3.find(&"b".into())
		.expect("missing 'b' node after merge");
	let phi = n.get(n4);
	assert_eq!(phi.expr(), &Expr::Phi);
	assert_eq!(phi.inputs(), &[n1, n3]);

	let mut temp1 = Scope::default();
	temp1.insert("a".into(), (ValueType::Any, n0));
	assert_eq!(s3.0[0], temp1);
	let mut temp2 = Scope::default();
	temp2.insert("a".into(), (ValueType::Any, n1));
	temp2.insert("b".into(), (ValueType::Any, n4));
	temp2.insert("c".into(), (ValueType::Any, n2));
	assert_eq!(s3.0[1], temp2);
}

