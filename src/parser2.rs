
use crate::tokens::{Token, TokenType};

#[derive(Debug, Clone)]
pub enum Node {
	VarDecl {
		name: String,
		ty: Option<String>,
		value: Box<Node>,
	},
	RecordDecl {
		name: String,
		fields: Vec<(String, String)>,
	},
	FunctionDecl {
		params: Vec<(String, String)>,
		ret: Option<String>,
		body: Vec<Node>,
	},
	Block(Vec<Node>),
	Int(i64),
	Ident(String),
}

#[derive(Debug)]
enum Op {
	Block,
	Item,
	Var,
	Rec,
	Fun,
	Expr,
}

pub struct Parser {
	tokens: Vec<Token>,
	cursor: usize,
	stack: Vec<Op>,
	value_stack: Vec<Node>,
}

impl Parser {
	pub fn new(tokens: Vec<Token>) -> Self {
		Self {
			tokens,
			cursor: 0,
			stack: vec![Op::Item],
			value_stack: vec![],
		}
	}

	pub fn result(&self) -> Vec<Node> {
		self.value_stack.clone()
	}

	const EOF: Token = Token { tt: TokenType::Eof, start: u16::MAX };
	fn peek(&self) -> &Token {
		self.tokens
			.get(self.cursor)
			.unwrap_or(&Self::EOF)
	}

	fn advance(&mut self) -> Token {
		let tok = self.tokens
			.get(self.cursor)
			.cloned()
			.unwrap_or(Self::EOF);
		self.cursor += 1;
		tok
	}

	pub fn step(&mut self) -> Option<String> {
		use TokenType as TT;

		let op = self.stack.pop()?;
		match op {
			Op::Block => {
				if matches!(self.peek().tt, TT::OBrace) {
					self.advance();
					let items = vec![];
					if !matches!(self.peek().tt, TT::CBrace | TT::Eof) {
						self.stack.push(Op::Item);
						// fake break for REPL-style step: return control
						return Some("Pushed Item inside block".into());
					}
					self.advance();
					self.value_stack.push(Node::Block(items));
					Some("Parsed block".into())
				} else {
					Some("Expected '{' to start block".into())
				}
			}
			Op::Item => match self.peek().tt.clone() {
				TT::Var => {
					self.advance();
					self.stack.push(Op::Var);
					Some("Begin variable parsing".into())
				}
				TT::Rec => {
					self.advance();
					self.stack.push(Op::Rec);
					Some("Begin record parsing".into())
				}
				TT::Fun => {
					self.advance();
					self.stack.push(Op::Fun);
					Some("Begin function parsing".into())
				}
				TT::Integer(n) => {
					self.advance();
					let n = n.parse::<i64>()
						.unwrap_or_else(|_| panic!("unable to parse integer '{n}'"));
					self.value_stack.push(Node::Int(n));
					Some(format!("Parsed int literal {n}"))
				}
				TT::Ident(name) => {
					let name = name.clone();
					self.advance();
					self.value_stack.push(Node::Ident(name.to_string()));
					Some("Parsed identifier".into())
				}
				_ => Some("Unknown construct".into()),
			},
			Op::Var => {
				if let TT::Ident(name) = self.advance().tt {
					self.advance();

					let mut ty = None;
					if matches!(self.peek().tt, TT::Colon) {
						self.advance();
						if let TT::Ident(name) = self.advance().tt {
							ty = Some(name);
						}
					}

					if !matches!(self.advance().tt, TT::Eq1) {
						return Some("Expected '=' in variable declaration".into());
					}

					self.stack.push(Op::Expr);
					self.value_stack.push(Node::Ident(name.to_string()));
					if let Some(ty) = ty {
						self.value_stack.push(Node::Ident(ty.to_string()));
					}

					Some("Parsed variable declaration head".into())
				} else {
					Some("Expected identifier after 'var'".into())
				}
			}
			Op::Rec => {
				if let TT::Ident(name) = self.advance().tt {
					if !matches!(self.advance().tt, TT::OBrace) {
						return Some("Expected '{' in record declaration".into());
					}
					if !matches!(self.peek().tt, TT::CBrace | TT::Eof) {
						self.advance();
						// match Ident
						let TT::Ident(name) = self.advance().tt else {
							return Some("Expected a field identifier in record declaration".into());
						};

						// match ':'
						if !matches!(self.advance().tt, TT::Colon) {
							return Some("Expected ':' after field identifier".into());
						}

						// match Type
						let ty = todo!();
						// match ',' or '}'
					}
					self.advance();
					self.value_stack.push(Node::RecordDecl {
						name: name.to_string(),
						fields: vec![],
					});
					Some("Parsed record declaration".into())
				} else {
					Some("Expected identifier after 'rec'".into())
				}
			}
			Op::Fun => {
				if !matches!(self.advance().tt, TT::OParen) {
					return Some("Expected '(' after 'fun'".into());
				}
				while !matches!(self.peek().tt, TT::CParen | TT::Eof) {
					self.advance();
				}
				self.advance();
				if matches!(self.peek().tt, TT::RetArrow) {
					self.advance();
					self.advance(); // skip return type for now
				}
				self.stack.push(Op::Block);
				Some("Parsed function header, pushing block".into())
			}
			Op::Expr => {
				match self.advance().tt {
					TT::Integer(n) => {
						let n = n.parse::<i64>()
							.unwrap_or_else(|_| panic!("unable to parse int literal '{n}'"));
						self.value_stack.push(Node::Int(n));
						Some(format!("Parsed expr int({n})"))
					}
					TT::Ident(name) => {
						self.value_stack.push(Node::Ident(name.to_string()));
						Some("Parsed expr ident".into())
					}
					_ => Some("Unknown expr".into()),
				}
			}
		}
	}
}

