
use std::fmt;
use std::rc::Rc;

use super::{
	BinaryOp,
	Meet,
	TokenInfo,
	TypedIdent,
	UnaryOp,
	ValueType,
};

// TODO - srenshaw - Move node-types from Expr into Node.

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

