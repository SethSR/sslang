
use std::fmt;
use std::rc::Rc;

use super::{BinaryOp, Meet, TokenInfo, UnaryOp, ValueType};

pub(crate) type NodeId = usize;

#[derive(Debug, Default)]
pub(crate) struct NodeStore {
	data: Vec<Option<Node>>,
	free: Vec<NodeId>,
}

impl NodeStore {
	pub fn iter(&self) -> impl Iterator<Item=(usize,&Node)> {
		self.data.iter()
			.enumerate()
			.filter_map(|(i,n)| n.as_ref().zip(Some(i)))
			.map(|(n,i)| (i,n))
	}
}

impl NodeStore {
	pub(crate) fn new_block(&mut self, body: Vec<NodeId>, info: TokenInfo) -> NodeId {
		let kind = body.last()
			.and_then(|nx| self.data.get(*nx))
			.and_then(|n| n.as_ref())
			.map(|n| n.kind.clone())
			.unwrap_or(ValueType::Unit);
		self.add(Node::new(Expr::Block { body }, kind, info))
	}

	pub(crate) fn new_bool(&mut self, b: bool, info: TokenInfo) -> NodeId {
		self.add(Node::new(Expr::Bool(b), ValueType::Bool, info))
	}

	pub(crate) fn new_id(
		&mut self,
		s: Rc<str>,
		kind: ValueType,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::Id(s), kind, info))
	}

	pub(crate) fn new_num(
		&mut self,
		n: i64,
		kind: ValueType,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::Num(n), kind, info))
	}

	pub(crate) fn new_rec(
		&mut self,
		name: Rc<str>,
		fields: Vec<NodeId>,
		info: TokenInfo,
	) -> NodeId {
		let udt = Rc::clone(&name);
		self.add(Node::new(Expr::Rec { name, fields }, ValueType::Udt(udt), info))
	}

	pub(crate) fn new_fun(
		&mut self,
		name: Rc<str>,
		params: Vec<NodeId>,
		rtype: ValueType,
		body: NodeId,
		info: TokenInfo,
	) -> NodeId {
		let kind = rtype.clone();
		self.add(Node::new(Expr::Fun { name, params, rtype, body }, kind, info))
	}

	pub(crate) fn new_var(
		&mut self,
		name: Rc<str>,
		vtype: ValueType,
		body: NodeId,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::Var { name, body }, vtype, info))
	}

	pub(crate) fn new_if(
		&mut self,
		cond: NodeId,
		bt: NodeId,
		bf: Option<NodeId>,
		info: TokenInfo,
	) -> NodeId {
		let bt_kind = self.data.get(bt)
			.and_then(|n| n.as_ref())
			.map(|n| &n.kind);
		let bf_kind = bf.and_then(|b| self.data.get(b))
			.and_then(|n| n.as_ref())
			.map(|n| &n.kind);
		let kind = match (bt_kind, bf_kind) {
			(Some(nt), Some(nf)) => nt.meet(nf),
			_ => ValueType::Unit,
		};
		self.add(Node::new(Expr::If { cond, bt, bf }, kind, info))
	}

	pub(crate) fn new_while(
		&mut self,
		cond: NodeId,
		body: NodeId,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::While { cond, body }, ValueType::Unit, info))
	}

	pub(crate) fn new_rec_init(
		&mut self,
		name: Rc<str>,
		field_inits: Vec<(Rc<str>, NodeId)>,
		info: TokenInfo,
	) -> NodeId {
		let udt = Rc::clone(&name);
		self.add(Node::new(Expr::RecInit { name, field_inits }, ValueType::Udt(udt), info))
	}

	pub(crate) fn new_unary(
		&mut self,
		op: UnaryOp,
		rhs: NodeId,
		info: TokenInfo,
	) -> miette::Result<NodeId> {
		let kind = self.get(rhs)?.kind.clone();
		Ok(self.add(Node::new(Expr::Unary { op, rhs }, kind, info)))
	}

	pub(crate) fn new_binary(
		&mut self,
		op: BinaryOp,
		lhs: NodeId,
		rhs: NodeId,
		info: TokenInfo,
	) -> miette::Result<NodeId> {
		let lhs_kind = &self.get(lhs)?.kind;
		let rhs_kind = &self.get(rhs)?.kind;
		let kind = lhs_kind.meet(rhs_kind);
		Ok(self.add(Node::new(Expr::Binary { op, lhs, rhs }, kind, info)))
	}

	pub(crate) fn new_call(
		&mut self,
		name: Rc<str>,
		args: Vec<NodeId>,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::FnCall { name, args }, ValueType::Any, info))
	}

	pub(crate) fn new_phi(
		&mut self,
		lhs: NodeId,
		rhs: NodeId,
	) -> miette::Result<NodeId> {
		let lnode = &self.get(lhs)?;
		let rnode = &self.get(rhs)?;
		let kind = lnode.kind.meet(&rnode.kind);
		let start = lnode.info.start.min(rnode.info.start);
		let end = lnode.info.end.max(rnode.info.end);
		Ok(self.add(Node::new(Expr::Phi { lhs, rhs }, kind, start..end)))
	}

	pub(crate) fn get(&self, nx: NodeId) -> miette::Result<&Node> {
		self.data.get(nx)
			.and_then(|n| n.as_ref())
			.ok_or_else(|| miette::miette!("Compiler Error: expression information not found in parser"))
	}

	pub(crate) fn get_mut(&mut self, nx: NodeId) -> miette::Result<&mut Node> {
		self.data.get_mut(nx)
			.and_then(|n| n.as_mut())
			.ok_or_else(|| miette::miette!("Compiler Error: expression information not found in parser"))
	}
}

impl NodeStore {
	pub(crate) fn add(&mut self, node: Node) -> NodeId {
		if let Some(idx) = self.free.pop() {
			self.data[idx] = Some(node);
			idx
		} else {
			self.data.push(Some(node));
			self.data.len() - 1
		}
	}
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
	pub(crate) info: TokenInfo,
	pub(crate) kind: ValueType,
	pub(crate) expr: Expr,
}

impl PartialEq for Node {
	fn eq(&self, rhs: &Self) -> bool {
		self.kind == rhs.kind && self.expr == rhs.expr
	}
}

impl Node {
	pub(crate) fn new(expr: Expr, kind: ValueType, info: TokenInfo) -> Self {
		Self { info, kind, expr }
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
	Bool(bool),
	Block {
		body: Vec<NodeId>,
	},
	Rec {
		name: Rc<str>,
		fields: Vec<NodeId>,
	},
	Fun {
		name: Rc<str>,
		params: Vec<NodeId>,
		rtype: ValueType,
		body: NodeId,
	},
	Var {
		name: Rc<str>,
		body: NodeId,
	},
	If {
		cond: NodeId,
		bt: NodeId,
		bf: Option<NodeId>,
	},
	While {
		cond: NodeId,
		body: NodeId,
	},
	RecInit {
		name: Rc<str>,
		field_inits: Vec<(Rc<str>, NodeId)>,
	},
	Unary {
		op: UnaryOp,
		rhs: NodeId,
	},
	Binary {
		op: BinaryOp,
		lhs: NodeId,
		rhs: NodeId,
	},
	FnCall {
		name: Rc<str>,
		args: Vec<NodeId>,
	},
	Phi {
		lhs: NodeId,
		rhs: NodeId,
	},
}

impl fmt::Display for Expr {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		fn join<T>(a: &[T], b: fn(&T) -> String) -> String {
			a.iter()
				.map(b)
				.collect::<Vec<_>>()
				.join(", ")
		}

		match self {
			Expr::Num(n)                  => write!(fmt, "{n}"),
			Expr::Id(s)                   => write!(fmt, "{s}"),
			Expr::Bool(b)                 => write!(fmt, "{b}"),
			Expr::Unary { op, rhs }       => write!(fmt, "({op} {rhs})"),
			Expr::Var { name, body }      => write!(fmt, "(var {name} = {body})"),
			Expr::Binary { op, lhs, rhs } => write!(fmt, "({op} {lhs} {rhs})"),
			Expr::Phi { lhs, rhs }        => write!(fmt, "(phi {lhs} {rhs})"),

			Expr::Block { body } => write!(fmt, "[{}]",
				join(body, |n| n.to_string()),
			),
			Expr::While { cond, body } => write!(fmt, "(while {cond} {body})"),
			Expr::RecInit { name, field_inits } => write!(fmt, "(init {name} [{}])",
				join(field_inits, |(a,b)| format!("{a}: {b}")),
			),
			Expr::FnCall { name, args } => write!(fmt, "(call {name} [{}])",
				join(args, |n| n.to_string()),
			),
			Expr::Rec { name, fields } => write!(fmt, "(rec {name} [{}])",
				join(fields, |n| n.to_string()),
			),
			Expr::If { cond, bt, bf: Some(bf) } => write!(fmt, "(if {cond} {bt} {bf})"),
			Expr::If { cond, bt, bf: None } => write!(fmt, "(if {cond} {bt} ())"),
			Expr::Fun { name, params, rtype, body } => write!(fmt, "(fn {name} [{}] -> {rtype} {body})",
				join(params, |n| n.to_string()),
			),
		}
	}
}

