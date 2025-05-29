
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

use super::{BinaryOp, Meet, TokenInfo, TypedIdent, UnaryOp, ValueType};

pub(crate) type NodeId = usize;

#[derive(Debug, Default)]
pub(crate) struct NodeStore {
	data: Vec<Option<Node>>,
	free: Vec<NodeId>,
}

impl NodeStore {
	pub(super) fn output(self) -> HashMap<NodeId, Node> {
		self.data.into_iter()
			.enumerate()
			.flat_map(|(nx,node)| node.map(|n| (nx,n)))
			.collect()
	}

	pub(super) fn simplify(&mut self, nx: NodeId) -> NodeId {
		match self.get(nx) {
			Ok(Node { expr, kind, info }) => match expr {
				Expr::Unary { op, rhs } => self.simplify_unary(*op, *rhs).unwrap_or(nx),
				Expr::Binary { op, lhs, rhs } => self.simplify_binary(*op, *lhs, *rhs, kind.clone(), info.clone()).unwrap_or(nx),
				_ => nx,
			}
			Err(_) => nx,
		}
	}
}

impl NodeStore {
	pub(super) fn new_block(&mut self, b: Vec<NodeId>, info: TokenInfo) -> NodeId {
		self.add(Node::new(Expr::Block(b), ValueType::Unit, info))
	}

	pub(super) fn new_id(
		&mut self,
		s: Rc<str>,
		kind: ValueType,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::Id(s.into()), kind, info))
	}

	pub(super) fn new_num(
		&mut self,
		n: i64,
		kind: ValueType,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::Num(n), kind, info))
	}

	pub(super) fn new_rec(
		&mut self,
		name: Rc<str>,
		fields: Vec<TypedIdent>,
		info: TokenInfo,
	) -> NodeId {
		let udt = Rc::clone(&name);
		self.add(Node::new(Expr::Rec { name, fields }, ValueType::UDT(udt), info))
	}

	pub(super) fn new_fun(
		&mut self,
		name: Rc<str>,
		params: Vec<TypedIdent>,
		rtype: ValueType,
		body: Vec<NodeId>,
		info: TokenInfo,
	) -> NodeId {
		let kind = rtype.clone();
		self.add(Node::new(Expr::Fun { name, params, rtype, body }, kind, info))
	}

	pub(super) fn new_var(
		&mut self,
		name: Rc<str>,
		vtype: ValueType,
		body: NodeId,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::Var { name, body }, vtype, info))
	}

	pub(super) fn new_if(
		&mut self,
		cond: NodeId,
		bt: Vec<NodeId>,
		bf: Vec<NodeId>,
		info: TokenInfo,
	) -> NodeId {
		let last_true_node = bt.last()
			.and_then(|nx| self.data.get(*nx))
			.and_then(|n| n.as_ref());
		let last_false_node = bf.last()
			.and_then(|nx| self.data.get(*nx))
			.and_then(|n| n.as_ref());
		let kind = match (last_true_node, last_false_node) {
			(Some(true_node), Some(false_node)) => true_node.kind.meet(&false_node.kind),
			_ => ValueType::Unit,
		};
		self.add(Node::new(Expr::If { cond, bt, bf }, kind, info))
	}

	pub(super) fn new_while(
		&mut self,
		cond: NodeId,
		body: Vec<NodeId>,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::While { cond, body }, ValueType::Unit, info))
	}

	pub(super) fn new_rec_init(
		&mut self,
		name: Rc<str>,
		field_inits: Vec<(Rc<str>, NodeId)>,
		info: TokenInfo,
	) -> NodeId {
		let udt = Rc::clone(&name);
		self.add(Node::new(Expr::RecInit { name, field_inits }, ValueType::UDT(udt), info))
	}

	pub(super) fn new_unary(
		&mut self,
		op: UnaryOp,
		rhs: NodeId,
		info: TokenInfo,
	) -> miette::Result<NodeId> {
		let kind = self.get(rhs)?.kind.clone();
		Ok(self.add(Node::new(Expr::Unary { op, rhs }, kind, info)))
	}

	pub(super) fn new_binary(
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

	pub(super) fn new_call(
		&mut self,
		name: Rc<str>,
		args: Vec<NodeId>,
		info: TokenInfo,
	) -> NodeId {
		self.add(Node::new(Expr::FnCall { name, args }, ValueType::Any, info))
	}

	pub(crate) fn get(&self, nx: NodeId) -> miette::Result<&Node> {
		self.data.get(nx)
			.and_then(|n| n.as_ref())
			.ok_or_else(|| miette::miette!("Compiler Error: expression information not found in parser"))
	}
}

impl NodeStore {
	fn add(&mut self, node: Node) -> NodeId {
		if let Some(idx) = self.free.pop() {
			self.data[idx] = Some(node);
			idx
		} else {
			self.data.push(Some(node));
			self.data.len() - 1
		}
	}

	fn simplify_unary(&mut self, op: UnaryOp, rhs: NodeId) -> Option<NodeId> {
		match op {
			UnaryOp::Pos => Some(rhs),
			UnaryOp::Neg => {
				let node = self.get(rhs).ok()?;
				match node.expr {
					Expr::Num(a) => Some(self.new_num(-a, node.kind.clone(), node.info.clone())),
					_ => None,
				}
			}
			UnaryOp::Not => {
				let node = self.get(rhs).ok()?;
				match node.expr {
					Expr::Num(a) => Some(self.new_num(!a, node.kind.clone(), node.info.clone())),
					_ => None,
				}
			}
			_ => None,
		}
	}

	fn simplify_binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId, kind: ValueType, info: TokenInfo) -> Option<NodeId> {
		let vt = kind.clone();
		let ti = info.clone();
		let reduce = |lhs, rhs, f: fn(i64,i64) -> i64| {
			let lexpr = &self.get(lhs).ok()?.expr;
			let rexpr = &self.get(rhs).ok()?.expr;
			match (lexpr, rexpr) {
				(Expr::Num(a), Expr::Num(b)) => Some(self.new_num(f(*a,*b), vt, ti)),
				_ => None,
			}
		};

		match op {
			BinaryOp::Accessor => None, // TODO - srenshaw - Add simplify code for Accessor operator.
			BinaryOp::Add => self.simplify_add(lhs, rhs, kind, info),
			BinaryOp::AndB => reduce(lhs, rhs, |a,b| a & b),
			BinaryOp::AndL => reduce(lhs, rhs, |a,b| ((a != 0) && (b != 0)) as i64),
			// TODO - srenshaw - Assigns should get converted into scope
			// insertions before we reach this point.
			BinaryOp::Assign => None,
			BinaryOp::CmpEq => reduce(lhs, rhs, |a,b| (a == b) as i64),
			BinaryOp::CmpNE => reduce(lhs, rhs, |a,b| (a != b) as i64),
			BinaryOp::CmpGE => reduce(lhs, rhs, |a,b| (a >= b) as i64),
			BinaryOp::CmpGT => reduce(lhs, rhs, |a,b| (a >  b) as i64),
			BinaryOp::CmpLE => reduce(lhs, rhs, |a,b| (a <= b) as i64),
			BinaryOp::CmpLT => reduce(lhs, rhs, |a,b| (a <  b) as i64),
			// TODO - srenshaw - Check whether commas are actually used as binary-operators, and whether
			// they can be handled before this point.
			BinaryOp::Comma => None,
			// TODO - srenshaw - Need to deal with div-by-zero

			// TODO - srenshaw - Remember to read the Hitachi manual for info on how to do "automatic"
			// division processing. (SH7604 Hardware Manual, pg 289)
			BinaryOp::Div => reduce(lhs, rhs, |a,b| a / b),
			// TODO - srenshaw - DivMod will need special-case handling as it "returns" 2 values.
			BinaryOp::DivMod => None,
			// TODO - srenshaw - Should probably do constant checking on left-rotates.
			BinaryOp::LRot => reduce(lhs, rhs, |a,b| a.rotate_left(b as u32)),
			// TODO - srenshaw - Should probably do constant checking on left-shifts.
			BinaryOp::LShift => reduce(lhs, rhs, |a,b| a << b),
			// TODO - srenshaw - Need to deal with div-by-zero
			BinaryOp::Mod => reduce(lhs, rhs, |a,b| a % b),
			BinaryOp::Mul => self.simplify_mul(lhs, rhs, kind, info),
			BinaryOp::OrB => reduce(lhs, rhs, |a,b| a | b),
			BinaryOp::OrL => reduce(lhs, rhs, |a,b| ((a != 0) || (b != 0)) as i64),
			// TODO - srenshaw - Should probably do constant checking on right-rotates.
			BinaryOp::RRot => reduce(lhs, rhs, |a,b| a.rotate_right(b as u32)),
			// TODO - srenshaw - Should probably do constant checking on right-shifts.
			BinaryOp::RShift => reduce(lhs, rhs, |a,b| a >> b),
			BinaryOp::Sub => reduce(lhs, rhs, |a,b| a - b),
			BinaryOp::XorB => reduce(lhs, rhs, |a,b| a ^ b),
			BinaryOp::XorL => reduce(lhs, rhs, |a,b| ((a != 0) ^ (b != 0)) as i64),
		}
	}

	fn simplify_add(&mut self, lhs: NodeId, rhs: NodeId, kind: ValueType, info: TokenInfo) -> Option<NodeId> {
		let lnode = self.get(lhs).ok()?.clone();
		let rnode = self.get(rhs).ok()?.clone();
		match (&lnode.expr, &rnode.expr) {
			// Collapse 'unit' values
			(Expr::Id(_), Expr::Num(0)) => Some(lhs),

			// Collapse matching node IDs
			(Expr::Id(a), Expr::Id(b)) if a == b => {
				let nx = self.new_num(1, ValueType::Any, rnode.info);
				self.new_binary(BinaryOp::LShift, lhs, nx, info)
					.ok()
			}

			// Collapse constants
			(Expr::Num(a), Expr::Num(b)) => Some(self.new_num(a+b, kind, info)),

			// Tree-rotate nested ADDs to be simplify friendly
			(Expr::Num(_), Expr::Binary { op: BinaryOp::Add, lhs: r_lhs, rhs: r_rhs }) => {
				let r_lnode = self.get(*r_lhs).ok()?.clone();
				let r_rnode = self.get(*r_rhs).ok()?.clone();
				match (&r_lnode.expr, &r_rnode.expr) {
					(Expr::Id(_), Expr::Num(_)) => {
						let new_rhs = self.new_binary(BinaryOp::Add, lhs, *r_rhs, info)
							.ok()?;
						self.new_binary(BinaryOp::Add, *r_lhs, new_rhs, r_lnode.info.start..r_rnode.info.end)
							.ok()
					}
					_ => None,
				}
			}

			// Tree-Rotate numbers to the right-branch
			(Expr::Num(_), _) => self.new_binary(BinaryOp::Add, rhs, lhs, info).ok(),

			_ => None,
		}
	}

	fn simplify_mul(&mut self, lhs: NodeId, rhs: NodeId, kind: ValueType, info: TokenInfo) -> Option<NodeId> {
		let lnode = self.get(lhs).ok()?.clone();
		let rnode = self.get(rhs).ok()?.clone();
		match (&lnode.expr, &rnode.expr) {
			// Collapse 'unit' values
			(Expr::Id(_), Expr::Num(1)) => Some(lhs),

			// Collapse constants
			(Expr::Num(a), Expr::Num(b)) => Some(self.new_num(a*b, kind, info)),

			// Tree-rotate nested MULs to be simplify friendly
			(Expr::Num(_), Expr::Binary { op: BinaryOp::Mul, lhs: r_lhs, rhs: r_rhs }) => {
				let r_lnode = self.get(*r_lhs).ok()?.clone();
				let r_rnode = self.get(*r_rhs).ok()?.clone();
				match (&r_lnode.expr, &r_rnode.expr) {
					(Expr::Id(_), Expr::Num(_)) => {
						let new_rhs = self.new_binary(BinaryOp::Mul, lhs, *r_rhs, info)
							.ok()?;
						self.new_binary(BinaryOp::Mul, *r_lhs, new_rhs, r_lnode.info.start..r_rnode.info.end)
							.ok()
					}
					_ => None,
				}
			}

			// Tree-Rotate constants to the right-branch
			(Expr::Num(_), _) => self.new_binary(BinaryOp::Mul, rhs, lhs, info).ok(),

			_ => None,
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
		// TODO - srenshaw - At some point, we'll probably want to add 'kind' to this check.
		self.expr == rhs.expr
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
	Block(Vec<NodeId>),
	Rec {
		name: Rc<str>,
		fields: Vec<TypedIdent>,
	},
	Fun {
		name: Rc<str>,
		params: Vec<TypedIdent>,
		rtype: ValueType,
		body: Vec<NodeId>,
	},
	Var {
		name: Rc<str>,
		body: NodeId,
	},
	If {
		cond: NodeId,
		bt: Vec<NodeId>,
		bf: Vec<NodeId>,
	},
	While {
		cond: NodeId,
		body: Vec<NodeId>,
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
}

impl fmt::Display for Expr {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Expr::While { cond, body }              => write!(fmt, "(while {cond} {body:?})"),
			Expr::Num(n)                            => write!(fmt, "{n}"),
			Expr::Id(s)                             => write!(fmt, "{s}"),
			Expr::Block(b)                          => write!(fmt, "{b:?}"),
			Expr::RecInit { name, field_inits }     => write!(fmt, "(init {name} {field_inits:?})"),
			Expr::Unary { op, rhs }                 => write!(fmt, "({op} {rhs})"),
			Expr::Binary { op, lhs, rhs }           => write!(fmt, "({op} {lhs} {rhs})"),
			Expr::Var { name, body }                => write!(fmt, "(var {name} = {body})"),
			Expr::If { cond, bt, bf }               => write!(fmt, "(if {cond} {bt:?} {bf:?})"),
			Expr::FnCall { name, args }             => write!(fmt, "(call {name} {args:?})"),
			Expr::Rec { name, fields }              => write!(fmt, "(rec {name} {fields:?})"),
			Expr::Fun { name, params, rtype, body } => write!(fmt, "(fn {name} {params:?} -> {rtype} {body:?})"),
		}
	}
}

