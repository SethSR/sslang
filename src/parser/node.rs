
use std::fmt;
use std::rc::Rc;

use slotmap::{SecondaryMap, SlotMap, new_key_type};

use super::{BinaryOp, Meet, TokenInfo, UnaryOp, ValueType};
use super::parser::Scope;

new_key_type! { pub(crate) struct NodeId; }

pub(crate) struct NodeRef<'a> {
	id: NodeId,
	store: &'a NodeStore,
}

impl NodeRef<'_> {
	pub(crate) fn id(&self) -> u64 {
		use slotmap::Key;
		self.id.data().as_ffi()
	}

	pub(crate) fn expr(&self) -> &Expr {
		&self.store.data[self.id].expr
	}

	pub(crate) fn kind(&self) -> &ValueType {
		&self.store.data[self.id].kind
	}

	pub(crate) fn info(&self) -> &TokenInfo {
		&self.store.data[self.id].info
	}

	pub(crate) fn number(&self) -> i64 {
		self.store.numbers[self.id]
	}

	pub(crate) fn name(&self) -> Rc<str> {
		self.store.names[self.id].clone()
	}

	pub(crate) fn boolean(&self) -> bool {
		self.store.bools[self.id]
	}

	pub(crate) fn inputs(&self) -> &[NodeId] {
		self.store.inputs.get(self.id)
			.map(|s| s.as_slice())
			.unwrap_or(&[])
	}

	pub(crate) fn input(&self, nx: NodeId) -> NodeRef {
		self.store.get(nx)
	}

	pub(crate) fn scope(&self) -> &Scope {
		&self.store.scopes[self.id]
	}

	pub(crate) fn params(&self) -> &[(Rc<str>, ValueType)] {
		self.store.paramlists[self.id].as_slice()
	}

	pub(crate) fn args(&self) -> &[(Rc<str>, NodeId)] {
		self.store.arglists[self.id].as_slice()
	}

	pub(crate) fn vtype(&self) -> &ValueType {
		&self.store.types[self.id]
	}

	pub(crate) fn unary_op(&self) -> UnaryOp {
		self.store.unary_ops[self.id]
	}

	pub(crate) fn binary_op(&self) -> BinaryOp {
		self.store.binary_ops[self.id]
	}
}

/// Expr::Number   -> value: i64
/// Expr::Id       -> name: Rc<str>
/// Expr::Bool     -> value: bool
/// Expr::Block    -> body: Option<NodeId>, scope: Scope
/// Expr::Rec      -> name: Rc<str>, fields: Vec<(Rc<str>, ValueType)>
/// Expr::Fun      -> name: Rc<str>, params: Vec<(Rc<str>, ValueType)>, rtype: ValueType
/// Expr::Var      -> name: Rc<str>, body: Option<NodeId>
/// Expr::If       -> cond: NodeId, bt: NodeId, bf: Option<NodeId>
/// Expr::While    -> cond: NodeId, body: NodeId
/// Expr::RecInit  -> name: Rc<str>, field_inits: Vec<(Rc<str>, NodeId)>
/// Expr::UnaryOp  -> op: UnaryOp, rhs: NodeId
/// Expr::BinaryOp -> op: BinaryOp, lhs: NodeId, rhs: NodeId
/// Expr::FnCall   -> name: Rc<str>, args: Vec<NodeId>
/// Expr::Phi      -> lhs: NodeId, rhs: NodeId
#[derive(Debug, Default)]
pub(crate) struct NodeStore {
	data: SlotMap<NodeId, Node>,

	pub(crate) numbers   : SecondaryMap<NodeId, i64>,
	pub(crate) names     : SecondaryMap<NodeId, Rc<str>>,
	pub(crate) bools     : SecondaryMap<NodeId, bool>,
	pub(crate) inputs    : SecondaryMap<NodeId, Vec<NodeId>>,
	pub(crate) scopes    : SecondaryMap<NodeId, Scope>,
	pub(crate) paramlists: SecondaryMap<NodeId, Vec<(Rc<str>, ValueType)>>,
	pub(crate) arglists  : SecondaryMap<NodeId, Vec<(Rc<str>, NodeId)>>,
	pub(crate) types     : SecondaryMap<NodeId, ValueType>,
	pub(crate) unary_ops : SecondaryMap<NodeId, UnaryOp>,
	pub(crate) binary_ops: SecondaryMap<NodeId, BinaryOp>,
}

impl NodeStore {
	pub fn iter<'a>(&'a self) -> impl Iterator<Item=NodeRef<'a>> {
		self.data.iter().map(|(id,_)| NodeRef { id, store: self })
	}
}

impl NodeStore {
	pub(crate) fn new_block(&mut self, body: Option<NodeId>, scope: Scope, info: TokenInfo) -> NodeId {
		let kind = body.and_then(|bx| self.data.get(bx))
			.map(|n| n.kind.clone())
			.unwrap_or(ValueType::Unit);
		let nx = self.data.insert(Node::new(Expr::Block, kind, info));
		if let Some(bx) = body {
			self.inputs.insert(nx, vec![bx]);
		}
		self.scopes.insert(nx, scope);
		nx
	}

	pub(crate) fn new_bool(&mut self, b: bool, info: TokenInfo) -> NodeId {
		let nx = self.data.insert(Node::new(Expr::Bool, ValueType::Bool, info));
		self.bools.insert(nx, b);
		nx
	}

	pub(crate) fn new_id(
		&mut self,
		id: Rc<str>,
		info: TokenInfo,
	) -> NodeId {
		let nx = self.data.insert(Node::new(Expr::Id, ValueType::Any, info));
		self.names.insert(nx, id);
		nx
	}

	pub(crate) fn new_num(
		&mut self,
		num: i64,
		kind: ValueType,
		info: TokenInfo,
	) -> NodeId {
		let nx = self.data.insert(Node::new(Expr::Num, kind, info));
		self.numbers.insert(nx, num);
		nx
	}

	pub(crate) fn new_rec(
		&mut self,
		name: &Rc<str>,
		fields: Vec<(Rc<str>, ValueType)>,
		info: TokenInfo,
	) -> NodeId {
		let udt = Rc::clone(name);
		let name = Rc::clone(name);
		let nx = self.data.insert(Node::new(Expr::Rec, ValueType::Udt(udt), info));
		self.names.insert(nx, name);
		self.paramlists.insert(nx, fields);
		nx
	}

	pub(crate) fn new_fun(
		&mut self,
		name: &Rc<str>,
		params: Vec<(Rc<str>, ValueType)>,
		rtype: ValueType,
		body: NodeId,
		info: TokenInfo,
	) -> NodeId {
		let kind = rtype.clone();
		let nx = self.data.insert(Node::new(Expr::Fun, kind, info));
		self.names.insert(nx, Rc::clone(name));
		self.paramlists.insert(nx, params);
		self.types.insert(nx, rtype);
		self.inputs.insert(nx, vec![body]);
		nx
	}

	pub(crate) fn new_if(
		&mut self,
		cond: NodeId,
		bt: NodeId,
		bf: Option<NodeId>,
		info: TokenInfo,
	) -> NodeId {
		let tkind = self.get(bt).kind().clone();
		let kind = bf.map(|nx| {
			tkind.meet(self.get(nx).kind())
		}).unwrap_or(ValueType::Unit);

		let nx = self.data.insert(Node::new(Expr::If, kind, info));
		let mut inputs = vec![cond, bt];
		if let Some(bx) = bf {
			inputs.push(bx);
		}
		self.inputs.insert(nx, inputs);
		nx
	}

	pub(crate) fn new_while(
		&mut self,
		cond: NodeId,
		body: NodeId,
		info: TokenInfo,
	) -> NodeId {
		let nx = self.data.insert(Node::new(Expr::While, ValueType::Unit, info));
		self.inputs.insert(nx, vec![cond, body]);
		nx
	}

	pub(crate) fn new_rec_init(
		&mut self,
		name: &Rc<str>,
		field_inits: Vec<(Rc<str>, NodeId)>,
		info: TokenInfo,
	) -> NodeId {
		let nx = self.data.insert(Node::new(Expr::RecInit, ValueType::Udt(Rc::clone(name)), info));
		self.names.insert(nx, Rc::clone(name));
		self.arglists.insert(nx, field_inits);
		nx
	}

	pub(crate) fn new_unary(
		&mut self,
		op: UnaryOp,
		rhs: NodeId,
		info: TokenInfo,
	) -> miette::Result<NodeId> {
		let kind = self.get(rhs).kind().clone();
		let nx = self.data.insert(Node::new(Expr::Unary, kind, info));
		self.unary_ops.insert(nx, op);
		self.inputs.insert(nx, vec![rhs]);
		Ok(nx)
	}

	pub(crate) fn new_binary(
		&mut self,
		op: BinaryOp,
		lhs: NodeId,
		rhs: NodeId,
		info: TokenInfo,
	) -> miette::Result<NodeId> {
		let lhs_kind = self.get(lhs).kind().clone();
		let rhs_kind = self.get(rhs).kind().clone();
		let kind = lhs_kind.meet(&rhs_kind);
		let nx = self.data.insert(Node::new(Expr::Binary, kind, info));
		self.binary_ops.insert(nx, op);
		self.inputs.insert(nx, vec![lhs, rhs]);
		Ok(nx)
	}

	pub(crate) fn new_call(
		&mut self,
		name: &Rc<str>,
		args: Vec<NodeId>,
		info: TokenInfo,
	) -> NodeId {
		let name = Rc::clone(name);
		let nx = self.data.insert(Node::new(Expr::FnCall, ValueType::Any, info));
		self.names.insert(nx, name);
		self.inputs.insert(nx, args);
		nx
	}

	pub(crate) fn new_phi(
		&mut self,
		lhs: NodeId,
		rhs: NodeId,
	) -> miette::Result<NodeId> {
		let lnode = self.get(lhs);
		let rnode = self.get(rhs);
		let kind = lnode.kind().meet(&rnode.kind());
		let start = lnode.info().start.min(rnode.info().start);
		let end = lnode.info().end.max(rnode.info().end);
		let nx = self.data.insert(Node::new(Expr::Phi, kind, start..end));
		self.inputs.insert(nx, vec![lhs, rhs]);
		Ok(nx)
	}

	pub(crate) fn nodes_to_string(&self, nx: NodeId, mut padding: usize, out: &mut Vec<String>) {
		let node = self.data.get(nx);
		let space = "  ".repeat(padding);
		match node {
			Some(node) => {
				out.push(format!("[{nx:3?}] {space}> {node:?}"));
				padding += 1;
				match &node.expr {
					Expr::Block => if let Some(body) = self.inputs.get(nx) {
						self.nodes_to_string(body[0], padding, out);
					}
					Expr::Fun => if let Some(body) = self.inputs.get(nx) {
						self.nodes_to_string(body[0], padding, out);
					}
					Expr::Var => if let Some(body) = self.inputs.get(nx) {
						self.nodes_to_string(body[0], padding, out);
					}
					Expr::If => match &self.inputs[nx][..] {
						[cond, bt, bf] => {
							self.nodes_to_string(*cond, padding, out);
							self.nodes_to_string(*bt, padding, out);
							self.nodes_to_string(*bf, padding, out);
						}
						[cond, bt] => {
							self.nodes_to_string(*cond, padding, out);
							self.nodes_to_string(*bt, padding, out);
						}
						ns => {
							out.push(format!("[{nx:3?}] {space}E > IF {ns:?}"));
						}
					}
					Expr::While => match &self.inputs[nx][..] {
						[cond, body] => {
							self.nodes_to_string(*cond, padding, out);
							self.nodes_to_string(*body, padding, out);
						}
						[cond] => {
							self.nodes_to_string(*cond, padding, out);
						}
						ns => {
							out.push(format!("[{nx:3?}] {space}E > WHILE {ns:?}"));
						}
					}
					Expr::FnCall => for (_,item) in &self.arglists[nx] {
						self.nodes_to_string(*item, padding, out);
					}
					Expr::Unary => for item in &self.inputs[nx] {
						self.nodes_to_string(*item, padding, out);
					}
					Expr::Binary => for item in &self.inputs[nx] {
						self.nodes_to_string(*item, padding, out);
					}
					Expr::RecInit => for (name, vtype) in &self.paramlists[nx] {
						out.push(format!("[{nx:?}] {space}  > ({name}, {vtype})"));
					}
					Expr::Phi => for item in &self.inputs[nx] {
						self.nodes_to_string(*item, padding, out);
					}
					Expr::Num => {}
					Expr::Id => {}
					Expr::Bool => {}
					Expr::Rec => {}
				}
			}
			None => {
				out.push(format!("[{nx:?}] {space}> ERROR: no node found for index"));
			}
		}
	}

	pub(crate) fn get(&self, id: NodeId) -> NodeRef {
		NodeRef { id, store: self }
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

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Expr {
	Num,
	Id,
	Bool,
	Block,
	Rec,
	Fun,
	Var,
	If,
	While,
	RecInit,
	Unary,
	Binary,
	FnCall,
	Phi,
}

impl NodeRef<'_> {
	pub fn is_const(&self) -> bool {
		match self.expr() {
			Expr::Num => true,
			Expr::Bool => true,
			Expr::Phi => {
				let inputs = self.inputs();
				let lhs = self.input(inputs[0]);
				let rhs = self.input(inputs[1]);
				lhs.is_const() && rhs.is_const()
			}
			Expr::Id => false,
			Expr::Block => {
				let inputs = self.inputs();
				self.input(inputs[0]).is_const()
			}
			Expr::Rec => true,
			Expr::Fun => true,
			Expr::Var => false,
			// TODO - srenshaw - We could check whether the conditional and/or the branches are constant
			// and propagate the result here, but it may be better to leave that for an optimization pass
			// somewhere else.
			Expr::If => false,
			Expr::While => false,
			Expr::RecInit => false,
			Expr::Unary => {
				let inputs = self.inputs();
				self.input(inputs[0]).is_const()
			}
			Expr::Binary => {
				let inputs = self.inputs();
				let lhs = self.input(inputs[0]);
				let rhs = self.input(inputs[1]);
				lhs.is_const() && rhs.is_const()
			}
			Expr::FnCall => false,
		}
	}
}

impl PartialEq for NodeRef<'_> {
	fn eq(&self, _rhs: &NodeRef<'_>) -> bool {
		todo!()
	}
}

impl fmt::Debug for NodeRef<'_> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		write!(fmt, "{self}")
	}
}

impl fmt::Display for NodeRef<'_> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		match self.expr() {
			Expr::Block => {
				let scope = self.scope().keys()
					.map(|name| name.to_string())
					.reduce(|out, name| format!("{out},{name}"))
					.unwrap_or_default();
				match self.store.inputs.get(self.id) {
					Some(inputs) => write!(fmt, "{inputs:?} ({scope})"),
					None         => write!(fmt, "[] ({scope})"),
				}
			}
			Expr::Var => {
				let inputs = self.inputs();
				if inputs.is_empty() {
					write!(fmt, "(var {})", self.name())
				} else {
					write!(fmt, "(var {} = {:?})", self.name(), inputs)
				}
			}
			Expr::Id      => write!(fmt, "{}", self.name()),
			Expr::Num     => write!(fmt, "{}", self.number()),
			Expr::Bool    => write!(fmt, "{}", self.boolean()),
			Expr::Unary   => write!(fmt, "({} {:?})", self.unary_op(), self.inputs()),
			Expr::Binary  => write!(fmt, "({} {:?})", self.binary_op(), self.inputs()),
			Expr::If      => write!(fmt, "(if {:?})", self.inputs()),
			Expr::Phi     => write!(fmt, "(phi {:?})", self.inputs()),
			Expr::Rec     => write!(fmt, "(rec {} {:?})", self.name(), self.params()),
			Expr::RecInit => write!(fmt, "(init {} {:?})", self.name(), self.args()),
			Expr::While   => write!(fmt, "(while {:?})", self.inputs()),
			Expr::FnCall  => write!(fmt, "(call {} {:?})", self.name(), self.args()),
			Expr::Fun     => write!(fmt, "(fn {} {:?} -> {} {:?})",
				self.name(), self.params(), self.vtype(), self.inputs()),
		}
	}
}

