
use std::rc::Rc;

use crate::lexer;

use super::{
	BinaryOp,
	Expr,
	Int,
	NodeId,
	NodeStore,
	Output,
	Token,
	TypedIdent,
	UnaryOp,
	ValueType as VT,
};
use super::parser::{Parser, Scope, ScopeTracker};
use super::node::NodeRef;

fn eval(
	source: &str,
	input: Vec<Token>,
) -> miette::Result<Output> {
	if input.is_empty() {
		miette::bail!("Empty input");
	}

	let mut parser = Parser::new(source, &input);
	while parser.step_and_continue(false) {}

	let out = match parser.finish() {
		Ok(out) => out,
		Err(e) => miette::bail!("{e}"),
	};

	eprintln!("{}", out.store.get(out.start));

	Ok(out)
}

struct Tester(NodeId, NodeStore, ScopeTracker);

impl Default for Tester {
	fn default() -> Self {
		let scopes = ScopeTracker::default();
		Self(NodeId::default(), NodeStore::default(), scopes)
	}
}

impl Tester {
	fn finish(mut self) -> Self {
		self.0 = self.block(&[self.0]);
		self
	}

	fn num(&mut self, n: i64, vt: VT) -> NodeId {
		self.1.new_num(n, vt, 0..0)
	}

	fn ident(&mut self, s: &str) -> NodeId {
		self.1.new_id(s.into(), 0..0)
	}

	fn unary(&mut self, op: UnaryOp, rhs: NodeId) -> miette::Result<NodeId> {
		self.1.new_unary(op, rhs, 0..0)
	}

	fn binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> miette::Result<NodeId> {
		self.1.new_binary(op, lhs, rhs, 0..0)
	}

	fn call(&mut self, name: &str, s: &[NodeId]) -> NodeId {
		self.1.new_call(&name.into(), s.to_vec(), 0..0)
	}

	fn var(
		&mut self,
		name: &str,
		vtype: VT,
		body: NodeId,
	) -> NodeId {
		let name = name.into();
		let nx = self.1.new_var(&name, vtype.clone(), Some(body), 0..0);
		self.2.insert(&name, vtype, nx);
		nx
	}

	fn block(
		&mut self,
		body: &[NodeId],
	) -> NodeId {
		let scope = self.2.pop()
			.unwrap_or_default();
		self.1.new_block(body.last().copied(), scope, 0..0)
	}

	fn fun(
		&mut self,
		name: &str,
		params: &[TypedIdent],
		rtype: VT,
		body: &[NodeId],
	) -> NodeId {
		let name = name.into();
		let params = params.to_vec();
		self.2.push();
		let body = self.block(body);
		let nx = self.1.new_fun(&name, params, rtype.clone(), body, 0..0);
		self.2.insert(&name, rtype, nx);
		nx
	}

	fn rec(
		&mut self,
		name: &str,
		fields: &[TypedIdent],
	) -> NodeId {
		let name = name.into();
		let fields = fields.to_vec();
		let nx = self.1.new_rec(&name, fields, 0..0);
		let type_name = VT::Udt(name.clone());
		self.2.insert(&name, type_name, nx);
		nx
	}

	fn if_s(
		&mut self,
		cond: NodeId,
		bt: &[NodeId],
		bf: &[NodeId],
	) -> NodeId {
		let bt = self.block(bt);
		let bf = if bf.is_empty() {
			None
		} else {
			Some(self.block(bf))
		};
		self.1.new_if(cond, bt, bf, 0..0)
	}

	fn while_s(
		&mut self,
		cond: NodeId,
		body: &[NodeId],
	) -> NodeId {
		let body = self.block(body);
		self.1.new_while(cond, body, 0..0)
	}
}

fn print_tokens(tokens: &[Token]) {
	eprint!("Tokens:");
	for token in tokens {
		eprint!(" {token}");
	}
	eprintln!();
}

fn print_nodes<'a>(node_iter: impl Iterator<Item=NodeRef<'a>>) {
	eprint!("Nodes:");
	for node in node_iter {
		eprint!(" {node}");
	}
	eprintln!();
}

fn expr_test(source: &str, tester: &Tester) -> miette::Result<()> {
	use crate::parser::parser::StackValue;

	eprintln!("Source: {source}");

	let tokens = lexer::eval(source)?;
	print_tokens(&tokens);

	let mut parser = Parser::new(source, &tokens);
	print_nodes(parser.nodes.iter());

	parser.scopes.add(Scope::default());
	parser.expr_start(0)?;
	let Some(StackValue::NodeId(expr)) = parser.values.pop() else {
		panic!("expected a node-id at top of value stack")
	};
	// assert_nodes(expr, &parser.nodes, tester.0, &tester.1, 0);
	assert_eq!(parser.nodes.get(expr), tester.1.get(tester.0));
	Ok(())
}

#[derive(Debug, Clone)]
enum Pattern<'a> {
	Const(i64),
	BinOp(BinaryOp, Box<Pattern<'a>>, Box<Pattern<'a>>),
	UnOp(UnaryOp, Box<Pattern<'a>>),
	Block(Vec<Pattern<'a>>, Vec<Rc<str>>),
	Id(&'a str),
	Var(&'a str, VT, Box<Pattern<'a>>),
}

impl PartialEq<NodeRef<'_>> for Pattern<'_> {
	fn eq(&self, rhs: &NodeRef<'_>) -> bool {
		match (self, rhs.expr()) {
			(Self::Const(a), Expr::Num) => *a == rhs.number(),
			(Self::BinOp(op, lnx, rnx), Expr::Binary) => {
				*op == rhs.binary_op() &&
					**lnx == rhs.store.get(rhs.inputs()[0]) &&
					**rnx == rhs.store.get(rhs.inputs()[1])
			}
			(Self::UnOp(op, rnx), Expr::Unary) => {
				*op == rhs.unary_op() &&
					**rnx == rhs.store.get(rhs.inputs()[0])
			}
			(Self::Block(patterns,scope), Expr::Block) => {
				let rscope = rhs.scope();
				patterns == &rhs.inputs().iter().map(|nx| rhs.store.get(*nx)).collect::<Vec<_>>() &&
					scope.iter().all(|s| rscope.contains_key(s))
			}
			(Self::Id(id), Expr::Id) => {
				**id == *rhs.name()
			}
			(Self::Var(_,vtype,rnx), _) => {
				vtype == rhs.kind() &&
					**rnx == *rhs
			}
			_ => {
				eprintln!("END: {self:?}");
				eprintln!("END: {rhs}");
				panic!()
			}
		}
	}
}

impl PartialEq<Pattern<'_>> for NodeRef<'_> {
	fn eq(&self, rhs: &Pattern<'_>) -> bool {
		rhs == self
	}
}

fn const_int(n: i64) -> Pattern<'static> { Pattern::Const(n) }
fn binop<'a>(op: BinaryOp, lhs: Pattern<'a>, rhs: Pattern<'a>) -> Pattern<'a> {
	Pattern::BinOp(op, Box::new(lhs), Box::new(rhs))
}
fn unop<'a>(op: UnaryOp, rhs: Pattern<'a>) -> Pattern<'a> {
	Pattern::UnOp(op, Box::new(rhs))
}
fn block<'a>(items: &'a [Pattern<'a>]) -> Pattern<'a> {
	let mut scope = vec![];
	let mut patterns = vec![];
	for item in items {
		if let Pattern::Var(s,_,_) = item {
			scope.push((*s).into());
		} else {
			patterns.push(item.clone());
		}
	}
	Pattern::Block(patterns, scope)
}
fn id<'a>(id: &'a str) -> Pattern<'a> {
	Pattern::Id(id)
}
fn var<'a>(id: &'a str, vtype: VT, expr: Pattern<'a>) -> Pattern<'a> {
	Pattern::Var(id, vtype, Box::new(expr))
}

fn parse_test(
	input: &str,
	pattern: Pattern,
) -> miette::Result<()> {
	eprintln!("input: {input}");

	let tokens = lexer::eval(input)?;
	print_tokens(&tokens);

	let out = eval(input, tokens)?;
	assert_eq!(out.store.get(out.start), pattern);

	Ok(())
}

#[test]
fn empty_input() -> miette::Result<()> {
	parse_test("", block(&[]))
}

#[test]
fn constant() -> miette::Result<()> {
	parse_test("7", block(&[const_int(7)]))
}

#[test]
fn unary_op_deref() -> miette::Result<()> {
	parse_test("@a", block(&[
		unop(UnaryOp::Deref, id("a")),
	]))
}

#[test]
fn unary_op_neg() -> miette::Result<()> {
	parse_test("-a", block(&[
		unop(UnaryOp::Neg, id("a")),
	]))
}

#[test]
fn unary_op_not() -> miette::Result<()> {
	parse_test("!a", block(&[
		unop(UnaryOp::Not, id("a")),
	]))
}

#[test]
fn unary_op_pos() -> miette::Result<()> {
	parse_test("+a", block(&[
		unop(UnaryOp::Pos, id("a")),
	]))
}

#[test]
fn unary_op_ref() -> miette::Result<()> {
	parse_test("$a", block(&[
		unop(UnaryOp::Ref, id("a")),
	]))
}

#[test]
fn precedence() -> miette::Result<()> {
	parse_test("1 + 2 * 3", block(&[
		binop(BinaryOp::Add,
			const_int(1),
			binop(BinaryOp::Mul,
				const_int(2),
				const_int(3),
			),
		),
	]))
}

#[test]
fn precedence2() -> miette::Result<()> {
	parse_test("1 * 2 + 3", block(&[
		binop(BinaryOp::Add,
			binop(BinaryOp::Mul,
				const_int(1),
				const_int(2),
			),
			const_int(3),
		),
	]))
}

#[test]
fn parentheses() -> miette::Result<()> {
	parse_test("1 * (2 + 3)", block(&[
		binop(BinaryOp::Mul,
			const_int(1),
			binop(BinaryOp::Add,
				const_int(2),
				const_int(3),
			),
		),
	]))
}

#[test]
fn var_stmt() -> miette::Result<()> {
	parse_test("var a = 0", block(&[
		var("a", VT::Any, const_int(0)),
	]))
}

#[test]
fn var_stmt_expr() -> miette::Result<()> {
	parse_test("var a = 3 * 2 + 1", block(&[
		var("a", VT::Any, binop(BinaryOp::Add,
			binop(BinaryOp::Mul,
				const_int(3),
				const_int(2),
			),
			const_int(1),
		)),
	]))
}

#[test]
fn var_stmt_vtype() -> miette::Result<()> {
	parse_test("var a: u8 = 0", block(&[
		var("a", VT::Int(Int::Bot), const_int(0)),
	]))
}

#[test]
fn var_stmt_udt_simple() -> miette::Result<()> {
	parse_test("var b = 0; var a = b;", block(&[
		var("b", VT::Any, const_int(0)),
		var("a", VT::Any, const_int(0)),
	]))
}

/*
#[test]
fn var_stmt_udt_fncall_empty() -> miette::Result<()> {
	let mut t = Tester::default();
	let _ = t.fun("b", &[], VT::Unit, &[]);
	let a = t.call("b", &[]);
	t.var("a", VT::Unit, a);
	t.0 = t.block(&[]);
	parse_test("fn b() {} var a = b();", &t)
}

#[test]
fn var_stmt_udt_fncall_single() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.ident("c");
	let b = t.call("b", &[a]);
	t.var("a", VT::Unit, b);
	t.0 = t.block(&[]);
	parse_test("var a = b(c)", &t)
}

#[test]
fn var_stmt_udt_fncall_multi() -> miette::Result<()> {
	let mut t = Tester::default();
	let c = t.ident("c");
	let d = t.ident("d");
	let e = t.ident("e");
	let add = t.binary(BinaryOp::Add, d, e)?;
	let call = t.call("b", &[c, add]);
	t.var("a", VT::Unit, call);
	t.0 = t.block(&[]);
	parse_test("var a = b(c, d + e)", &t)
}

#[test]
fn fn_def() -> miette::Result<()> {
	let mut t = Tester::default();
	t.fun("a", &[], VT::Unit, &[]);
	t.0 = t.block(&[]);
	parse_test("fn a() {}", &t)
}

#[test]
fn fn_def_params() -> miette::Result<()> {
	let mut t = Tester::default();
	t.fun("a", &[
		("b".into(), VT::to_u8()),
		("c".into(), VT::to_s16()),
		("d".into(), VT::to_f16(6)),
		("e".into(), VT::to_f32(10)),
	], VT::Unit, &[]);
	t.0 = t.block(&[]);
	parse_test("fn a(b:u8, c:s16, d:fw6, e:fd10) {}", &t)
}

#[test]
fn fn_def_params_trailing_comma() -> miette::Result<()> {
	let mut t = Tester::default();
	t.fun("a", &[
		("b".into(), VT::to_u8()),
		("c".into(), VT::to_s16()),
	], VT::Unit, &[]);
	t.0 = t.block(&[]);
	parse_test("fn a(b:u8, c:s16, ) {}", &t)
}

#[test]
fn fn_def_rtype_simple() -> miette::Result<()> {
	let mut t = Tester::default();
	t.fun("a", &[], VT::to_u8(), &[]);
	t.0 = t.block(&[]);
	parse_test("fn a() -> u8 {}", &t)
}

#[test]
fn fn_def_rtype_udt() -> miette::Result<()> {
	let mut t = Tester::default();
	t.fun("a", &[], VT::Udt("b".into()), &[]);
	t.0 = t.block(&[]);
	parse_test("fn a() -> b {}", &t)
}

#[test]
fn fn_def_body() -> miette::Result<()> {
	let mut t = Tester::default();
	let b = t.num(1, VT::Int(Int::Bot));
	let c = t.num(2, VT::Int(Int::Bot));
	let add = t.binary(BinaryOp::Add, b, c)?;
	t.fun("a", &[], VT::Unit, &[b, c, add]);
	t.0 = t.block(&[]);
	parse_test("fn a() {
		var b = 1;
		var c = 2;
		b + c
	}", &t)
}

#[test]
fn rec_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.rec("a", &[]);
	t.0 = t.block(&[]);
	parse_test("rec a{}", &t)
}

#[test]
fn rec_stmt_fields() -> miette::Result<()> {
	let mut t = Tester::default();
	t.rec("vec", &[
		("x".into(), VT::to_f32(16)),
		("y".into(), VT::to_f32(16)),
	]);
	t.0 = t.block(&[]);
	parse_test("rec vec{x:fd, y:fd}", &t)
}

#[test]
fn rec_stmt_fields_trailing_comma() -> miette::Result<()> {
	let mut t = Tester::default();
	t.rec("vec", &[
		("x".into(), VT::to_f32(16)),
		("y".into(), VT::to_f32(16)),
	]);
	t.0 = t.block(&[]);
	parse_test("rec vec{ x:fd, y:fd, }", &t)
}

#[test]
fn field_access() -> miette::Result<()> {
	let mut t = Tester::default();
	let b = t.ident("b");
	let c = t.ident("c");
	let d = t.ident("d");
	let acc = t.binary(BinaryOp::Accessor, c, d)?;
	let acc = t.binary(BinaryOp::Accessor, b, acc)?;
	t.0 = t.var("a", VT::Unit, acc);
	parse_test("var b = 0; var a = b.c.d", &t.finish())
}

#[test]
fn if_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.ident("a");
	let b = t.ident("b");
	let gt = t.binary(BinaryOp::CmpGT, a, b)?;
	t.0 = t.if_s(gt, &[a], &[]);
	parse_test("if a > b {a}", &t)
}

#[test]
fn if_else_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.ident("a");
	let b = t.ident("b");
	let lt = t.binary(BinaryOp::CmpLT, a, b)?;
	t.0 = t.if_s(lt, &[a], &[b]);
	parse_test("if a < b {a} else {b}", &t.finish())
}

#[test]
fn while_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.ident("a");
	let b = t.ident("b");
	let eq = t.binary(BinaryOp::CmpEq, a, b)?;
	t.while_s(eq, &[b]);
	t.0 = t.block(&[]);
	parse_test("while 3 == 2 {1}", &t)
}

#[test]
fn assign_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.ident("a");
	let n3 = t.num(3, VT::Int(Int::Bot));
	t.binary(BinaryOp::Assign, a, n3)?;
	t.0 = t.block(&[]);
	parse_test("var a = 0; a = 3", &t)
}
*/

