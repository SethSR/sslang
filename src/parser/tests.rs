
use crate::lexer;
use crate::parser::{
	BinaryOp,
	Node,
	node::{NodeId, NodeStore},
	TypedIdent,
	UnaryOp,
	Int,
	ValueType as VT,
};

use super::Parser;

#[derive(Default)]
struct Tester(NodeId, NodeStore);

impl Tester {
	fn finish(mut self) -> Self {
		self.0 = self.block(&[self.0]);
		self
	}

	fn num(&mut self, n: i64, vt: VT) -> NodeId {
		self.1.new_num(n, vt, 0..0)
	}

	fn ident(&mut self, s: &str) -> NodeId {
		self.1.new_id(s.into(), VT::Any, 0..0)
	}

	fn unary(&mut self, op: UnaryOp, rhs: NodeId) -> miette::Result<NodeId> {
		self.1.new_unary(op, rhs, 0..0)
	}

	fn binary(&mut self, op: BinaryOp, lhs: NodeId, rhs: NodeId) -> miette::Result<NodeId> {
		self.1.new_binary(op, lhs, rhs, 0..0)
	}

	fn call(&mut self, name: &str, s: &[NodeId]) -> NodeId {
		self.1.new_call(name.into(), s.to_vec(), 0..0)
	}

	fn var(
		&mut self,
		name: &str,
		vtype: VT,
		body: NodeId,
	) -> NodeId {
		self.1.new_var(name.into(), vtype, body, 0..0)
	}

	fn block(
		&mut self,
		body: &[NodeId],
	) -> NodeId {
		self.1.new_block(body.to_vec(), 0..0)
	}

	fn fun(
		&mut self,
		name: &str,
		params: &[TypedIdent],
		rtype: VT,
		body: &[NodeId],
	) -> NodeId {
		let body = self.block(body);
		self.1.new_fun(name.into(), params.to_vec(), rtype, body, 0..0)
	}

	fn rec(
		&mut self,
		name: &str,
		fields: &[TypedIdent],
	) -> NodeId {
		self.1.new_rec(name.into(), fields.to_vec(), 0..0)
	}

	fn if_s(
		&mut self,
		cond: NodeId,
		bt: &[NodeId],
		bf: &[NodeId],
	) -> NodeId {
		let bt = self.block(bt);
		let bf = self.block(bf);
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

fn expr_test(source: &str, tester: &Tester) -> miette::Result<()> {
	let input = lexer::eval(source)?;
	let mut parser = Parser::new(source, &input);
	let expr = parser.expr(0)?;
	assert_nodes(expr, &parser.nodes, tester.0, &tester.1);
	Ok(())
}

#[test]
fn unary_op_deref() -> miette::Result<()> {
	let mut t = Tester::default();
	let nx = t.ident("a");
	t.0 = t.unary(UnaryOp::Deref, nx)?;
	expr_test("@a", &t)
}

#[test]
fn unary_op_neg() -> miette::Result<()> {
	let mut t = Tester::default();
	let nx = t.ident("a");
	t.0 = t.unary(UnaryOp::Neg, nx)?;
	expr_test("-a", &t)
}

#[test]
fn unary_op_not() -> miette::Result<()> {
	let mut t = Tester::default();
	let nx = t.ident("a");
	t.0 = t.unary(UnaryOp::Not, nx)?;
	expr_test("!a", &t)
}

#[test]
fn unary_op_pos() -> miette::Result<()> {
	let mut t = Tester::default();
	let nx = t.ident("a");
	t.0 = t.unary(UnaryOp::Pos, nx)?;
	expr_test("+a", &t)
}

#[test]
fn unary_op_ref() -> miette::Result<()> {
	let mut t = Tester::default();
	let nx = t.ident("a");
	t.0 = t.unary(UnaryOp::Ref, nx)?;
	expr_test("$a", &t)
}

#[test]
fn precedence() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.num(1, VT::Int(Int::Bot));
	let b = t.num(2, VT::Int(Int::Bot));
	let c = t.num(3, VT::Int(Int::Bot));
	let m = t.binary(BinaryOp::Mul, b, c)?;
	t.0 = t.binary(BinaryOp::Add, a, m)?;
	expr_test("1 + 2 * 3", &t)?;
	let m = t.binary(BinaryOp::Mul, a, b)?;
	t.0 = t.binary(BinaryOp::Add, m, c)?;
	expr_test("1 * 2 + 3", &t)
}

#[test]
fn parentheses() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.num(1, VT::Int(Int::Bot));
	let b = t.num(2, VT::Int(Int::Bot));
	let c = t.num(3, VT::Int(Int::Bot));
	let add = t.binary(BinaryOp::Add, b, c)?;
	t.0 = t.binary(BinaryOp::Mul, a, add)?;
	expr_test("1 * (2 + 3)", &t)
}

fn parse_test(
	input: &str,
	tester: &Tester,
) -> miette::Result<()> {
	eprintln!("input: {input}");
	let tokens = lexer::eval(input)?;
	eprintln!("tokens: {tokens:?}");
	let mut parser = Parser::new(input, &tokens);
	let start = parser.program()?;
	assert_nodes(start, &parser.nodes, tester.0, &tester.1);
	Ok(())
}

fn assert_nodes(nxa: NodeId, sa: &NodeStore, nxb: NodeId, sb: &NodeStore) {
	use crate::parser::Expr;

	match (sa.get(nxa), sb.get(nxb)) {
		(Ok(Node { expr: a, ..}), Ok(Node { expr: b, ..})) => match (a, b) {
			(Expr::Id(ia), Expr::Id(ib)) => assert_eq!(ia, ib),
			(Expr::Num(na), Expr::Num(nb)) => assert_eq!(na, nb),
			(Expr::Block(ba), Expr::Block(bb)) => {
				assert_eq!(ba.len(), bb.len());
				for (a,b) in ba.iter().zip(bb.iter()) {
					assert_nodes(*a, sa, *b, sb);
				}
			}
			(Expr::Rec { name: na, ..}, Expr::Rec { name: nb, ..}) => assert_eq!(na, nb),
			(
				Expr::Fun { name: na, params: pa, rtype: ra, body: ba },
				Expr::Fun { name: nb, params: pb, rtype: rb, body: bb }
			) => {
				assert_eq!(na, nb);
				assert_eq!(pa, pb);
				assert_eq!(ra, rb);
				assert_nodes(*ba, sa, *bb, sb);
			}
			(Expr::Var { name: na, body: ba }, Expr::Var { name: nb, body: bb }) => {
				assert_eq!(na, nb);
				assert_nodes(*ba, sa, *bb, sb);
			}
			(Expr::If { cond: ca, bt: ta, bf: fa }, Expr::If { cond: cb, bt: tb, bf: fb }) => {
				assert_nodes(*ca, sa, *cb, sb);
				assert_nodes(*ta, sa, *tb, sb);
				assert_nodes(*fa, sa, *fb, sb);
			}
			(Expr::While { cond: ca, body: ba }, Expr::While { cond: cb, body: bb }) => {
				assert_nodes(*ca, sa, *cb, sb);
				assert_nodes(*ba, sa, *bb, sb);
			}
			(Expr::Unary { op: oa, rhs: ra }, Expr::Unary { op: ob, rhs: rb }) => {
				assert_eq!(oa, ob);
				assert_nodes(*ra, sa, *rb, sb);
			}
			(Expr::Binary { op: oa, lhs: la, rhs: ra }, Expr::Binary { op: ob, lhs: lb, rhs: rb }) => {
				assert_eq!(oa, ob);
				assert_nodes(*la, sa, *lb, sb);
				assert_nodes(*ra, sa, *rb, sb);
			}
			(Expr::FnCall { name: na, args: aa }, Expr::FnCall { name: nb, args: ab }) => {
				assert_eq!(na, nb);
				assert_eq!(aa.len(), ab.len());
				for (a,b) in aa.iter().zip(ab.iter()) {
					assert_nodes(*a, sa, *b, sb);
				}
			}
			(a,b) => panic!("{a:?} != {b:?}"),
		}
		(a,b) => panic!("{a:?} != {b:?}"),
	}
}

#[test]
fn empty_input() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.block(&[]);
	parse_test("", &t)
}

#[test]
fn var_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	let nx = t.num(0, VT::Int(Int::Bot));
	t.0 = t.var("a", VT::Unit, nx);
	parse_test("var a = 0", &t.finish())
}

#[test]
fn var_stmt_expr() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.num(3, VT::Int(Int::Bot));
	let b = t.num(2, VT::Int(Int::Bot));
	let c = t.num(1, VT::Int(Int::Bot));
	let mul = t.binary(BinaryOp::Mul, a, b)?;
	let add = t.binary(BinaryOp::Add, mul, c)?;
	t.0 = t.var("a", VT::Unit, add);
	parse_test("var a = 3 * 2 + 1", &t.finish())
}

#[test]
fn var_stmt_vtype() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.num(0, VT::Int(Int::Bot));
	t.0 = t.var("a", VT::to_u8(), a);
	parse_test("var a: u8 = 0", &t.finish())
}

#[test]
fn var_stmt_udt_simple() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.ident("b");
	t.0 = t.var("a", VT::Unit, a);
	parse_test("var a = b", &t.finish())
}

#[test]
fn var_stmt_udt_fncall_empty() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.call("b", &[]);
	t.0 = t.var("a", VT::Unit, a);
	parse_test("var a = b()", &t.finish())
}

#[test]
fn var_stmt_udt_fncall_single() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.ident("c");
	let b = t.call("b", &[a]);
	t.0 = t.var("a", VT::Unit, b);
	parse_test("var a = b(c)", &t.finish())
}

#[test]
fn var_stmt_udt_fncall_multi() -> miette::Result<()> {
	let mut t = Tester::default();
	let c = t.ident("c");
	let d = t.ident("d");
	let e = t.ident("e");
	let add = t.binary(BinaryOp::Add, d, e)?;
	let call = t.call("b", &[c, add]);
	t.0 = t.var("a", VT::Unit, call);
	parse_test("var a = b(c, d + e)", &t.finish())
}

#[test]
fn fn_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.fun("a", &[], VT::Unit, &[]);
	parse_test("fn a() {}", &t.finish())
}

#[test]
fn fn_stmt_params() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.fun("a", &[
		("b".into(), VT::to_u8()),
		("c".into(), VT::to_s16()),
		("d".into(), VT::to_f16(6)),
		("e".into(), VT::to_f32(10)),
	], VT::Unit, &[]);
	parse_test("fn a(b:u8, c:s16, d:fw6, e:fd10) {}", &t.finish())
}

#[test]
fn fn_stmt_params_trailing_comma() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.fun("a", &[
		("b".into(), VT::to_u8()),
		("c".into(), VT::to_s16()),
	], VT::Unit, &[]);
	parse_test("fn a(b:u8, c:s16, ) {}", &t.finish())
}

#[test]
fn fn_stmt_rtype_simple() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.fun("a", &[], VT::to_u8(), &[]);
	parse_test("fn a() -> u8 {}", &t.finish())
}

#[test]
fn fn_stmt_rtype_udt() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.fun("a", &[], VT::UDT("b".into()), &[]);
	parse_test("fn a() -> b {}", &t.finish())
}

#[test]
fn fn_stmt_body() -> miette::Result<()> {
	let mut t = Tester::default();
	let n1 = t.num(1, VT::Int(Int::Bot));
	let vb = t.var("b", VT::Unit, n1);
	let n2 = t.num(2, VT::Int(Int::Bot));
	let vc = t.var("c", VT::Unit, n2);
	let b = t.ident("b");
	let c = t.ident("c");
	let add = t.binary(BinaryOp::Add, b, c)?;
	t.0 = t.fun("a", &[], VT::Unit, &[vb, vc, add]);
	parse_test("fn a() {
		var b = 1
		var c = 2
		b + c
	}", &t.finish())
}

#[test]
fn rec_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.rec("a", &[]);
	parse_test("rec a{}", &t.finish())
}

#[test]
fn rec_stmt_fields() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.rec("vec", &[
		("x".into(), VT::to_f32(16)),
		("y".into(), VT::to_f32(16)),
	]);
	parse_test("rec vec{x:fd, y:fd}", &t.finish())
}

#[test]
fn rec_stmt_fields_trailing_comma() -> miette::Result<()> {
	let mut t = Tester::default();
	t.0 = t.rec("vec", &[
		("x".into(), VT::to_f32(16)),
		("y".into(), VT::to_f32(16)),
	]);
	parse_test("rec vec{ x:fd, y:fd, }", &t.finish())
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
	parse_test("var a = b.c.d", &t.finish())
}

#[test]
fn if_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.ident("a");
	let b = t.ident("b");
	let gt = t.binary(BinaryOp::CmpGT, a, b)?;
	t.0 = t.if_s(gt, &[a], &[]);
	parse_test("if a > b {a}", &t.finish())
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
	t.0 = t.while_s(eq, &[b]);
	parse_test("while a == b {b}", &t.finish())
}

#[test]
fn assign_stmt() -> miette::Result<()> {
	let mut t = Tester::default();
	let a = t.ident("a");
	let n3 = t.num(3, VT::Int(Int::Bot));
	t.0 = t.binary(BinaryOp::Assign, a, n3)?;
	parse_test("a = 3", &t.finish())
}

