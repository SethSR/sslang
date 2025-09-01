
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

fn print_tokens(tokens: &[Token]) {
	eprint!("Tokens:");
	for token in tokens {
		eprint!(" {token}");
	}
	eprintln!();
}

#[derive(Debug, Clone)]
enum Pattern<'a> {
	Const(i64),
	BinOp(BinaryOp, Box<Pattern<'a>>, Box<Pattern<'a>>),
	UnOp(UnaryOp, Box<Pattern<'a>>),
	Block(Vec<Pattern<'a>>, Vec<Rc<str>>),
	Id(&'a str),
	Var(&'a str, VT, Box<Pattern<'a>>),
	If(Box<Pattern<'a>>, Box<Pattern<'a>>, Box<Pattern<'a>>),
	While(Box<Pattern<'a>>, Box<Pattern<'a>>),
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
			(Self::If(cond,tbody,fbody), Expr::If) => {
				match rhs.inputs() {
					[rcond, rtbody, rfbody] => {
						**cond == rhs.store.get(*rcond) &&
							**tbody == rhs.store.get(*rtbody) &&
							**fbody == rhs.store.get(*rfbody)
					}
					[rcond, rtbody] => {
						// TODO - srenshaw - Should also ensure fbody is empty
						**cond == rhs.store.get(*rcond) &&
							**tbody == rhs.store.get(*rtbody)
					}
					_ => false,
				}
			}
			(Self::While(cond,body), Expr::While) => {
				match rhs.inputs() {
					[rcond, rbody] => {
						**cond == rhs.store.get(*rcond) &&
							**body == rhs.store.get(*rbody)
					}
					_ => false,
				}
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
fn if_<'a>(cond: Pattern<'a>, tbody: &'a [Pattern<'a>], fbody: &'a [Pattern<'a>]) -> Pattern<'a> {
	Pattern::If(Box::new(cond), Box::new(block(tbody)), Box::new(block(fbody)))
}
fn while_<'a>(cond: Pattern<'a>, body: &'a [Pattern<'a>]) -> Pattern<'a> {
	Pattern::While(Box::new(cond), Box::new(block(body)))
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
	parse_test("fn a() {}", block(&[
		fun("a", &[], VT::Unit, &[]),
	]))
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
*/

#[test]
fn if_stmt() -> miette::Result<()> {
	parse_test("var a = 3; var b = 1; if a > b {a}", block(&[
		if_(
			binop(BinaryOp::CmpGT,
				id("a"),
				id("b"),
			),
			&[id("a")],
			&[],
		),
	]))
}

#[test]
fn if_else_stmt() -> miette::Result<()> {
	parse_test("var a = 3; var b = 1; if a < b {a} else {b}",
		block(&[
			if_(
				binop(BinaryOp::CmpLT,
					id("a"),
					id("b"),
				),
				&[id("a")],
				&[id("b")],
			),
		]),
	)
}

#[test]
fn while_stmt() -> miette::Result<()> {
	parse_test("while 3 == 2 {1}", block(&[
		while_(
			binop(BinaryOp::CmpEq,
				const_int(3),
				const_int(2),
			),
			&[const_int(1)],
		),
	]))
}

#[test]
fn assign_stmt() -> miette::Result<()> {
	parse_test("var a = 0; a = 3", block(&[
		var("a", VT::Any, const_int(0)),
		binop(BinaryOp::Assign,
			id("a"),
			const_int(3),
		),
	]))
}

