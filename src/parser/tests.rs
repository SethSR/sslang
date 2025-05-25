
use crate::parser::{
	BinaryOp,
	Expr,
	Node,
	TypedIdent,
	UnaryOp,
	Meet,
	ValueType as VT,
};

fn num(n: i64) -> Node {
	Node {
		kind: VT::Any,
		expr: Expr::Num(n).into(),
		info: 0..0,
	}
}

fn ident(s: &str) -> Node {
	Node {
		kind: VT::Any,
		expr: Expr::Id(s.into()).into(),
		info: 0..0,
	}
}

fn unary(op: UnaryOp, rhs: Node) -> Node {
	Node {
		kind: rhs.kind.clone(),
		expr: Expr::Unary { op, rhs }.into(),
		info: 0..0,
	}
}

fn binary(op: BinaryOp, lhs: Node, rhs: Node) -> Node {
	Node {
		kind: lhs.kind.meet(&rhs.kind),
		expr: Expr::Binary { op, lhs, rhs }.into(),
		info: 0..0,
	}
}

fn fn_call(name: &str, s: &[Node]) -> Node {
	Node {
		kind: VT::Any,
		expr: Expr::FnCall {
			name: name.into(),
			args: s.to_vec(),
		}.into(),
		info: 0..0,
	}
}

fn expr_test(source: &str, s: Node) -> miette::Result<()> {
	use crate::parser::Parser;
	use crate::lexer::eval;

	let input = eval(source)?;
	let mut parser = Parser::new(source, &input);
	assert_eq!(parser.test_expr(0)?, s);
	Ok(())
}

fn var_s(
	name: &str,
	vtype: VT,
	body: Node,
) -> Node {
	Node {
		kind: vtype,
		expr: Expr::Var {
			name: name.into(),
			body,
		}.into(),
		info: 0..0,
	}
}

fn fn_s(
	name: &str,
	params: &[TypedIdent],
	rtype: VT,
	body: &[Node],
) -> Node {
	Node {
		kind: VT::Any,
		expr: Expr::Fun {
			name: name.into(),
			params: params.to_vec(),
			rtype,
			body: body.to_vec()
		}.into(),
		info: 0..0,
	}
}

fn rec_s(
	name: &str,
	fields: &[TypedIdent],
) -> Node {
	Node {
		kind: VT::Unit,
		expr: Expr::Rec {
			name: name.into(),
			fields: fields.to_vec(),
		}.into(),
		info: 0..0,
	}
}

fn if_s(
	cond: Node,
	bt: &[Node],
	bf: &[Node],
) -> Node {
	Node {
		kind: bt.last()
			.zip(bf.last())
			.map(|(t,f)| t.kind.meet(&f.kind))
			.unwrap_or(VT::Unit),
		expr: Expr::If {
			cond,
			bt: bt.to_vec(),
			bf: bf.to_vec(),
		}.into(),
		info: 0..0,
	}
}

fn while_s(
	cond: Node,
	body: &[Node],
) -> Node {
	Node {
		kind: VT::Unit,
		expr: Expr::While {
			cond,
			body: body.to_vec(),
		}.into(),
		info: 0..0,
	}
}

fn assign_s(
	name: &str,
	body: Node,
) -> Node {
	Node {
		kind: body.kind.clone(),
		expr: Expr::Assign {
			name: name.into(),
			body,
		}.into(),
		info: 0..0,
	}
}

#[test]
fn unary_op_deref() -> miette::Result<()> {
	expr_test("@a", unary(UnaryOp::Deref, ident("a")))
}

#[test]
fn unary_op_neg() -> miette::Result<()> {
	expr_test("-3", unary(UnaryOp::Neg, num(3)))
}

#[test]
fn unary_op_not() -> miette::Result<()> {
	expr_test("!3", unary(UnaryOp::Not, num(3)))
}

#[test]
fn unary_op_pos() -> miette::Result<()> {
	expr_test("+3", unary(UnaryOp::Pos, num(3)))
}

#[test]
fn unary_op_ref() -> miette::Result<()> {
	expr_test("$a", unary(UnaryOp::Ref, ident("a")))
}

#[test]
fn precedence() -> miette::Result<()> {
	expr_test("1 + 2 * 3", binary(BinaryOp::Add,
		num(1),
		binary(BinaryOp::Mul, num(2), num(3)),
	))?;
	expr_test("1 * 2 + 3", binary(BinaryOp::Add,
		binary(BinaryOp::Mul, num(1), num(2)),
		num(3),
	))
}

#[test]
fn parentheses() -> miette::Result<()> {
	expr_test("1 * (2 + 3)", binary(BinaryOp::Mul,
		num(1),
		binary(BinaryOp::Add, num(2), num(3)),
	))
}

fn parse_test(
	input: &str,
	stmts: &[Node],
) -> miette::Result<()> {
	use crate::parser;
	use crate::lexer;

	eprintln!("input: {input}");
	let tokens = lexer::eval(input)?;
	eprintln!("tokens: {tokens:?}");
	let ast = parser::eval(input, tokens)?;
	assert_eq!(ast, stmts.to_vec());
	Ok(())
}

#[test]
fn empty_input() -> miette::Result<()> {
	parse_test("", &[])
}

#[test]
fn var_stmt() -> miette::Result<()> {
	parse_test("var a = 0", &[
		var_s("a", VT::Unit, num(0)),
	])
}

#[test]
fn var_stmt_expr() -> miette::Result<()> {
	parse_test("var a = 3 * 2 + 1", &[
		var_s("a", VT::Unit, binary(BinaryOp::Add,
			binary(BinaryOp::Mul, num(3), num(2)),
			num(1),
		)),
	])
}

#[test]
fn var_stmt_vtype() -> miette::Result<()> {
	parse_test("var a: u8 = 0", &[
		var_s("a", VT::to_u8(), num(0)),
	])
}

#[test]
fn var_stmt_udt_simple() -> miette::Result<()> {
	parse_test("var a = b", &[
		var_s("a", VT::Unit, ident("b")),
	])
}

#[test]
fn var_stmt_udt_fncall_empty() -> miette::Result<()> {
	parse_test("var a = b()", &[
		var_s("a", VT::Unit, fn_call("b", &[]))
	])
}

#[test]
fn var_stmt_udt_fncall_single() -> miette::Result<()> {
	parse_test("var a = b(c)", &[
		var_s("a", VT::Unit, fn_call("b", &[ident("c")]))
	])
}

#[test]
fn var_stmt_udt_fncall_multi() -> miette::Result<()> {
	parse_test("var a = b(c, d + e)", &[
		var_s("a", VT::Unit, fn_call("b", &[
			ident("c"),
			binary(BinaryOp::Add, ident("d"), ident("e")),
		]))
	])
}

#[test]
fn fn_stmt() -> miette::Result<()> {
	parse_test("fn a() {}", &[
		fn_s("a", &[], VT::Unit, &[]),
	])
}

#[test]
fn fn_stmt_params() -> miette::Result<()> {
	parse_test("fn a(b:u8 c:s16 d:fw6 e:fd10) {}", &[
		fn_s("a", &[
			("b".into(), VT::to_u8()),
			("c".into(), VT::to_s16()),
			("d".into(), VT::to_f16(6)),
			("e".into(), VT::to_f32(10)),
		], VT::Unit, &[]),
	])
}

#[test]
fn fn_stmt_rtype_simple() -> miette::Result<()> {
	parse_test("fn a() -> u8 {}", &[
		fn_s("a", &[], VT::to_u8(), &[])
	])
}

#[test]
fn fn_stmt_rtype_udt() -> miette::Result<()> {
	parse_test("fn a() -> b {}", &[
		fn_s("a", &[], VT::UDT("b".to_string()), &[])
	])
}

#[test]
fn fn_stmt_body() -> miette::Result<()> {
	parse_test("fn a() {
		var b = 1
		var c = 2
		b + c
	}", &[
		fn_s("a", &[], VT::Unit, &[
			var_s("b", VT::Unit, num(1)),
			var_s("c", VT::Unit, num(2)),
			binary(BinaryOp::Add, ident("b"), ident("c")),
		]),
	])
}

#[test]
fn rec_stmt() -> miette::Result<()> {
	parse_test("rec a{}", &[
		rec_s("a", &[]),
	])
}

#[test]
fn rec_stmt_fields() -> miette::Result<()> {
	parse_test("rec vec{x:fd y:fd}", &[
		rec_s("vec", &[
			("x".into(), VT::to_f32(16)),
			("y".into(), VT::to_f32(16)),
		])
	])
}

#[test]
fn field_access() -> miette::Result<()> {
	parse_test("var a = b.c.d", &[
		var_s("a", VT::Unit, binary(BinaryOp::Accessor,
			ident("b"),
			binary(BinaryOp::Accessor, ident("c"), ident("d")),
		))
	])
}

#[test]
fn if_stmt() -> miette::Result<()> {
	parse_test("if a > b {a}", &[
		if_s(
			binary(BinaryOp::CmpGT, ident("a"), ident("b")),
			&[ident("a")],
			&[],
		)
	])
}

#[test]
fn if_else_stmt() -> miette::Result<()> {
	parse_test("if a < b {a} else {b}", &[
		if_s(
			binary(BinaryOp::CmpLT, ident("a"), ident("b")),
			&[ident("a")],
			&[ident("b")],
		)
	])
}

#[test]
fn while_stmt() -> miette::Result<()> {
	parse_test("while a == b {b}", &[
		while_s(
			binary(BinaryOp::CmpEq, ident("a"), ident("b")),
			&[ident("b")],
		)
	])
}

#[test]
fn assign_stmt() -> miette::Result<()> {
	parse_test("a = 3", &[
		assign_s("a", num(3)),
	])
}

