
use std::ops::Range;

use tracing::warn;

use crate::parser::{BinaryOp, Expr, Node, UnaryOp};
use crate::parser::{Meet, ValueType};

pub(crate) fn eval(
	ast: Vec<Node>,
	mut limit: u32,
) -> Vec<Node> {
	let mut prev = ast;
	loop {
		let out = prev.iter()
			.cloned()
			.map(expr)
			.collect();
		if out == prev {
			break out;
		}
		prev = out;
		limit -= 1;
		if limit <= 0 {
			warn!("reducer limit exceeded");
			break prev;
		}
	}
}

fn expr(s: Node) -> Node {
	match *s.expr {
		Expr::If { cond, bt, bf } => Node::new_if(expr(cond), reduce_list(bt), reduce_list(bf), s.info),
		Expr::While { cond, body } => Node::new_while(expr(cond), reduce_list(body), s.info),
		Expr::Var { name, body } => Node::new_var(name, s.kind, expr(body), s.info),
		Expr::Rec {..} => s,
		Expr::Fun { name, params, rtype, body } => Node::new_fun(
			name,
			params,
			rtype,
			reduce_list(body),
			s.info,
		),
		Expr::Assign { name, body } => Node::new_assign(name, expr(body), s.info),
		Expr::Num(_) => s,
		Expr::Id(_) => s,
		Expr::Block(b) => Node::new_block(reduce_list(b), s.info),
		Expr::Unary { op, rhs } => unary(op, expr(rhs), s.info),
		Expr::Binary { op, lhs, rhs } => binary(op, expr(lhs), expr(rhs), s.info),
		Expr::FnCall { name, args } => Node::new_call(name, reduce_list(args), s.info),
	}
}

fn reduce_list(nodes: Vec<Node>) -> Vec<Node> {
	nodes.into_iter().map(expr).collect()
}

fn unary(op: UnaryOp, s: Node, info: Range<usize>) -> Node {
	match op {
		UnaryOp::Neg => if let Expr::Num(n) = *s.expr {
			Node::new(Expr::Num(-n), s.kind, info)
		} else {
			Node::new_unary(op, s, info)
		}
		UnaryOp::Not => if let Expr::Num(n) = *s.expr {
			Node::new(Expr::Num(!n), s.kind, info)
		} else {
			Node::new_unary(op, s, info)
		}
		UnaryOp::Pos => s,
		UnaryOp::Deref |
		UnaryOp::Ref => Node::new_unary(op, s, info),
	}
}

fn collapse_binop(
	f: fn(i64,i64) -> i64,
	op: BinaryOp,
	lhs: Node,
	rhs: Node,
	info: Range<usize>,
) -> Node {
	match (*lhs.expr, *rhs.expr) {
		(Expr::Num(n0), Expr::Num(n1)) => Node::new(Expr::Num(f(n0,n1)), ValueType::Unit, info),
		(lex, rex) => Node::new_binary(
			op,
			Node::new(lex, lhs.kind, lhs.info),
			Node::new(rex, rhs.kind, rhs.info),
			info,
		),
	}
}

fn binary(op: BinaryOp, lhs: Node, rhs: Node, info: Range<usize>) -> Node {
	let kind = lhs.kind.meet(&rhs.kind);
	match op {
		BinaryOp::Add => match ((*lhs.expr).clone(), (*rhs.expr).clone()) {
			(Expr::Num(n0), Expr::Num(n1)) => Node::new(Expr::Num(n0 + n1), kind, info),
			(s_0, Expr::Num(n1)) => if let Expr::Binary { op: BinaryOp::Add, lhs: lhs0, rhs: lhs1 } = s_0 {
				// ((lhs0 + lhs1) + rhs)
				match (*lhs0.expr, *lhs1.expr) {
					(Expr::Num(n00), s_01) => Node::new_binary(
						BinaryOp::Add,
						Node::new(s_01, lhs1.kind, lhs1.info),
						Node::new(
							Expr::Num(n00 + n1),
							lhs0.kind.meet(&rhs.kind),
							lhs0.info.start..rhs.info.end,
						),
						info,
					),
					(s_00, Expr::Num(n01)) => Node::new_binary(
						BinaryOp::Add,
						Node::new(s_00, lhs0.kind, lhs0.info),
						Node::new(
							Expr::Num(n01 + n1),
							lhs1.kind.meet(&rhs.kind),
							lhs1.info.start..rhs.info.end,
						),
						info,
					),
					_ => Node::new_binary(BinaryOp::Add, lhs, rhs, info),
				}
			} else {
				Node::new_binary(op, lhs, rhs, info)
			}
			_ => Node::new_binary(op, lhs, rhs, info),
		}
		BinaryOp::AndB =>
			collapse_binop(|n0,n1| n0 & n1, op, lhs, rhs, info),
		BinaryOp::AndL =>
			collapse_binop(|n0,n1| (n0 != 0 && n1 != 0) as i64, op, lhs, rhs, info),
		BinaryOp::CmpEq =>
			collapse_binop(|n0,n1| (n0 == n1) as i64, op, lhs, rhs, info),
		BinaryOp::CmpGE =>
			collapse_binop(|n0,n1| (n0 >= n1) as i64, op, lhs, rhs, info),
		BinaryOp::CmpGT =>
			collapse_binop(|n0,n1| (n0 > n1) as i64, op, lhs, rhs, info),
		BinaryOp::CmpLE =>
			collapse_binop(|n0,n1| (n0 <= n1) as i64, op, lhs, rhs, info),
		BinaryOp::CmpLT =>
			collapse_binop(|n0,n1| (n0 < n1) as i64, op, lhs, rhs, info),
		BinaryOp::CmpNE =>
			collapse_binop(|n0,n1| (n0 != n1) as i64, op, lhs, rhs, info),
		BinaryOp::Div =>
			collapse_binop(|n0,n1| n0 / n1, op, lhs, rhs, info),
		BinaryOp::LShift =>
			collapse_binop(|n0,n1| n0 << n1, op, lhs, rhs, info),
		BinaryOp::Mod =>
			collapse_binop(|n0,n1| n0 % n1, op, lhs, rhs, info),
		BinaryOp::Mul =>
			collapse_binop(|n0,n1| n0 * n1, op, lhs, rhs, info),
		BinaryOp::OrB =>
			collapse_binop(|n0,n1| n0 | n1, op, lhs, rhs, info),
		BinaryOp::OrL =>
			collapse_binop(|n0,n1| (n0 != 0 || n1 != 0) as i64, op, lhs, rhs, info),
		BinaryOp::RShift =>
			collapse_binop(|n0,n1| n0 >> n1, op, lhs, rhs, info),
		BinaryOp::Sub => match ((*lhs.expr).clone(), (*rhs.expr).clone()) {
			(Expr::Num(n0), Expr::Num(n1)) => Node::new(
				Expr::Num(n0 - n1),
				lhs.kind.meet(&rhs.kind),
				info,
			),
			(_, Expr::Num(n1)) => Node::new_binary(
				BinaryOp::Add,
				lhs,
				Node::new(Expr::Num(-n1), rhs.kind, rhs.info),
				info,
			),
			_ => {
				let rinfo = rhs.info.clone();
				Node::new_binary(
					BinaryOp::Add,
					lhs,
					Node::new_unary(UnaryOp::Neg, rhs, rinfo),
					info,
				)
			}
		}
		BinaryOp::XorB =>
			collapse_binop(|n0,n1| n0 ^ n1, op, lhs, rhs, info),
		BinaryOp::XorL =>
			collapse_binop(|n0,n1| ((n0 != 0) ^ (n1 != 0)) as i64, op, lhs, rhs, info),

		BinaryOp::Accessor |
		BinaryOp::Assign |
		BinaryOp::Comma |
		BinaryOp::DivMod |
		BinaryOp::LRot |
		BinaryOp::RRot => Node::new_binary(op, lhs, rhs, info),
	}
}

#[cfg(test)]
mod collapses {
	use crate::{lexer, parser, reducer};
	use parser::{BinaryOp, Expr, Node, UnaryOp, ValueType};

	fn id(s: &str) -> Node {
		Node::new(Expr::Id(s.into()), ValueType::Any, 0..0)
	}

	fn num(n: i64) -> Node {
		Node::new(Expr::Num(n), ValueType::Any, 0..0)
	}

	fn binary(op: BinaryOp, a: Node, b: Node) -> Node {
		Node::new_binary(op, a, b, 0..0)
	}

	fn unary(op: UnaryOp, a: Node) -> Node {
		Node::new_unary(op, a, 0..0)
	}

	fn var(name: &str, vtype: ValueType, body: Node) -> Node {
		Node::new(Expr::Var { name: name.into(), body }, vtype, 0..0)
	}

	#[test]
	fn numeric_literal_expressions() {
		let input = "var a = 3 + 5 + 1 + 2 * 2";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast, 10);
		assert_eq!(ast, vec![
			var("a", ValueType::Unit, num(13))
		]);
	}

	#[test]
	fn numeric_literals_separated_by_identifier_with_same_op() {
		let input = "var a = 3 + b + 1";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast, 10);
		assert_eq!(ast, vec![
			var("a", ValueType::Unit, binary(BinaryOp::Add, id("b"), num(4)))
		]);
	}

	#[test]
	fn numeric_literals_separated_by_identifier_with_diff_op1() {
		let input = "var a = 3 + b - 1";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast, 10);
		assert_eq!(ast, vec![
			var("a", ValueType::Unit, binary(BinaryOp::Add, id("b"), num(2)))
		]);
	}

	#[test]
	fn numeric_literals_separated_by_identifier_with_diff_op2() {
		let input = "var a = 3 - b + 1";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast, 10);
		assert_eq!(ast, vec![
			var("a", ValueType::Unit, binary(BinaryOp::Add, unary(UnaryOp::Neg, id("b")), num(4)))
		]);
	}

	#[test]
	fn has_type() {
		let input = "var a: u8 = 2";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast, 10);
		assert_eq!(ast, vec![
			var("a", ValueType::to_u8(), num(2)),
		]);
	}
}
