
use std::ops::Range;

use tracing::warn;

use crate::parser::{BinaryOp, Block, S, Stmt, UnaryOp};

pub(crate) fn eval(
	ast: Vec<Stmt>,
) -> Vec<Stmt> {
	let mut limit = 50;
	let mut prev = ast;
	loop {
		let out = prev.iter()
			.cloned()
			.map(stmt)
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

fn stmt(
	stmt: Stmt,
) -> Stmt {
	match stmt {
		Stmt::If { cond, bt, bf } => Stmt::If {
			cond,
			bt: block(bt),
			bf: bf.map(block),
		},
		Stmt::While { cond, body } => Stmt::While {
			cond,
			body: block(body),
		},
		Stmt::Var { name, vtype, body } => Stmt::Var {
			name,
			vtype,
			body: expr(body),
		},
		rec @ Stmt::Rec {..} => rec,
		Stmt::Fun { name, params, rtype, body } => Stmt::Fun {
			name,
			params,
			rtype,
			body: block(body),
		},
		Stmt::Assign { referent, body } => Stmt::Assign {
			referent,
			body: expr(body),
		},
	}
}

fn expr(s: S) -> S {
	match s {
		S::Num(n, info) => S::Num(n, info),
		S::Id(s, info) => S::Id(s, info),
		S::Block(b,info) => S::Block(Box::new(block(*b)), info),
		S::If(cond, bt, Some(bf), info) => S::If(
			Box::new(expr(*cond)),
			Box::new(block(*bt)),
			Some(Box::new(block(*bf))),
			info,
		),
		S::If(cond, bt, None, info) => S::If(
			Box::new(expr(*cond)),
			Box::new(block(*bt)),
			None,
			info,
		),
		S::Unary(op, s, info) => unary(op, expr(*s), info),
		S::Binary(op, s0, s1, info) => binary(op, expr(*s0), expr(*s1), info),
		S::FnCall(name, list, info) => S::FnCall(
			name,
			list.into_iter()
				.map(expr)
				.collect(),
			info,
		),
	}
}

fn block(
	Block(stmts, opt_expr): Block,
) -> Block {
	Block(
		stmts.into_iter().map(stmt).collect(),
		opt_expr.map(expr),
	)
}

fn unary(op: UnaryOp, s: S, info: Range<usize>) -> S {
	match op {
		UnaryOp::Neg => if let S::Num(n, info) = s {
			S::Num(-n, info.start-1..info.end)
		} else {
			S::Unary(op, Box::new(s), info)
		}
		UnaryOp::Not => if let S::Num(n, info) = s {
			S::Num(!n, info.start-1..info.end)
		} else {
			S::Unary(op, Box::new(s), info)
		}
		UnaryOp::Pos => s,
		UnaryOp::Deref |
		UnaryOp::Ref => S::Unary(op, Box::new(s), info),
	}
}

fn collapse_binop(
	f: fn(i64,i64) -> i64,
	op: BinaryOp,
	s0: S,
	s1: S,
	info: Range<usize>,
) -> S {
	match (&s0, &s1) {
		(S::Num(n0,i0), S::Num(n1,i1)) => S::Num(f(*n0,*n1), i0.start..i1.end),
		_ => S::Binary(op, Box::new(s0), Box::new(s1), info),
	}
}

fn binary(op: BinaryOp, s0: S, s1: S, info: Range<usize>) -> S {
	match op {
		BinaryOp::Add =>
			collapse_binop(|n0,n1| n0 + n1, op, s0, s1, info),
		BinaryOp::AndB =>
			collapse_binop(|n0,n1| n0 & n1, op, s0, s1, info),
		BinaryOp::AndL =>
			collapse_binop(|n0,n1| (n0 != 0 && n1 != 0) as i64, op, s0, s1, info),
		BinaryOp::CmpEq =>
			collapse_binop(|n0,n1| (n0 == n1) as i64, op, s0, s1, info),
		BinaryOp::CmpGE =>
			collapse_binop(|n0,n1| (n0 >= n1) as i64, op, s0, s1, info),
		BinaryOp::CmpGT =>
			collapse_binop(|n0,n1| (n0 > n1) as i64, op, s0, s1, info),
		BinaryOp::CmpLE =>
			collapse_binop(|n0,n1| (n0 <= n1) as i64, op, s0, s1, info),
		BinaryOp::CmpLT =>
			collapse_binop(|n0,n1| (n0 < n1) as i64, op, s0, s1, info),
		BinaryOp::CmpNE =>
			collapse_binop(|n0,n1| (n0 != n1) as i64, op, s0, s1, info),
		BinaryOp::Div =>
			collapse_binop(|n0,n1| n0 / n1, op, s0, s1, info),
		BinaryOp::LShift =>
			collapse_binop(|n0,n1| n0 << n1, op, s0, s1, info),
		BinaryOp::Mod =>
			collapse_binop(|n0,n1| n0 % n1, op, s0, s1, info),
		BinaryOp::Mul =>
			collapse_binop(|n0,n1| n0 * n1, op, s0, s1, info),
		BinaryOp::OrB =>
			collapse_binop(|n0,n1| n0 | n1, op, s0, s1, info),
		BinaryOp::OrL =>
			collapse_binop(|n0,n1| (n0 != 0 || n1 != 0) as i64, op, s0, s1, info),
		BinaryOp::RShift =>
			collapse_binop(|n0,n1| n0 >> n1, op, s0, s1, info),
		BinaryOp::Sub =>
			collapse_binop(|n0,n1| n0 - n1, op, s0, s1, info),
		BinaryOp::XorB =>
			collapse_binop(|n0,n1| n0 ^ n1, op, s0, s1, info),
		BinaryOp::XorL =>
			collapse_binop(|n0,n1| ((n0 != 0) ^ (n1 != 0)) as i64, op, s0, s1, info),

		BinaryOp::Accessor |
		BinaryOp::Assign |
		BinaryOp::Comma |
		BinaryOp::DivMod |
		BinaryOp::LFShift |
		BinaryOp::RFShift => S::Binary(op, Box::new(s0), Box::new(s1), info),
	}
}

#[cfg(test)]
mod collapses {
	use crate::{lexer, parser, reducer};
	use parser::{S, Stmt};

	#[test]
	fn numeric_literal_expressions() {
		let input = "var a = 3 + 5 + 1 + 2 * 2";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast);
		assert_eq!(ast, vec![
			Stmt::Var {
				name: "a".to_string(),
				vtype: None,
				body: S::Num(13, 0..0),
			}
		]);
	}
}
