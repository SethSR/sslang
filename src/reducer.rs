
use crate::parser::{BinaryOp, Expr, Node, UnaryOp};
use crate::parser::{ValueType, TokenInfo};

pub(crate) fn eval(node: Node) -> Node {
	match &*node.expr {
		// Expr::If { cond, bt, bf } => Node::new_if(eval(cond), reduce_list(bt), reduce_list(bf), s.info),
		// Expr::While { cond, body } => Node::new_while(expr(cond), reduce_list(body), s.info),
		// Expr::Var { name, body } => Node::new_var(name, node.kind, expr(body), s.info),
		// Expr::Rec {..} => node,
		/* Expr::Fun { name, params, rtype, body } => Node::new_fun(
			std::rc::Rc::clone(name),
			params.to_vec(),
			rtype.clone(),
			reduce_list(body),
			node.info,
		), */
		// Expr::Num(_) => node,
		// Expr::Id(_) => node,
		// Expr::Block(b) => Node::new_block(reduce_list(b), s.info),
		Expr::Unary { op, rhs } => simplify_unary(*op, rhs).unwrap_or(node),
		Expr::Binary { op, lhs, rhs } => simplify_binary(*op, lhs, rhs,
			node.kind.clone(), node.info.clone()).unwrap_or(node),
		// Expr::FnCall { name, args } => Node::new_call(name, reduce_list(args), s.info),
		_ => node,
	}
}

fn simplify_unary(op: UnaryOp, rhs: &Node) -> Option<Node> {
	match op {
		UnaryOp::Pos => Some(rhs.clone()),
		UnaryOp::Neg => {
			match &*rhs.expr {
				Expr::Num(a) => Some(Node::new_num(-a, rhs.kind.clone(), rhs.info.clone())),
				_ => None,
			}
		}
		UnaryOp::Not => {
			match &*rhs.expr {
				Expr::Num(a) => Some(Node::new_num(!a, rhs.kind.clone(), rhs.info.clone())),
				_ => None,
			}
		}
		_ => None,
	}
}

fn simplify_binary(
	op: BinaryOp,
	lhs: &Node,
	rhs: &Node,
	kind: ValueType,
	info: TokenInfo,
) -> Option<Node> {
	let vt = kind.clone();
	let ti = info.clone();
	let reduce = |lhs: &Node, rhs: &Node, f: fn(i64,i64) -> i64| {
		match (&*lhs.expr, &*rhs.expr) {
			(Expr::Num(a), Expr::Num(b)) => Some(Node::new_num(f(*a,*b), vt, ti)),
			_ => None,
		}
	};

	match op {
		BinaryOp::Accessor => None, // TODO - srenshaw - Add simplify code for Accessor operator.
		BinaryOp::Add => simplify_add(lhs, rhs, kind, info),
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
		BinaryOp::Mul => simplify_mul(lhs, rhs, kind, info),
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

fn simplify_add(
	lhs: &Node,
	rhs: &Node,
	kind: ValueType,
	info: TokenInfo,
) -> Option<Node> {
	match (&*lhs.expr, &*rhs.expr) {
		// Collapse 'unit' values
		(Expr::Id(_), Expr::Num(0)) => Some(lhs.clone()),

		// Collapse matching hs IDs
		(Expr::Id(a), Expr::Id(b)) if a == b => {
			let nx = Node::new_num(1, ValueType::Any, rhs.info.clone());
			Node::new_binary(BinaryOp::LShift, lhs.clone(), nx, info)
				.ok()
		}

		// Collapse constants
		(Expr::Num(a), Expr::Num(b)) => Some(Node::new_num(a+b, kind, info)),

		// Tree-rotate nested ADDs to be simplify friendly
		(Expr::Num(_), Expr::Binary { op: BinaryOp::Add, lhs: r_lhs, rhs: r_rhs }) => {
			match (&*r_lhs.expr, &*r_rhs.expr) {
				(Expr::Id(_), Expr::Num(_)) => {
					let new_rhs = Node::new_binary(BinaryOp::Add, lhs.clone(), r_rhs.clone(), info)
						.ok()?;
					Node::new_binary(BinaryOp::Add, r_lhs.clone(), new_rhs, r_lhs.info.start..r_rhs.info.end)
						.ok()
				}
				_ => None,
			}
		}

		// Tree-Rotate numbers to the right-branch
		(Expr::Num(_), _) => Node::new_binary(BinaryOp::Add, rhs.clone(), lhs.clone(), info).ok(),

		_ => None,
	}
}

fn simplify_mul(lhs: &Node, rhs: &Node, kind: ValueType, info: TokenInfo) -> Option<Node> {
	match (&*lhs.expr, &*rhs.expr) {
		// Collapse 'unit' values
		(Expr::Id(_), Expr::Num(1)) => Some(lhs.clone()),

		// Collapse constants
		(Expr::Num(a), Expr::Num(b)) => Some(Node::new_num(a*b, kind, info)),

		// Tree-rotate nested MULs to be simplify friendly
		(Expr::Num(_), Expr::Binary { op: BinaryOp::Mul, lhs: r_lhs, rhs: r_rhs }) => {
			match (&*r_lhs.expr, &*r_rhs.expr) {
				(Expr::Id(_), Expr::Num(_)) => {
					let new_rhs = Node::new_binary(BinaryOp::Mul, lhs.clone(), r_rhs.clone(), info)
						.ok()?;
					Node::new_binary(BinaryOp::Mul, r_lhs.clone(), new_rhs, r_lhs.info.start..r_rhs.info.end)
						.ok()
				}
				_ => None,
			}
		}

		// Tree-Rotate constants to the right-branch
		(Expr::Num(_), _) => Node::new_binary(BinaryOp::Mul, rhs.clone(), lhs.clone(), info).ok(),

		_ => None,
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
			.unwrap()
	}

	fn unary(op: UnaryOp, a: Node) -> Node {
		Node::new_unary(op, a, 0..0)
			.unwrap()
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
		let ast = reducer::eval(ast);
		assert_eq!(ast, var("a", ValueType::Unit, num(13)));
	}

	#[test]
	fn numeric_literals_separated_by_identifier_with_same_op() {
		let input = "var a = 3 + b + 1";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast);
		assert_eq!(ast,
			var("a", ValueType::Unit, binary(BinaryOp::Add, id("b"), num(4)))
		);
	}

	#[test]
	fn numeric_literals_separated_by_identifier_with_diff_op1() {
		let input = "var a = 3 + b - 1";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast);
		assert_eq!(ast,
			var("a", ValueType::Unit, binary(BinaryOp::Add, id("b"), num(2)))
		);
	}

	#[test]
	fn numeric_literals_separated_by_identifier_with_diff_op2() {
		let input = "var a = 3 - b + 1";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast);
		assert_eq!(ast,
			var("a", ValueType::Unit, binary(BinaryOp::Add, unary(UnaryOp::Neg, id("b")), num(4)))
		);
	}

	#[test]
	fn has_type() {
		let input = "var a: u8 = 2";
		let tokens = lexer::eval(input)
			.expect("valid token list");
		let ast = parser::eval(input, tokens)
			.expect("valid AST");
		let ast = reducer::eval(ast);
		assert_eq!(ast,
			var("a", ValueType::to_u8(), num(2)),
		);
	}
}
