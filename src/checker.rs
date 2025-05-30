
use crate::parser::{
	// Expr,
	Node,
};

pub(crate) fn eval(node: Node) -> Node {
	/*
	match &*node.expr {
		Expr::Num(n) => println!("{n}"),
		Expr::Id(s) => println!("{s}"),
		Expr::Bool(b) => println!("{b}"),
		Expr::Block(b) => println!("[{}]", b.iter()
			.map(|n| n.to_string())
			.collect::<Vec<_>>()
			.join(", ")),
		Expr::While { cond, body } => println!("(while {cond} [{}])", body.iter()
			.map(|n| n.to_string())
			.collect::<Vec<_>>()
			.join(", ")),
		Expr::RecInit { name, field_inits } => println!("(init {name} {field_inits:?})"),
		Expr::Unary { op, rhs } => println!("({op} {rhs})"),
		Expr::Binary { op, lhs, rhs } => println!("({op} {lhs} {rhs})"),
		Expr::Var { name, body } => println!("(var {name} = {body})"),
		Expr::If { cond, bt, bf } => println!("(if {cond} {bt:?} {bf:?})"),
		Expr::FnCall { name, args } => println!("(call {name} {args:?})"),
		Expr::Rec { name, fields } => println!("(rec {name} {fields:?})"),
		Expr::Fun { name, params, rtype, body } => println!("(fn {name} {params:?} -> {rtype} {body:?})"),
	};
	*/
	println!("{node}");
	node
}

