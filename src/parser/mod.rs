
use std::collections::HashSet;
use std::ops::Range;
use std::rc::Rc;

use crate::tokens::Token;

mod node;
mod operators;
#[allow(clippy::module_inception)]
mod parser;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use types::{Meet, ValueType, Int, Fix};
pub(crate) use operators::{BinaryOp, UnaryOp};
pub(crate) use node::{Expr, Node, NodeId, NodeStore};
pub(crate) use parser::ScopeTracker;

pub(crate) use parser::stepper;

pub(crate) type TokenInfo = Range<usize>;
#[cfg(test)]
pub(crate) type TypedIdent = (Rc<str>, ValueType);

#[derive(Debug)]
pub(crate) struct Output {
	pub(crate) start: NodeId,
	pub(crate) store: NodeStore,
	pub(crate) records: HashSet<Rc<str>>,
	pub(crate) functions: HashSet<Rc<str>>,
	pub(crate) scopes: ScopeTracker,
}

fn print_nodes(s: usize, nx: NodeId, ns: &NodeStore) {
	let node = ns.get(nx);
	match node {
		Ok(node) => {
			println!("[{nx:3}] {:>1$}{node}", "> ", s);
			match &node.expr {
				Expr::Block{body,..} => {
					for item in body {
						print_nodes(s + 2, *item, ns);
					}
				}
				Expr::Fun { body, ..} => {
					print_nodes(s + 2, *body, ns);
				}
				Expr::Var { body, ..} => {
					if let Some(item) = body {
						print_nodes(s + 2, *item, ns);
					}
				}
				Expr::If { cond, bt, bf } => {
					print_nodes(s + 2, *cond, ns);
					print_nodes(s + 2, *bt, ns);
					if let Some(bf) = bf {
						print_nodes(s + 2, *bf, ns);
					}
				}
				Expr::While { cond, body } => {
					print_nodes(s + 2, *cond, ns);
					print_nodes(s + 2, *body, ns);
				}
				Expr::FnCall { args, ..} => {
					for item in args {
						print_nodes(s + 2, *item, ns);
					}
				}
				Expr::Unary { rhs, ..} => {
					print_nodes(s + 2, *rhs, ns);
				}
				Expr::Binary { lhs, rhs, ..} => {
					print_nodes(s + 2, *lhs, ns);
					print_nodes(s + 2, *rhs, ns);
				}
				Expr::RecInit { field_inits, ..} => {
					for (_,item) in field_inits {
						print_nodes(s + 2, *item, ns);
					}
				}
				Expr::Phi { lhs, rhs } => {
					print_nodes(s + 2, *lhs, ns);
					print_nodes(s + 2, *rhs, ns);
				}
				Expr::Num(_) => {}
				Expr::Id(_) => {}
				Expr::Bool(_) => {}
				Expr::Rec {..} => {}
			}
		}
		Err(e) => {
			println!("[{nx}] {:>1$}{e}", "> ", s);
		}
	}
}

pub fn eval(
	source: &str,
	input: Vec<Token>,
) -> miette::Result<Output> {
	if input.is_empty() {
		miette::bail!("Empty input");
	}

	let mut stepper = stepper(source, &input);
	while let Some(result) = stepper.step() {
		match result {
			Ok(msg) => println!("[step] {msg}"),
			Err(e) => eprintln!("[erro] {e}"),
		}
	}

	let Some(out) = stepper.finish() else {
		miette::bail!("Unable to retrieve output from Parser");
	};

	print_nodes(2, out.start, &out.store);
	Ok(out)
}

