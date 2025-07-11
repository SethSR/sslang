
use std::collections::HashSet;
use std::ops::Range;
use std::rc::Rc;

#[macro_use]
mod error;
mod node;
mod operators;
#[allow(clippy::module_inception)]
mod parser;
mod types;

pub(crate) use error::Error;
pub(crate) use node::{Expr, Node, NodeId, NodeStore};
pub(crate) use operators::{BinaryOp, UnaryOp};
pub(crate) use parser::{Parser, ScopeTracker, StepResult};
pub(crate) use types::{Meet, ValueType, Int, Fix};

pub(crate) type TokenInfo = Range<usize>;

#[derive(Debug)]
pub(crate) struct Output {
	pub(crate) start: NodeId,
	pub(crate) store: NodeStore,
	pub(crate) records: HashSet<Rc<str>>,
	pub(crate) functions: HashSet<Rc<str>>,
	pub(crate) scopes: ScopeTracker,
}

pub(crate) fn nodes_to_string(nx: NodeId, ns: &NodeStore, mut padding: usize, out: &mut Vec<String>) {
	let node = ns.get(nx);
	match node {
		Ok(node) => {
			out.push(format!("[{nx:3}] {:>1$}{node}", "> ", padding));
			padding += 2;
			match &node.expr {
				Expr::Block{body,..} => {
					if let Some(bx) = body {
						nodes_to_string(*bx, ns, padding, out);
					}
				}
				Expr::Fun { body, ..} => {
					nodes_to_string(*body, ns, padding, out);
				}
				Expr::Var { body, ..} => {
					if let Some(item) = body {
						nodes_to_string(*item, ns, padding, out);
					}
				}
				Expr::If { cond, bt, bf } => {
					nodes_to_string(*cond, ns, padding, out);
					nodes_to_string(*bt, ns, padding, out);
					if let Some(bf) = bf {
						nodes_to_string(*bf, ns, padding, out);
					}
				}
				Expr::While { cond, body } => {
					nodes_to_string(*cond, ns, padding, out);
					nodes_to_string(*body, ns, padding, out);
				}
				Expr::FnCall { args, ..} => {
					for item in args {
						nodes_to_string(*item, ns, padding, out);
					}
				}
				Expr::Unary { rhs, ..} => {
					nodes_to_string(*rhs, ns, padding, out);
				}
				Expr::Binary { lhs, rhs, ..} => {
					nodes_to_string(*lhs, ns, padding, out);
					nodes_to_string(*rhs, ns, padding, out);
				}
				Expr::RecInit { field_inits, ..} => {
					for (_,item) in field_inits {
						nodes_to_string(*item, ns, padding, out);
					}
				}
				Expr::Phi { lhs, rhs } => {
					nodes_to_string(*lhs, ns, padding, out);
					nodes_to_string(*rhs, ns, padding, out);
				}
				Expr::Num(_) => {}
				Expr::Id(_) => {}
				Expr::Bool(_) => {}
				Expr::Rec {..} => {}
			}
		}
		Err(e) => {
			out.push(format!("[{nx}] {:>1$}ERROR: {e}", "> ", padding));
		}
	}
}


#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) type TypedIdent = (Rc<str>, ValueType);

#[cfg(test)]
use crate::tokens::Token;

#[cfg(test)]
pub fn eval(
	source: &str,
	input: Vec<Token>,
) -> miette::Result<Output> {
	if input.is_empty() {
		miette::bail!("Empty input");
	}

	let mut parser = Parser::new(source, &input);
	loop {
		match parser.step() {
			StepResult::Ok(msg) => println!("[step] {msg}"),
			StepResult::Err(e) => eprintln!("[erro] {e}"),
			StepResult::Fatal(e) => {
				eprintln!("[exit] {e}");
				break;
			}
			StepResult::Done => {
				println!("[done] Finished");
				break;
			}
		}
	}

	let out = match parser.finish() {
		Ok(out) => out,
		Err(e) => miette::bail!("{e}"),
	};

	let mut print_out = vec![];
	nodes_to_string(out.start, &out.store, 0, &mut print_out);
	eprintln!("{}", print_out.join("\n"));

	Ok(out)
}

