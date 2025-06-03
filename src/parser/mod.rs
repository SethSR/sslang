
use std::ops::Range;
use std::rc::Rc;

use crate::tokens::Token;

mod node;
mod operators;
mod parser;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use types::{Meet, ValueType, Int, Signed, Unsigned, Fix};
pub(crate) use operators::{BinaryOp, UnaryOp};
pub(crate) use node::{Expr, Node, NodeId, NodeStore};
pub(crate) use parser::{FuncStore, RecStore};

pub(crate) type TokenInfo = Range<usize>;
pub(crate) type TypedIdent = (Rc<str>, ValueType);

use parser::Parser;

#[derive(Debug)]
pub(crate) struct Output {
	pub(crate) start: NodeId,
	pub(crate) store: NodeStore,
	pub(crate) records: RecStore,
	pub(crate) functions: FuncStore,
}

pub fn eval(
	source: &str,
	input: Vec<Token>,
) -> miette::Result<Output> {
	if input.len() == 0 {
		miette::bail!("Empty input");
	}

	let mut parser = Parser::new(source, &input);
	let start = parser.program()?;
	Ok(Output {
		start,
		store: parser.nodes,
		records: parser.records,
		functions: parser.functions,
	})
}

