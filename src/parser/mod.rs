
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;

use crate::tokens::Token;

mod node;
mod operators;
mod parser;
mod types;

#[cfg(test)]
mod tests;

pub(crate) use types::{Meet, ValueType};
pub(crate) use operators::{BinaryOp, UnaryOp};
pub(crate) use node::{Expr, Node, NodeId};

pub(crate) type TypedIdent = (Rc<str>, ValueType);

use parser::Parser;
use types::Int;

type TokenInfo = Range<usize>;
pub(crate) type NodeMap = HashMap<NodeId, Node>;

pub fn eval(
	source: &str,
	input: Vec<Token>,
) -> miette::Result<(NodeId, NodeMap)> {
	if input.len() == 0 {
		miette::bail!("Empty input");
	}

	let mut parser = Parser::new(source, &input);
	let start = parser.program()?;
	Ok((start, parser.nodes.output()))
}

