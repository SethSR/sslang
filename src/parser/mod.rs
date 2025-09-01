
use std::ops::Range;

#[macro_use]
mod error;
mod node;
mod operators;
#[allow(clippy::module_inception)]
mod parser;
mod types;

pub(crate) use node::{Expr, NodeId, NodeRef, NodeStore};
pub(crate) use operators::{BinaryOp, UnaryOp};
pub(crate) use parser::{Parser, ScopeTracker};
pub(crate) use types::{Meet, ValueType, Int, Fix};

pub(crate) type TokenInfo = Range<usize>;

#[derive(Debug)]
pub(crate) struct Output {
	pub(crate) start: NodeId,
	pub(crate) store: NodeStore,
	// pub(crate) records: HashSet<Rc<str>>,
	// pub(crate) functions: HashSet<Rc<str>>,
	pub(crate) scopes: ScopeTracker,
}


#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) type TypedIdent = (Rc<str>, ValueType);

#[cfg(test)]
use crate::tokens::Token;

