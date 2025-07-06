
use std::fmt;

use miette::LabeledSpan;

use super::TokenInfo;

#[derive(Debug)]
pub(crate) enum Error {
	Internal(Context),
	Parse(miette::Report),
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Internal(context) => {
				writeln!(f, "Context:")?;
				for frame in &context.call_stack {
					writeln!(f, "  - {frame}")?;
				}
				if context.message.is_empty() {
					write!(f, "")
				} else {
					write!(f, "{}", context.message)
				}
			}
			Self::Parse(report) => write!(f, "{report:?}"),
		}
	}
}

impl std::error::Error for Error {}

impl From<miette::Report> for Error {
	fn from(report: miette::Report) -> Self {
		Self::Parse(report)
	}
}

impl miette::Diagnostic for Error {}

pub(super) fn report(source: &str, info: TokenInfo, marker: &str, msg: &str) -> Error {
	Error::Parse(miette::miette! {
		labels = [
			LabeledSpan::at(info, marker),
		],
		"{msg}"
	}.with_source_code(source.to_owned()))
}

pub(super) fn error(source: &str, info: TokenInfo, msg: &str) -> Error {
	report(source, info, "here", msg)
}

#[derive(Debug, Default, Clone)]
pub(crate) struct Context {
	pub call_stack: Vec<&'static str>,
	pub debug_log: Vec<String>,
	message: String,
}

impl Context {
	pub(super) fn with_msg(&self, msg: &str) -> Error {
		Error::Internal(Self {
			call_stack: self.call_stack.clone(),
			debug_log: self.debug_log.clone(),
			message: msg.to_string(),
		})
	}
}

macro_rules! with_ctx {
	($parser:expr, $name:expr, $body:block) => {{
		$parser.dbg_ctx.call_stack.push($name);
		$parser.dbg_ctx.debug_log.push(format!("Entering {}", $name));
		let result = $body;
		$parser.dbg_ctx.debug_log.push(format!("Exiting {}", $name));
		$parser.dbg_ctx.call_stack.pop();
		result
	}};
}

pub(super) type Result<T> = std::result::Result<T, Error>;

