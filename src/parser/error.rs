
use std::fmt;

use miette::LabeledSpan;

use super::TokenInfo;

#[derive(Debug)]
pub(crate) struct Error(pub(crate) miette::Report);

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{:?}", self.0)
	}
}

impl std::error::Error for Error {}

impl From<miette::Report> for Error {
	fn from(report: miette::Report) -> Self {
		Self(report)
	}
}

impl miette::Diagnostic for Error {}

pub(super) fn report(source: &str, info: TokenInfo, marker: &str, msg: &str) -> Error {
	Error(miette::miette! {
		labels = [
			LabeledSpan::at(info, marker),
		],
		"{msg}"
	}.with_source_code(source.to_owned()))
}

pub(super) fn error(source: &str, info: TokenInfo, msg: &str) -> Error {
	report(source, info, "here", msg)
}

pub(super) type Result<T> = std::result::Result<T, Error>;

