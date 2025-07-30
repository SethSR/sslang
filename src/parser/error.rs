
use std::fmt;

use miette::LabeledSpan;

use super::TokenInfo;

#[derive(Debug)]
pub(crate) enum Error {
	Report(miette::Report),
	Fatal(miette::Report),
}

impl fmt::Display for Error {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Report(s) => write!(f, "ERROR: {s}"),
			Self::Fatal(s)  => write!(f, "FATAL: {s}"),
		}
	}
}

impl std::error::Error for Error {}

impl From<miette::Report> for Error {
	fn from(report: miette::Report) -> Self {
		Self::Report(report)
	}
}

impl miette::Diagnostic for Error {}

impl Error {
	pub(super) fn report(source: &str, info: TokenInfo, marker: &str, msg: &str) -> Self {
		Self::Report(miette::miette! {
			labels = [
				LabeledSpan::at(info, marker),
			],
			"{msg}"
		}.with_source_code(source.to_owned()))
	}

	pub(super) fn fatal(source: &str, info: TokenInfo, msg: &str) -> Self {
		Self::Fatal(miette::miette! {
			labels = [
				LabeledSpan::at(info, "here"),
			],
			"{msg}"
		}.with_source_code(source.to_owned()))
	}
}

