
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenType {
	Let,
	Fn,
	Rec,
	U8,
	U16,
	U32,
	S8,
	S16,
	S32,
	F16,
	F32,
	OParen,
	CParen,
	LArrow,
	RArrow,
	Plus,
	Minus,
	Star,
	Slash,
	Dot,
	Percent,
	Ampersand,
	Pipe,
	Carrot,
	Bang,
	Eq,
	Ident,
	Num,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Token<'a> {
	pub(crate) tt: TokenType,
	pub(crate) src: &'a str,
	pub(crate) range: Range<usize>
}

impl<'a> Token<'a> {
	pub(crate) fn new(
		tt: TokenType,
		src: &'a str,
		range: Range<usize>,
	) -> Self {
		Self { tt, src, range }
	}

	pub(crate) fn get_token(&self) -> &str {
		&self.src[self.range.clone()]
	}
}

