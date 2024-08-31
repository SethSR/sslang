
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
pub(crate) struct Token {
	pub(crate) tt: TokenType,
	pub(crate) range: Range<usize>
}

impl Token {
	pub(crate) fn new(
		tt: TokenType,
		range: Range<usize>,
	) -> Self {
		Self { tt, range }
	}

	pub(crate) fn get_token<'a>(&self,
		source: &'a str,
	) -> &'a str {
		&source[self.range.clone()]
	}
}

