
use std::ops::Range;
use std::cmp::PartialEq;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenType {
	EOF,

	// Keywords
	If,    // 'if'
	Fun,   // 'fun'
	Rec,   // 'rec'
	Var,   // 'var'
	Else,  // 'else'
	While, // 'while'

	// Value Types
	U8,  // u8
	U16, // u16
	U32, // u32
	S8,  // s8
	S16, // s16
	S32, // s32
	F16, // fw[0-9]*
	F32, // fl[0-9]*

	// Literals
	Ident,  // [a-zA-Z_][a-zA-Z0-9_]*
	Number, // [0-9_]+\.?[0-9_]*

	// Operators
	Amp1,     // &
	Amp2,     // &&
	At,       // @
	Bang,     // !
	BangEq,   // !=
	Bar1,     // |
	Bar2,     // ||
	Carrot1,  // ^
	Carrot2,  // ^^
	CParen,   // )
	Colon,    // :
	Dollar,   // $
	Dot,      // .
	Eq1,      // =
	Eq2,      // ==
	LArrow1,  // <
	LArrow2,  // <<
	LArrBar,  // <|
	LArrEq,   // <=
	Minus,    // -
	OParen,   // (
	Percent,  // %
	Plus,     // +
	RArrow1,  // >
	RArrow2,  // >>
	RArrBar,  // |>
	RArrEq,   // >=
	Slash,    // /
	SlashPer, // /%
	Star,     // *
}

#[derive(Clone, Eq)]
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
}

impl PartialEq for Token<'_> {
	fn eq(&self, rhs: &Self) -> bool {
		match (self.tt, rhs.tt) {
			(TokenType::Ident, TokenType::Ident) |
			(TokenType::Number, TokenType::Number) =>
				self.to_string() == rhs.to_string(),
			_ => self.tt == rhs.tt,
		}
	}
}

impl PartialEq<TokenType> for Token<'_> {
	fn eq(&self, rhs: &TokenType) -> bool {
		self.tt == *rhs
	}
}

impl PartialEq<&TokenType> for Token<'_> {
	fn eq(&self, rhs: &&TokenType) -> bool {
		self.tt == **rhs
	}
}

use std::fmt;

impl fmt::Display for Token<'_> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		let text = &self.src[self.range.clone()];
		write!(fmt, "{text}")
	}
}

impl fmt::Debug for Token<'_> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		let tt = self.tt;
		let text = &self.src[self.range.clone()];
		write!(fmt, "{tt:?}({text})")
	}
}

