
use std::ops::Range;
use std::cmp::PartialEq;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenType<'a> {
	EOF,

	// Literals
	Ident(&'a str),  // [a-zA-Z_][a-zA-Z0-9_]*
	Number(&'a str), // [0-9_]+\.?[0-9_]*

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
	F16(&'a str), // fw[0-9]*
	F32(&'a str), // fl[0-9]*

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
	Comma,    // ,
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
	RetArrow, // ->
	Slash,    // /
	SlashPer, // /%
	Star,     // *
}

#[derive(Clone, Eq)]
pub(crate) struct Token<'a> {
	pub(crate) tt: TokenType<'a>,
	pub(crate) start: u16,
}

impl<'a> Token<'a> {
	pub(crate) fn new(
		tt: TokenType<'a>,
		start: usize,
	) -> Token<'a> {
		Self { tt, start: start as u16 }
	}

	pub(crate) fn range(&self) -> Range<usize> {
		use TokenType as TT;

		let end = match self.tt {
			TT::EOF => 0,

			TT::Amp1 | TT::At | TT::Bang | TT::Bar1 |
			TT::Carrot1 | TT::CParen | TT::Colon | TT::Comma |
			TT::Dollar | TT::Dot | TT::Eq1 | TT::LArrow1 |
			TT::Minus | TT::OParen | TT::Percent | TT::Plus |
			TT::RArrow1 | TT::Slash | TT::Star => 1,

			TT::If | TT::U8 | TT::S8 | TT::Amp2 | TT::BangEq |
			TT::Bar2 | TT::Carrot2 | TT::Eq2 | TT::LArrow2 |
			TT::LArrBar | TT::LArrEq | TT::RArrow2 | TT::RetArrow |
			TT::RArrBar | TT::RArrEq | TT::SlashPer => 2,

			TT::Fun | TT::Rec | TT::Var |
			TT::U16 | TT::U32 |
			TT::S16 | TT::S32 => 3,

			TT::Else => 4,

			TT::While => 5,

			TT::Ident(s) | TT::Number(s) |
			TT::F16(s) | TT::F32(s) => s.len(),
		};

		self.start as usize..self.start as usize + end
	}
}

impl PartialEq for Token<'_> {
	fn eq(&self, rhs: &Self) -> bool {
		match (self.tt, rhs.tt) {
			(TokenType::Ident(a), TokenType::Ident(b)) |
			(TokenType::Number(a), TokenType::Number(b)) => a == b,
			_ => self.tt == rhs.tt,
		}
	}
}

impl PartialEq<TokenType<'_>> for Token<'_> {
	fn eq(&self, rhs: &TokenType) -> bool {
		self.tt == *rhs
	}
}

impl PartialEq<&TokenType<'_>> for Token<'_> {
	fn eq(&self, rhs: &&TokenType) -> bool {
		self.tt == **rhs
	}
}

use std::fmt;

impl fmt::Display for Token<'_> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		use TokenType as TT;

		match self.tt {
			TT::Ident(s) | TT::Number(s) |
			TT::F16(s) | TT::F32(s) => write!(fmt, "{s}"),
			tt => write!(fmt, "{tt:?}"),
		}
	}
}

impl fmt::Debug for Token<'_> {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		write!(fmt, "{:?}", self.tt)
	}
}

