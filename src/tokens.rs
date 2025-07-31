
use std::cmp::PartialEq;
use std::ops::Range;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TokenType {
	Eof,

	// Literals
	Ident(Rc<str>),   // [a-zA-Z_][a-zA-Z0-9_]*
	Integer(Rc<str>), // [0-9_]+
	Fixed(Rc<str>),   // [0-9_]+\.[0-9_]*

	// Keywords
	If,    // 'if'
	Fun,   // 'fn'
	Rec,   // 'rec'
	Var,   // 'var'
	Else,  // 'else'
	While, // 'while'
	True,  // 'true'
	False, // 'false'

	// Value Types
	Bool, // bool
	U8,   // u8
	U16,  // u16
	U32,  // u32
	S8,   // s8
	S16,  // s16
	S32,  // s32
	F16(Rc<str>), // fw[0-9]*
	F32(Rc<str>), // fd[0-9]*

	// Operators
	Amp1,     // '&'
	Amp2,     // '&&'
	At,       // '@'
	Bang,     // '!'
	BangEq,   // '!='
	Bar1,     // '|'
	Bar2,     // '||'
	Carrot1,  // '^'
	Carrot2,  // '^^'
	CBrace,   // '}'
	CParen,   // ')'
	Colon,    // ':'
	Comma,    // ','
	Dollar,   // '$'
	Dot,      // '.'
	Eq1,      // '='
	Eq2,      // '=='
	LArrow1,  // '<'
	LArrow2,  // '<<'
	LArrBar,  // '<|'
	LArrEq,   // '<='
	Minus,    // '-'
	OBrace,   // '{'
	OParen,   // '('
	Percent,  // '%'
	Plus,     // '+'
	RArrow1,  // '>'
	RArrow2,  // '>>'
	RArrBar,  // '|>'
	RArrEq,   // '>='
	RetArrow, // '->'
	Slash,    // '/'
	SlashPer, // '/%'
	Star,     // '*'
}

impl std::fmt::Display for TokenType {
	fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
		let out = match self {
			Self::Eof => "End-of-File",

			// Literals
			Self::Ident(s)   => return write!(f, "'{s}'"),
			Self::Integer(n) => return write!(f, "{n}"),
			Self::Fixed(n)   => return write!(f, "{n}"),

			// Keywords
			Self::If    => "'if' keyword",
			Self::Fun   => "'fn' keyword",
			Self::Rec   => "'rec' keyword",
			Self::Var   => "'var' keyword",
			Self::Else  => "'else' keyword",
			Self::While => "'while' keyword",
			Self::True  => "'true' keyword",
			Self::False => "'false' keyword",

			// Value Types
			Self::Bool => "'bool' type",
			Self::U8   => "'u8' type",
			Self::U16  => "'u16' type",
			Self::U32  => "'u32' type",
			Self::S8   => "'s8' type",
			Self::S16  => "'s16' type",
			Self::S32  => "'s32' type",
			Self::F16(b) => return write!(f, "'fw{b}' type"),
			Self::F32(b) => return write!(f, "'fd{b}' type"),

			// Operators
			Self::Amp1     => "'&' operator",
			Self::Amp2     => "'&&' operator",
			Self::At       => "'@' operator",
			Self::Bang     => "'!' operator",
			Self::BangEq   => "'!=' operator",
			Self::Bar1     => "'|' operator",
			Self::Bar2     => "'||' operator",
			Self::Carrot1  => "'^' operator",
			Self::Carrot2  => "'^^' operator",
			Self::CBrace   => "'}' operator",
			Self::CParen   => "')' operator",
			Self::Colon    => "':' operator",
			Self::Comma    => "',' operator",
			Self::Dollar   => "'$' operator",
			Self::Dot      => "'.' operator",
			Self::Eq1      => "'=' operator",
			Self::Eq2      => "'==' operator",
			Self::LArrow1  => "'<' operator",
			Self::LArrow2  => "'<<' operator",
			Self::LArrBar  => "'<|' operator",
			Self::LArrEq   => "'<=' operator",
			Self::Minus    => "'-' operator",
			Self::OBrace   => "'{' operator",
			Self::OParen   => "'(' operator",
			Self::Percent  => "'%' operator",
			Self::Plus     => "'+' operator",
			Self::RArrow1  => "'>' operator",
			Self::RArrow2  => "'>>' operator",
			Self::RArrBar  => "'|>' operator",
			Self::RArrEq   => "'>=' operator",
			Self::RetArrow => "'->' operator",
			Self::Slash    => "'/' operator",
			Self::SlashPer => "'/%' operator",
			Self::Star     => "'*' operator",
		};

		write!(f, "{out}")
	}
}

#[derive(Clone, Eq)]
pub(crate) struct Token {
	pub(crate) tt: TokenType,
	pub(crate) start: u16,
}

impl Token {
	pub(crate) fn new(
		tt: TokenType,
		start: usize,
	) -> Token {
		Self { tt, start: start as u16 }
	}

	pub(crate) fn range(&self) -> Range<usize> {
		use TokenType as TT;

		let end = match &self.tt {
			TT::Eof => 0,

			TT::Amp1 | TT::At | TT::Bang | TT::Bar1 |
			TT::Carrot1 | TT::CBrace | TT::CParen | TT::Colon | TT::Comma |
			TT::Dollar | TT::Dot | TT::Eq1 | TT::LArrow1 |
			TT::Minus | TT::OBrace | TT::OParen | TT::Percent | TT::Plus |
			TT::RArrow1 | TT::Slash | TT::Star => 1,

			TT::If | TT::U8 | TT::S8 | TT::Amp2 | TT::BangEq |
			TT::Bar2 | TT::Carrot2 | TT::Eq2 | TT::LArrow2 |
			TT::LArrBar | TT::LArrEq | TT::RArrow2 | TT::RetArrow |
			TT::RArrBar | TT::RArrEq | TT::SlashPer => 2,

			TT::Fun | TT::Rec | TT::Var |
			TT::U16 | TT::U32 |
			TT::S16 | TT::S32 => 3,

			TT::Else | TT::True | TT::Bool => 4,

			TT::While | TT::False => 5,

			TT::Ident(s) | TT::Integer(s) | TT::Fixed(s) |
			TT::F16(s) | TT::F32(s) => s.len(),
		};

		self.start as usize..self.start as usize + end
	}
}

impl PartialEq for Token {
	fn eq(&self, rhs: &Self) -> bool {
		match (&self.tt, &rhs.tt) {
			(TokenType::Ident(a), TokenType::Ident(b)) |
			(TokenType::Integer(a), TokenType::Integer(b)) => a == b,
			(TokenType::Fixed(a), TokenType::Fixed(b)) => a == b,
			_ => self.tt == rhs.tt,
		}
	}
}

impl PartialEq<TokenType> for Token {
	fn eq(&self, rhs: &TokenType) -> bool {
		self.tt == *rhs
	}
}

impl PartialEq<&TokenType> for Token {
	fn eq(&self, rhs: &&TokenType) -> bool {
		self.tt == **rhs
	}
}

use std::fmt;

impl fmt::Display for Token {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		use TokenType as TT;

		match &self.tt {
			TT::Ident(s) | TT::Integer(s) | TT::Fixed(s) |
			TT::F16(s) | TT::F32(s) => write!(fmt, "{s}"),
			tt => write!(fmt, "{tt:?}"),
		}
	}
}

impl fmt::Debug for Token {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		write!(fmt, "{:?}[{:?}]", self.tt, self.range())
	}
}

