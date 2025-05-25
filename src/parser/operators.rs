
use std::fmt;

use crate::tokens::TokenType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOp {
	/// '.'
	Accessor,
	/// '+'
	Add,
	/// '&'
	AndB,
	/// '&&'
	AndL,
	/// '='
	Assign,
	/// '=='
	CmpEq,
	/// '>='
	CmpGE,
	/// '>'
	CmpGT,
	/// '<='
	CmpLE,
	/// '<'
	CmpLT,
	/// '!='
	CmpNE,
	/// ','
	Comma,
	/// '/'
	Div,
	/// '/%'
	DivMod,
	/// '<|'
	LRot,
	/// '<<'
	LShift,
	/// '%'
	Mod,
	/// '*'
	Mul,
	/// '|'
	OrB,
	/// '||'
	OrL,
	/// '|>'
	RRot,
	/// '>>'
	RShift,
	/// '-'
	Sub,
	/// '^'
	XorB,
	/// '^^'
	XorL,
}

impl TryFrom<TokenType> for BinaryOp {
	type Error = miette::Report;
	fn try_from(tt: TokenType) -> Result<Self, Self::Error> {
		match tt {
			TokenType::Amp1     => Ok(BinaryOp::AndB),
			TokenType::Amp2     => Ok(BinaryOp::AndL),
			TokenType::BangEq   => Ok(BinaryOp::CmpNE),
			TokenType::Bar1     => Ok(BinaryOp::OrB),
			TokenType::Bar2     => Ok(BinaryOp::OrL),
			TokenType::Carrot1  => Ok(BinaryOp::XorB),
			TokenType::Carrot2  => Ok(BinaryOp::XorL),
			TokenType::Comma    => Ok(BinaryOp::Comma),
			TokenType::Dot      => Ok(BinaryOp::Accessor),
			TokenType::Eq1      => Ok(BinaryOp::Assign),
			TokenType::Eq2      => Ok(BinaryOp::CmpEq),
			TokenType::LArrow1  => Ok(BinaryOp::CmpLT),
			TokenType::LArrow2  => Ok(BinaryOp::LShift),
			TokenType::LArrBar  => Ok(BinaryOp::LRot),
			TokenType::LArrEq   => Ok(BinaryOp::CmpLE),
			TokenType::Minus    => Ok(BinaryOp::Sub),
			TokenType::Percent  => Ok(BinaryOp::Mod),
			TokenType::Plus     => Ok(BinaryOp::Add),
			TokenType::RArrow1  => Ok(BinaryOp::CmpGT),
			TokenType::RArrow2  => Ok(BinaryOp::RShift),
			TokenType::RArrBar  => Ok(BinaryOp::RRot),
			TokenType::RArrEq   => Ok(BinaryOp::CmpGE),
			TokenType::Slash    => Ok(BinaryOp::Div),
			TokenType::SlashPer => Ok(BinaryOp::DivMod),
			TokenType::Star     => Ok(BinaryOp::Mul),
			_ => Err(miette::miette! {
				"{tt:?} is not an operator"
			})
		}
	}
}

impl TryFrom<&TokenType> for BinaryOp {
	type Error = miette::Report;
	fn try_from(tt: &TokenType) -> Result<Self, Self::Error> {
		match tt {
			TokenType::Amp1     => Ok(BinaryOp::AndB),
			TokenType::Amp2     => Ok(BinaryOp::AndL),
			TokenType::BangEq   => Ok(BinaryOp::CmpNE),
			TokenType::Bar1     => Ok(BinaryOp::OrB),
			TokenType::Bar2     => Ok(BinaryOp::OrL),
			TokenType::Carrot1  => Ok(BinaryOp::XorB),
			TokenType::Carrot2  => Ok(BinaryOp::XorL),
			TokenType::Comma    => Ok(BinaryOp::Comma),
			TokenType::Dot      => Ok(BinaryOp::Accessor),
			TokenType::Eq1      => Ok(BinaryOp::Assign),
			TokenType::Eq2      => Ok(BinaryOp::CmpEq),
			TokenType::LArrow1  => Ok(BinaryOp::CmpLT),
			TokenType::LArrow2  => Ok(BinaryOp::LShift),
			TokenType::LArrBar  => Ok(BinaryOp::LRot),
			TokenType::LArrEq   => Ok(BinaryOp::CmpLE),
			TokenType::Minus    => Ok(BinaryOp::Sub),
			TokenType::Percent  => Ok(BinaryOp::Mod),
			TokenType::Plus     => Ok(BinaryOp::Add),
			TokenType::RArrow1  => Ok(BinaryOp::CmpGT),
			TokenType::RArrow2  => Ok(BinaryOp::RShift),
			TokenType::RArrBar  => Ok(BinaryOp::RRot),
			TokenType::RArrEq   => Ok(BinaryOp::CmpGE),
			TokenType::Slash    => Ok(BinaryOp::Div),
			TokenType::SlashPer => Ok(BinaryOp::DivMod),
			TokenType::Star     => Ok(BinaryOp::Mul),
			_ => Err(miette::miette! {
				"{tt:?} is not an operator"
			})
		}
	}
}

impl fmt::Display for BinaryOp {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		let op = match self {
			BinaryOp::Accessor => ".",
			BinaryOp::Add      => "+",
			BinaryOp::AndB     => "&",
			BinaryOp::AndL     => "&&",
			BinaryOp::Assign   => "=",
			BinaryOp::CmpEq    => "==",
			BinaryOp::CmpGE    => ">=",
			BinaryOp::CmpGT    => ">",
			BinaryOp::CmpLE    => "<=",
			BinaryOp::CmpLT    => "<",
			BinaryOp::CmpNE    => "!=",
			BinaryOp::Comma    => ",",
			BinaryOp::Div      => "/",
			BinaryOp::DivMod   => "/%",
			BinaryOp::LRot     => "<|",
			BinaryOp::LShift   => "<<",
			BinaryOp::Mod      => "%",
			BinaryOp::Mul      => "*",
			BinaryOp::OrB      => "|",
			BinaryOp::OrL      => "||",
			BinaryOp::RRot     => "|>",
			BinaryOp::RShift   => ">>",
			BinaryOp::Sub      => "-",
			BinaryOp::XorB     => "^",
			BinaryOp::XorL     => "^^",
		};
		write!(fmt, "{op}")
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOp {
	/// '@'
	Deref,
	/// '-'
	Neg,
	/// '!'
	Not,
	/// '+'
	Pos,
	/// '$'
	Ref,
}

impl TryFrom<TokenType> for UnaryOp {
	type Error = miette::Report;
	fn try_from(tt: TokenType) -> Result<Self, Self::Error> {
		match tt {
			TokenType::At       => Ok(UnaryOp::Deref),
			TokenType::Bang     => Ok(UnaryOp::Not),
			TokenType::Dollar   => Ok(UnaryOp::Ref),
			TokenType::Minus    => Ok(UnaryOp::Neg),
			TokenType::Plus     => Ok(UnaryOp::Pos),
			_ => Err(miette::miette! {
				"{tt:?} is not a unary operator"
			})
		}
	}
}

impl TryFrom<&TokenType> for UnaryOp {
	type Error = miette::Report;
	fn try_from(tt: &TokenType) -> Result<Self, Self::Error> {
		match tt {
			TokenType::At       => Ok(UnaryOp::Deref),
			TokenType::Bang     => Ok(UnaryOp::Not),
			TokenType::Dollar   => Ok(UnaryOp::Ref),
			TokenType::Minus    => Ok(UnaryOp::Neg),
			TokenType::Plus     => Ok(UnaryOp::Pos),
			_ => Err(miette::miette! {
				"{tt:?} is not a unary operator"
			})
		}
	}
}

impl fmt::Display for UnaryOp {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		let op = match self {
			UnaryOp::Deref => "@",
			UnaryOp::Neg   => "-",
			UnaryOp::Not   => "!",
			UnaryOp::Pos   => "+",
			UnaryOp::Ref   => "$",
		};
		write!(fmt, "{op}")
	}
}

