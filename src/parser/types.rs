
use std::fmt;

// TODO - srenshaw - Create a more robust type system. (Probably based on github.com/SeaOfNodes)

pub(crate) trait Meet {
	fn meet(&self, rhs: &Self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unsigned {
	Top, U8, U16, U32, Bot,
}

impl fmt::Display for Unsigned {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Top => write!(f, "U::Top"),
			Self::U8 => write!(f, "u8"),
			Self::U16 => write!(f, "u16"),
			Self::U32 => write!(f, "u32"),
			Self::Bot => write!(f, "U::Bot"),
		}
	}
}

impl Meet for Unsigned {
	fn meet(&self, rhs: &Self) -> Self {
		match (self, rhs) {
			(Self::Bot, _) | (_, Self::Bot) => Self::Bot,
			(Self::U32, _) | (_, Self::U32) => Self::U32,
			(Self::U16, _) | (_, Self::U16) => Self::U16,
			(Self::U8,  _) | (_, Self::U8)  => Self::U8,
			(Self::Top, Self::Top) => Self::Top,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Signed {
	Top, S8, S16, S32, Bot,
}

impl fmt::Display for Signed {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Top => write!(f, "S::Top"),
			Self::S8 => write!(f, "s8"),
			Self::S16 => write!(f, "s16"),
			Self::S32 => write!(f, "s32"),
			Self::Bot => write!(f, "S::Bot"),
		}
	}
}

impl Meet for Signed {
	fn meet(&self, rhs: &Self) -> Self {
		match (self, rhs) {
			(Self::Bot, _) | (_, Self::Bot) => Self::Bot,
			(Self::S32, _) | (_, Self::S32) => Self::S32,
			(Self::S16, _) | (_, Self::S16) => Self::S16,
			(Self::S8,  _) | (_, Self::S8)  => Self::S8,
			(Self::Top, Self::Top) => Self::Top,
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Int {
	Top,
	Unsigned(Unsigned),
	Signed(Signed),
	Bot,
}

impl fmt::Display for Int {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Top => write!(f, "I::Top"),
			Self::Unsigned(u) => write!(f, "{u}"),
			Self::Signed(s) => write!(f, "{s}"),
			Self::Bot => write!(f, "I::Bot"),
		}
	}
}

impl Meet for Int {
	fn meet(&self, rhs: &Self) -> Self {
		match (self, rhs) {
			(Self::Bot, _) | (_, Self::Bot) |
			(Self::Signed(_), Self::Unsigned(_)) |
			(Self::Unsigned(_), Self::Signed(_)) => Self::Bot,

			(Self::Unsigned(a), Self::Unsigned(b)) => Self::Unsigned(a.meet(b)),

			(Self::Signed(a), Self::Signed(b)) => Self::Signed(a.meet(b)),

			(Self::Top, rhs) => rhs.clone(),
			(lhs, Self::Top) => lhs.clone(),
		}
	}
}

// TODO - srenshaw - Change 'fw' and 'fd' to 'f' and 'd'?

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fix {
	Top,
	F16(u8), // 16-bit fixed point with <u8> integer bits
	F32(u8), // 32-bit fixed point with <u8> integer bits
	Bot,
}

impl fmt::Display for Fix {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		match self {
			Self::Top => write!(f, "F::Top"),
			Self::F16(n) => write!(f, "fw{n}"),
			Self::F32(n) => write!(f, "fd{n}"),
			Self::Bot => write!(f, "F::Bot"),
		}
	}
}

impl Meet for Fix {
	fn meet(&self, rhs: &Self) -> Self {
		match (self, rhs) {
			(Self::Bot, _) | (_, Self::Bot) => Self::Bot,

			(Self::F16(a), Self::F16(b)) => if a == b {
				Self::F16(*a)
			} else {
				Self::Bot
			}
			(Self::F16(a), Self::F32(b)) => if (*b..=*b+16).contains(a) {
				Self::F32(*b)
			} else {
				Self::Bot
			}

			(Self::F32(a), Self::F16(b)) => if (*a..=*a+16).contains(b) {
				Self::F32(*a)
			} else {
				Self::Bot
			}
			(Self::F32(a), Self::F32(b)) => if a == b {
				Self::F32(*a)
			} else {
				Self::Bot
			}

			(Self::Top, rhs) => rhs.clone(),
			(lhs, Self::Top) => lhs.clone(),
		}
	}
}

// TODO - srenshaw - Add UDT cache to parser::Parser.

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ValueType {
	// Used for function calls, we'll resolve these later
	/// Any type - also used for unknown/unresolved types during compilation
	Any,
	/// All integer types
	Int(Int),
	/// All Fixed-point types
	Fix(Fix),
	/// All User Defined Types
	///
	/// UDTs are stored in a type-cache in parser::Parser.
	UDT(String),
	/// Bottom type
	Unit,
}

impl ValueType {
	pub(crate) fn to_u8() -> Self {
		Self::Int(Int::Unsigned(Unsigned::U8))
	}

	pub(crate) fn to_u16() -> Self {
		Self::Int(Int::Unsigned(Unsigned::U16))
	}

	pub(crate) fn to_u32() -> Self {
		Self::Int(Int::Unsigned(Unsigned::U32))
	}

	pub(crate) fn to_s8() -> Self {
		Self::Int(Int::Signed(Signed::S8))
	}

	pub(crate) fn to_s16() -> Self {
		Self::Int(Int::Signed(Signed::S16))
	}

	pub(crate) fn to_s32() -> Self {
		Self::Int(Int::Signed(Signed::S32))
	}

	pub(crate) fn to_f16(bits: u8) -> Self {
		Self::Fix(Fix::F16(bits))
	}

	pub(crate) fn to_f32(bits: u8) -> Self {
		Self::Fix(Fix::F32(bits))
	}
}

impl Meet for ValueType {
	fn meet(&self, rhs: &Self) -> Self {
		match (self, rhs) {
			(Self::Unit, _) | (_, Self::Unit) => Self::Unit,

			(Self::Int(a), Self::Int(b)) => Self::Int(a.meet(b)),
			(Self::Int(_), Self::Fix(_)) => Self::Unit,
			(Self::Int(_), Self::UDT(_)) => Self::Unit,
			(Self::Int(int), Self::Any) => Self::Int(*int),

			(Self::Fix(_), Self::Int(_)) => Self::Unit,
			(Self::Fix(a), Self::Fix(b)) => Self::Fix(a.meet(b)),
			(Self::Fix(_), Self::UDT(_)) => Self::Unit,
			(Self::Fix(fix), Self::Any) => Self::Fix(*fix),

			(Self::UDT(_), Self::Int(_)) => Self::Unit,
			(Self::UDT(_), Self::Fix(_)) => Self::Unit,
			(Self::UDT(_), Self::UDT(_)) => Self::Unit,
			(Self::UDT(udt), Self::Any) => Self::UDT(udt.clone()),

			(Self::Any, Self::Int(_)) => Self::Unit,
			(Self::Any, Self::Fix(_)) => Self::Unit,
			(Self::Any, Self::UDT(_)) => Self::Unit,
			(Self::Any, Self::Any) => Self::Any,
		}
	}
}

impl fmt::Display for ValueType {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		use ValueType as VT;

		match self {
			VT::Int(int) => write!(fmt, "{int}"),
			VT::Fix(fix) => write!(fmt, "{fix}"),
			VT::UDT(s) => write!(fmt, "{s}"),
			VT::Unit => write!(fmt, "()"),
			VT::Any => write!(fmt, "??"),
		}
	}
}

