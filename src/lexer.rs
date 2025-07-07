
use std::rc::Rc;

use miette::LabeledSpan;

use crate::tokens::{Token, TokenType};

pub(crate) struct Lexer {
	source: Rc<str>,
	rest: Rc<str>,
	index: usize,
}

impl Lexer {
	pub(crate) fn new(input: &str) -> Self {
		let input: Rc<str> = input.into();
		Self {
			source: Rc::clone(&input),
			rest: input,
			index: 0,
		}
	}

	fn next(&mut self, index: usize) {
		self.index += index;
		self.rest = self.rest[index..].into();
	}
}

impl Iterator for Lexer {
	type Item = Result<Token, miette::Error>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let mut chars = self.rest.chars();
			let c = chars.next()?;
			let c_at = self.index;
			self.next(c.len_utf8());

			let output = |tt| Some(Ok(Token::new(tt, c_at)));

			match c {
				'$' => break output(TokenType::Dollar),
				'%' => break output(TokenType::Percent),
				'(' => break output(TokenType::OParen),
				')' => break output(TokenType::CParen),
				'*' => break output(TokenType::Star),
				'+' => break output(TokenType::Plus),
				',' => break output(TokenType::Comma),
				'.' => break output(TokenType::Dot),
				':' => break output(TokenType::Colon),
				'@' => break output(TokenType::At),
				'{' => break output(TokenType::OBrace),
				'}' => break output(TokenType::CBrace),

				'^' => break if let Some((j,'^')) = self.rest.char_indices().next() {
					self.next(j+1);
					output(TokenType::Carrot2)
				} else {
					output(TokenType::Carrot1)
				},
				'&' => break if let Some((j,'&')) = self.rest.char_indices().next() {
					self.next(j+1);
					output(TokenType::Amp2)
				} else {
					output(TokenType::Amp1)
				},
				'!' => break if let Some((j,'=')) = self.rest.char_indices().next() {
					self.next(j+1);
					output(TokenType::BangEq)
				} else {
					output(TokenType::Bang)
				},
				'=' => break if let Some((j,'=')) = self.rest.char_indices().next() {
					self.next(j+1);
					output(TokenType::Eq2)
				} else {
					output(TokenType::Eq1)
				},
				'/' => break if let Some((j,'%')) = self.rest.char_indices().next() {
					self.next(j+1);
					output(TokenType::SlashPer)
				} else {
					output(TokenType::Slash)
				},
				'|' => break match self.rest.char_indices().next() {
					Some((j,'|')) => {
						self.next(j+1);
						output(TokenType::Bar2)
					}
					Some((j,'>')) => {
						self.next(j+1);
						output(TokenType::RArrBar)
					}
					_ => output(TokenType::Bar1),
				},
				'<' => break match self.rest.char_indices().next() {
					Some((j,'<')) => {
						self.next(j+1);
						output(TokenType::LArrow2)
					}
					Some((j,'|')) => {
						self.next(j+1);
						output(TokenType::LArrBar)
					}
					Some((j,'=')) => {
						self.next(j+1);
						output(TokenType::LArrEq)
					}
					_ => output(TokenType::LArrow1),
				},
				'>' => break match self.rest.char_indices().next() {
					Some((j,'>')) => {
						self.next(j+1);
						output(TokenType::RArrow2)
					}
					Some((j,'=')) => {
						self.next(j+1);
						output(TokenType::RArrEq)
					}
					_ => output(TokenType::RArrow1),
				},

				'-' => {
					let mut inner_chars = self.rest.char_indices();
					match inner_chars.next() {
						Some((j,'-')) => {
							let mut index = j;
							loop {
								if let Some((j,x)) = inner_chars.next() {
									index = j;
									if x != '\n' {
										continue;
									}
								} else {
									index += self.rest.len() - index;
								}
								self.next(index);
								break;
							}
						}
						Some((j,'>')) => {
							self.next(j+1);
							break output(TokenType::RetArrow);
						}
						_ => break output(TokenType::Minus),
					}
				}

				'a'..='z' | 'A'..='Z' | '_' => {
					let mut inner_chars = self.rest.char_indices();
					let mut index = 0;
					loop {
						if let Some((j,x)) = inner_chars.next() {
							index = j;
							if x.is_ascii_lowercase()
							|| x.is_ascii_uppercase()
							|| x.is_ascii_digit()
							|| x == '_' {
								continue;
							}
						} else {
							index += self.rest.len() - index;
						}
						break self.next(index);
					}

					let ident = &self.source[c_at..self.index];
					break output(match ident {
						"if"    => TokenType::If,
						"fn"    => TokenType::Fun,
						"rec"   => TokenType::Rec,
						"var"   => TokenType::Var,
						"else"  => TokenType::Else,
						"while" => TokenType::While,
						"true"  => TokenType::True,
						"false" => TokenType::False,
						"bool"  => TokenType::Bool,
						"u8"    => TokenType::U8,
						"u16"   => TokenType::U16,
						"u32"   => TokenType::U32,
						"s8"    => TokenType::S8,
						"s16"   => TokenType::S16,
						"s32"   => TokenType::S32,
						s => if ident.starts_with("fw") {
							TokenType::F16(s.into())
						} else if ident.starts_with("fd") {
							TokenType::F32(s.into())
						} else {
							TokenType::Ident(s.into())
						}
					});
				}

				'0'..='9' => {
					let mut have_dot = false;
					let mut inner_chars = self.rest.char_indices();
					let mut index = 0;
					loop {
						if let Some((j,x)) = inner_chars.next() {
							index = j;
							if x.is_ascii_digit() || x == '_' {
								continue;
							}
							if x == '.' && !have_dot {
								have_dot = true;
								continue;
							}
						} else {
							index += self.rest.len() - index;
						}
						break self.next(index);
					}
					let s = &self.source[c_at..self.index];
					break if have_dot {
						output(TokenType::Fixed(s.into()))
					} else {
						output(TokenType::Integer(s.into()))
					};
				}

				w if w.is_whitespace() => {
					let mut inner_chars = self.rest.char_indices();
					loop {
						if let Some((j,x)) = inner_chars.next() {
							if x.is_ascii_whitespace() {
								continue;
							}
							self.next(j);
						}
						break;
					}
				}

				_ => break Some(Err(miette::miette! {
					labels = vec![
						LabeledSpan::at(
							c_at..self.index,
							"this character",
						),
					],
					"Unexpected char '{c}'"
				}.with_source_code(self.source.to_string()))),
			}
		}
	}
}

pub(crate) fn eval(
	input: &str,
) -> miette::Result<Vec<Token>> {
	use std::iter::once;

	Lexer::new(input)
		.chain(once(Ok(Token::new(TokenType::Eof, input.len()))))
		.collect()
}

#[cfg(test)]
mod tokenizes {
	use super::TokenType;

	fn lex_test(
		input: &str,
		check: &[TokenType],
	) -> miette::Result<()> {
		use std::iter::once;

		let tokens = crate::lexer::eval(input)?;
		assert_eq!(tokens, check.iter()
			.chain(once(&TokenType::Eof))
			.collect::<Vec<&TokenType>>());
		Ok(())
	}

	#[test]
	fn empty_input() -> miette::Result<()> {
		lex_test("", &[])
	}

	#[test]
	fn white_spaces_as_empty_input() -> miette::Result<()> {
		lex_test("  \n\t  ", &[])
	}

	#[test]
	fn comments_as_empty_input() -> miette::Result<()> {
		lex_test("-- var if hello", &[])
	}

	#[test]
	fn keywords() -> miette::Result<()> {
		lex_test("var fn rec if while else true false", &[
			TokenType::Var,
			TokenType::Fun,
			TokenType::Rec,
			TokenType::If,
			TokenType::While,
			TokenType::Else,
			TokenType::True,
			TokenType::False,
		])
	}

	#[test]
	fn value_types() -> miette::Result<()> {
		lex_test("bool u8 u16 u32 s8 s16 s32 fw fw4 fd fd22", &[
			TokenType::Bool,
			TokenType::U8,
			TokenType::U16,
			TokenType::U32,
			TokenType::S8,
			TokenType::S16,
			TokenType::S32,
			TokenType::F16("fw".into()),
			TokenType::F16("fw4".into()),
			TokenType::F32("fd".into()),
			TokenType::F32("fd22".into()),
		])
	}

	#[test]
	fn identifiers() -> miette::Result<()> {
		lex_test("abc_123 _123", &[
			TokenType::Ident("abc_123".into()),
			TokenType::Ident("_123".into()),
		])
	}

	#[test]
	fn numbers() -> miette::Result<()> {
		lex_test("1 2.34 5_6 7_8.9", &[
			TokenType::Integer("1".into()),
			TokenType::Fixed("2.34".into()),
			TokenType::Integer("5_6".into()),
			TokenType::Fixed("7_8.9".into()),
		])
	}
}
