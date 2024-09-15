
use miette::LabeledSpan;

use crate::tokens::{Token, TokenType};

pub(crate) struct Lexer<'a> {
	source: &'a str,
	rest: &'a str,
	index: usize,
}

impl<'a> Lexer<'a> {
	pub(crate) fn new(input: &'a str) -> Self {
		Self {
			source: input,
			rest: input,
			index: 0,
		}
	}

	fn next(&mut self, index: usize) {
		self.index += index;
		self.rest = &self.rest[index..];
	}
}

impl<'a> Iterator for Lexer<'a> {
	type Item = Result<Token<'a>, miette::Error>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let mut chars = self.rest.chars();
			let c = chars.next()?;
			let c_at = self.index;
			self.next(c.len_utf8());

			let output = |tt,idx| Some(Ok(Token::new(tt, self.source, c_at..idx)));

			match c {
				')' => break output(TokenType::CParen, self.index),
				':' => break output(TokenType::Colon, self.index),
				'.' => break output(TokenType::Dot, self.index),
				'(' => break output(TokenType::OParen, self.index),
				'%' => break output(TokenType::Percent, self.index),
				'+' => break output(TokenType::Plus, self.index),
				'*' => break output(TokenType::Star, self.index),
				'@' => break output(TokenType::At, self.index),
				'$' => break output(TokenType::Dollar, self.index),

				'^' => break if let Some((j,'^')) = self.rest.char_indices().next() {
					self.next(j);
					output(TokenType::Carrot2, self.index)
				} else {
					output(TokenType::Carrot1, self.index)
				},
				'&' => break if let Some((j,'&')) = self.rest.char_indices().next() {
					self.next(j);
					output(TokenType::Amp2, self.index)
				} else {
					output(TokenType::Amp1, self.index)
				},
				'!' => break if let Some((j,'=')) = self.rest.char_indices().next() {
					self.next(j);
					output(TokenType::BangEq, self.index)
				} else {
					output(TokenType::Bang, self.index)
				},
				'=' => break if let Some((j,'=')) = self.rest.char_indices().next() {
					self.next(j);
					output(TokenType::Eq2, self.index)
				} else {
					output(TokenType::Eq1, self.index)
				},
				'/' => break if let Some((j,'%')) = self.rest.char_indices().next() {
					self.next(j);
					output(TokenType::SlashPer, self.index)
				} else {
					output(TokenType::Slash, self.index)
				},
				'|' => break match self.rest.char_indices().next() {
					Some((j,'|')) => {
						self.next(j);
						output(TokenType::Bar2, self.index)
					}
					Some((j,'>')) => {
						self.next(j);
						output(TokenType::RArrBar, self.index)
					}
					_ => output(TokenType::Bar1, self.index),
				},
				'<' => break match self.rest.char_indices().next() {
					Some((j,'<')) => {
						self.next(j);
						output(TokenType::LArrow2, self.index)
					}
					Some((j,'|')) => {
						self.next(j);
						output(TokenType::LArrBar, self.index)
					}
					Some((j,'=')) => {
						self.next(j);
						output(TokenType::LArrEq, self.index)
					}
					_ => output(TokenType::LArrow1, self.index),
				},
				'>' => break match self.rest.char_indices().next() {
					Some((j,'>')) => {
						self.next(j);
						output(TokenType::RArrow2, self.index)
					}
					Some((j,'=')) => {
						self.next(j);
						output(TokenType::RArrEq, self.index)
					}
					_ => output(TokenType::RArrow1, self.index),
				},

				'-' => {
					let mut inner_chars = self.rest.char_indices();
					if let Some((j,'-')) = inner_chars.next() {
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
							break self.next(index);
						}
					} else {
						break output(TokenType::Minus, self.index);
					}
				}

				'a'..='z' | 'A'..='Z' | '_' => {
					let mut inner_chars = self.rest.char_indices();
					let mut index = 0;
					loop {
						if let Some((j,x)) = inner_chars.next() {
							index = j;
							if ('a'..='z').contains(&x)
							|| ('A'..='Z').contains(&x)
							|| ('0'..='9').contains(&x)
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
						"fun"   => TokenType::Fun,
						"rec"   => TokenType::Rec,
						"var"   => TokenType::Var,
						"else"  => TokenType::Else,
						"while" => TokenType::While,
						"u8"    => TokenType::U8,
						"u16"   => TokenType::U16,
						"u32"   => TokenType::U32,
						"s8"    => TokenType::S8,
						"s16"   => TokenType::S16,
						"s32"   => TokenType::S32,
						_ => if ident.starts_with("fw") {
							TokenType::F16
						} else if ident.starts_with("fl") {
							TokenType::F32
						} else {
							TokenType::Ident
						}
					}, self.index);
				}

				'0'..='9' => {
					let mut have_dot = false;
					let mut inner_chars = self.rest.char_indices();
					let mut index = 0;
					loop {
						if let Some((j,x)) = inner_chars.next() {
							index = j;
							if ('0'..='9').contains(&x) || x == '_' {
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
					break output(TokenType::Number, self.index);
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

	let input_rng = input.len()..input.len();
	Lexer::new(input)
		.chain(once(Ok(Token::new(TokenType::EOF, input, input_rng))))
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
			.chain(once(&TokenType::EOF))
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
		lex_test("var fun rec if while else", &[
			TokenType::Var,
			TokenType::Fun,
			TokenType::Rec,
			TokenType::If,
			TokenType::While,
			TokenType::Else,
		])
	}

	#[test]
	fn value_types() -> miette::Result<()> {
		lex_test("u8 u16 u32 s8 s16 s32 fw fw4 fl fl22", &[
			TokenType::U8,
			TokenType::U16,
			TokenType::U32,
			TokenType::S8,
			TokenType::S16,
			TokenType::S32,
			TokenType::F16,
			TokenType::F16,
			TokenType::F32,
			TokenType::F32,
		])
	}

	#[test]
	fn identifiers() -> miette::Result<()> {
		lex_test("abc_123 _123", &[
			TokenType::Ident,
			TokenType::Ident,
		])
	}

	#[test]
	fn numbers() -> miette::Result<()> {
		lex_test("1 2.34 5_6 7_8.9", &[
			TokenType::Number,
			TokenType::Number,
			TokenType::Number,
			TokenType::Number,
		])
	}
}
