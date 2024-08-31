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
}

impl<'a> Iterator for Lexer<'a> {
	type Item = Result<Token<'a>, miette::Error>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let mut chars = self.rest.chars();
			let c = chars.next()?;
			let c_at = self.index;
			self.index += c.len_utf8();
			self.rest = chars.as_str();

			let output = |tt| Some(Ok(Token::new(tt,
				self.source,
				c_at..self.index)));

			match c {
				'&' => break output(TokenType::Ampersand),
				'!' => break output(TokenType::Bang),
				'.' => break output(TokenType::Dot),
				'<' => break output(TokenType::LArrow),
				'(' => break output(TokenType::OParen),
				'%' => break output(TokenType::Percent),
				'|' => break output(TokenType::Pipe),
				'+' => break output(TokenType::Plus),
				'-' => break output(TokenType::Minus),
				'>' => break output(TokenType::RArrow),
				')' => break output(TokenType::CParen),
				'/' => break output(TokenType::Slash),
				'*' => break output(TokenType::Star),
				'^' => break output(TokenType::Carrot),
				'=' => break output(TokenType::Eq),

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
							index += 1;
						}
						self.index += index;
						self.rest = &self.rest[index..];
						break;
					}

					let ident = &self.source[c_at..self.index];
					eprintln!("ident:{ident}");
					let tt = match ident {
						"fn"  => TokenType::Fn,
						"let" => TokenType::Let,
						"rec" => TokenType::Rec,
						"u8"  => TokenType::U8,
						"u16" => TokenType::U16,
						"u32" => TokenType::U32,
						"s8"  => TokenType::S8,
						"s16" => TokenType::S16,
						"s32" => TokenType::S32,
						_ => if ident.starts_with("fw") {
							TokenType::F16
						} else if ident.starts_with("fl") {
							TokenType::F32
						} else {
							TokenType::Ident
						}
					};
					break Some(Ok(Token::new(tt, self.source, c_at..self.index)));
				}

				'0'..='9' => {
					let mut have_dot = false;
					let mut inner_chars = self.rest.char_indices();
					let mut index = 0;
					loop {
						if let Some((j,x)) = inner_chars.next() {
							index = j;
							if ('0'..='9').contains(&x) {
								continue;
							}
							if x == '.' && !have_dot {
								have_dot = true;
								continue;
							}
						} else {
							index += 1;
						}
						self.index += index;
						self.rest = &self.rest[index..];
						break;
					}
					break Some(Ok(Token::new(
						TokenType::Num,
						self.source,
						c_at..self.index,
					)));
				}

				w if w.is_whitespace() => {
					let mut inner_chars = self.rest.char_indices();
					loop {
						if let Some((j,x)) = inner_chars.next() {
							if x.is_ascii_whitespace() {
								continue;
							}
							self.index += j;
							self.rest = &self.rest[j..];
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
	Lexer::new(input).collect()
}

#[test]
fn tokenizes_empty_input() -> miette::Result<()> {
	let tokens = eval("")?;
	assert_eq!(tokens, vec![]);
	Ok(())
}

#[test]
fn tokenizes_let() -> miette::Result<()> {
	let input = "let";
	let tokens = eval(input)?;
	assert_eq!(tokens, vec![Token::new(TokenType::Let, input, 0..3)]);
	Ok(())
}

#[test]
fn tokenizes_fn() -> miette::Result<()> {
	let input = "fn";
	let tokens = eval(input)?;
	assert_eq!(tokens, vec![Token::new(TokenType::Fn, input, 0..2)]);
	Ok(())
}

#[test]
fn tokenizes_input_without_wrapping_parentheses(
) -> miette::Result<()> {
	let input = "let (a u8) (3)";
	let tokens = eval(input)?;
	assert_eq!(tokens, vec![
		Token::new(TokenType::Let, input, 0..3),
		Token::new(TokenType::OParen, input, 4..5),
		Token::new(TokenType::Ident, input, 5..6),
		Token::new(TokenType::U8, input, 7..9),
		Token::new(TokenType::CParen, input, 9..10),
		Token::new(TokenType::OParen, input, 11..12),
		Token::new(TokenType::Num, input, 12..13),
		Token::new(TokenType::CParen, input, 13..14),
	]);
	Ok(())
}
