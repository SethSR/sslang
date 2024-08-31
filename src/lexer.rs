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
	type Item = Result<Token, miette::Error>;

	fn next(&mut self) -> Option<Self::Item> {
		loop {
			let mut chars = self.rest.chars();
			let c = chars.next()?;
			let c_at = self.index;
			self.index += c.len_utf8();
			self.rest = chars.as_str();

			let output = |tt| Some(Ok(Token::new(tt,
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

				'a'..='z' | 'A'..='Z' | '_' => {
					let mut inner_chars = self.rest.char_indices();
					break loop {
						if let Some((j,x)) = inner_chars.next() {
							if ('a'..='z').contains(&x)
							|| ('A'..='Z').contains(&x)
							|| ('0'..='9').contains(&x)
							|| x == '_' {
								continue;
							}
							self.index += j;
							let ident = &self.source[c_at..self.index];
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
							self.rest = &self.rest[j..];
							break Some(Ok(Token::new(tt, c_at..self.index)));
						} else {
							break Some(Err(miette::miette!(
								"Unexpected EOF",
							)));
						}
					}
				}

				'0'..='9' => {
					let mut have_dot = false;
					let mut inner_chars = self.rest.char_indices();
					break loop {
						if let Some((j,x)) = inner_chars.next() {
							if ('0'..='9').contains(&x) {
								continue;
							}
							if x == '.' && !have_dot {
								have_dot = true;
								continue;
							}
							self.index += j;
							self.rest = &self.rest[j..];
							break Some(Ok(Token::new(
								TokenType::Num,
								c_at..self.index,
							)));
						} else {
							break Some(Err(miette::miette!(
								"Unexpected EOF",
							)));
						}
					}
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

