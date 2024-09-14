
mod tokens;
mod lexer;
mod parser;

fn main() -> miette::Result<()> {
	let mut buffer = String::default();

	loop {
		buffer.clear();

		if let Err(e) = std::io::stdin().read_line(&mut buffer) {
			eprintln!("{e}");
			continue;
		}

		if buffer.trim() == "exit" {
			break;
		}

		let _test = "
rec vec {
	x: fl,
	y: fl,
}

rec quat {
	s: fl,
	v: fl,
}

fun main() {
	let x: fl12 = 0.44;
	let y: fl12 = 0.01;

	let p = vec { x, y };
	let q = vec { 1.5, 2.6 };

	fun vmul(a: vec, b: vec) -> quat {
		quat {
			a.x * b.x + a.y * b.y,
			a.x * b.y - b.x * a.y,
		}
	}

	vmul(p,q)
}
";

		let tokens = lexer::eval(&buffer)?;

		let ast = match parser::eval(tokens) {
			Ok(out) => out,
			Err(e) => {
				eprintln!("{e}");
				continue;
			}
		};
		println!("{ast:#?}");
		//stage1::eval(&ast, 0);
	}

	Ok(())
}

/*
mod stage1 {
	use crate::tokens::TokenType;
	use crate::parser::S;

	const R_MAX: usize = 1;

	pub(crate) fn eval(ast: &S, r: usize) {
		match ast {
			S::Expr(token) => match token.tt {
				TokenType::Num => {
					let num = token.to_string().parse::<f64>().unwrap();
					if num.fract().abs() < 0.00000001 && (i8::MIN..=i8::MAX).contains(&(num.trunc() as i8)) {
						if r < R_MAX {
							println!("MOV #{},R{r}", num.trunc() as i8);
						} else {
							println!("MOV #{},@-SP", num.trunc() as i8);
						}
					} else if num.fract().abs() < 0.00000001 && (i16::MIN..=i16::MAX).contains(&(num.trunc() as i16)) {
						todo!("store i16 value and indirectly load into R0");
					} else if num.fract().abs() < 0.00000001 && (i32::MIN..=i32::MAX).contains(&(num.trunc() as i32)) {
						todo!("store i32 value and indirectly load into R0")
					}
				}
				_ => eprintln!("ERROR: non-numeric literal return value from program"),
			}
			S::List(list) => match &list[..] {
				[] => {}
				[first, rest @ ..] => {
					match first {
						S::Expr(op) => match op.tt {
							TokenType::Plus => {
								if r < R_MAX {
									println!("MOV #0,R{r}");
								} else {
									println!("MOV #0,@-SP");
								}
								for item in rest {
									match item {
										S::Expr(t) => if r < R_MAX {
											println!("ADD #{},R{r}", t.to_string());
										} else {
											println!("MOV @SP+,R{R_MAX}");
											println!("ADD #{},R{R_MAX}", t.to_string());
											println!("MOV R{R_MAX},@-SP");
										}
										_ => {
											eval(item, r+1);
											if r+1 < R_MAX {
												println!("ADD R{},R{r}", r+1);
											} else {
												println!("MOV @SP+,R{R_MAX}");
												println!("ADD R{R_MAX},R{r}");
											}
										}
									}
								}
							}
							TokenType::Minus => {}
							TokenType::Star => {}
							TokenType::Slash => {}
							//TokenType::SlashPercent => {}
							TokenType::Percent => {}
							//TokenType::DbLArrow => {}
							//TokenType::DbRArrow => {}
							//TokenType::Diamond => {}
							TokenType::LArrow1 => {}
							//TokenType::LArrEq => {}
							TokenType::RArrow1 => {}
							//TokenType::RArrEq => {}
							_ => eprintln!("Unknown Op '{first}'"),
						}
						S::List(_) => {}
					}
				}
			}
		}
	}
}
*/
