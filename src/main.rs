
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

/*
		let test = "
let (x u8) (+ 1 2)
fn do_stuff (
	(a s16)
	(b fw8)
) (+ x (+ (* a 2) (* b 3)))
rec vec (
	(x fl16)
	(y fl16)
)
rec quat (
	(s fl16)
	(v fl16)
)
fn main () (
	(let (x fl20) 0.44)
	(let (y fl20) 0.01)

	(let (p) (vec x y))
	(let (q) (vec 1.5 2.6))

	(fn vmul ((a vec) (b vec)) (
		quat
			(+ (* a.x b.x) (* a.y b.y))
			(- (* a.x b.y) (* b.x a.y))
	))
	vmul p q
)";
*/

		let tokens = lexer::eval(&buffer)?;

		let ast = match parser::eval(tokens) {
			Ok(out) => out,
			Err(e) => {
				eprintln!("{e}");
				continue;
			}
		};
		println!("({})", ast.iter()
			.map(|i| format!("{i}"))
			.collect::<Vec<String>>()
			.join(" "));
	}

	Ok(())
}
