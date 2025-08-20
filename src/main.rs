
use std::rc::Rc;

use miette::IntoDiagnostic;
use tracing::{info,debug};

mod checker;
mod tokens;
mod lexer;
mod parser;
mod reducer;

const TEST_INPUT: &str = "
rec vec {
	x:fd,
	y:fd,
}

rec quat {
	s:fd,
	v:fd,
}

fn main() {
	var x:fd12 = 0.44;
	var y:fd12 = 0.01;

	var p = vec {x:x, y:y};
	var q = vec {x:1.5, y:2.6};

	if x < y {
		x = y;
	} else {
		y = x;
	}

	while p.x > q.y {
		var b = 3;
		var x = p.y + 1;
	}

	fn vmul(a:vec, b:vec) -> quat {
		quat {
			s: a.x * b.x + a.y * b.y,
			v: a.x * b.y - b.x * a.y,
		}
	}

	vmul(p, q)
}

main()";

use clap::Parser;

use parser::{Error, Step};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
	Lexer,
	Parser,
	Checker,
	Reducer,
}

impl From<String> for Stage {
	fn from(s: String) -> Self {
		match s.to_lowercase().as_str() {
			"lexer" => Self::Lexer,
			"parser" => Self::Parser,
			"checker" => Self::Checker,
			"reducer" => Self::Reducer,
			_ => unimplemented!(),
		}
	}
}

#[derive(Parser)]
#[command(version)]
pub struct Options {
	/// Runs the compiler in REPL mode.
	#[arg(long,default_value_t=false)]
	repl: bool,

	/// List of compiler stages to show debug output for.
	#[arg(short,long)]
	debug: Vec<Stage>,

	/// Level of debug output to show.
	#[arg(short,long,default_value_t=tracing::Level::INFO)]
	level: tracing::Level,

	/// Outputs with release optimizations.
	#[arg(short,long,default_value_t=false)]
	release: bool,

	#[arg(short,long,default_value_t=String::from("a.out"))]
	output_file: String,

	source_file: String,
}

fn main() -> miette::Result<()> {
	let mut options = Options::parse();
	options.debug.dedup();

	tracing_subscriber::fmt()
		.compact()
		.with_max_level(options.level)
		.without_time()
		.init();

	/*
	let in_file_name = options.source_file;

	let out_file_name = options.output_file;

	let source = std::fs::read_to_string(&in_file_name.trim())
		.into_diagnostic()?;
	*/

	let mut input = String::new();
	if options.repl {
		use std::io::{self, Write};

		println!("ctrl+C to exit");
		loop {
			print!("> ");
			io::stdout().flush().unwrap();
			input.clear();
			let _ = io::stdin().read_line(&mut input);
			let source = input.trim();
			if let Err(e) = AppData::new(source).start() {
				panic!("ERR: {e:?}");
			}
		}
	}

	let source = TEST_INPUT;

	if let Err(e) = AppData::new(source).start() {
		panic!("ERR: {e}");
	}

	info!("lexing");
	let tokens = lexer::eval(source)?;
	if options.debug.contains(&Stage::Lexer) {
		let token_str = tokens.iter()
			.map(|t| t.to_string())
			.collect::<Vec<_>>()
			.join(", ");
		debug!("Tokens: [{token_str}]");
	}

	info!("parsing");
	let mut stepper = parser::Parser::new(source, &tokens);
	loop {
		match stepper.step() {
			Ok(Step::Next(msg)) => println!("[step] {msg}"),
			Err(Error::Report(e)) => eprintln!("[erep] {e}"),
			Err(Error::Fatal(e)) => {
				eprintln!("[exit] {e}");
				break;
			}
			Ok(Step::Done) => {
				println!("[done] Finished");
				break;
			}
		}
	}
	let out = stepper.finish()?;
	if options.debug.contains(&Stage::Parser) {
		debug!("Start ID: {}", out.start);
		for (nx, node) in out.store.iter() {
			debug!("[{nx:>3}]: {node}");
		}
	}

	info!("type-checking");
	let mut out = checker::eval(out);
	if options.debug.contains(&Stage::Checker) {
		debug!("checked AST: {out:?}");
	}

	if options.release {
		info!("reduction");
		let ast = reducer::eval(&mut out.store, out.start);
		if options.debug.contains(&Stage::Reducer) {
			debug!("reduced AST: {ast:?}");
		}
	}

	let output = format!("AST: {out:?}");

	std::fs::write("test.out", output)
//	std::fs::write(&out_file_name, output)
		.into_diagnostic()
}

use ratatui::{
	prelude::{CrosstermBackend, Terminal, Constraint, Direction, Layout, Line, Span, Style, Color, Modifier},
	widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
};
use crossterm::{
	event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
	execute,
	terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

enum AppState {
	Ready,
	Lexing(lexer::Lexer),
	Parsing(Option<Box<parser::Parser>>),
	Done(parser::NodeId, parser::NodeStore),
}

struct AppData {
	state: AppState,
	source: Rc<str>,
	tokens: Vec<tokens::Token>,
	status: String,
}

impl AppData {
	fn new(source: &str) -> Self {
		Self {
			state: AppState::Ready,
			source: source.into(),
			tokens: vec![],
			status: "Ready".into(),
		}
	}

	fn start(self) -> Result<(), std::io::Error> {
		enable_raw_mode()?;
		let mut stdout = std::io::stdout();
		execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
		let backend = CrosstermBackend::new(stdout);
		let mut terminal = Terminal::new(backend)?;

		let res = self.run(&mut terminal);

		disable_raw_mode()?;
		execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
		terminal.show_cursor()?;

		if let Err(err) = res {
			println!("Error: {err:?}");
		}

		Ok(())
	}

	fn run<B: ratatui::backend::Backend>(
		mut self,
		terminal: &mut Terminal<B>,
	) -> std::io::Result<()> {
		loop {
			{
				// Pull out all the App data
				let Self {
					state,
					tokens,
					status,
					..
				} = &self;

				terminal.draw(|f| {
					let chunks = Layout::default()
						.direction(Direction::Vertical)
						.margin(1)
						.constraints([
							Constraint::Percentage(70),
							Constraint::Percentage(30),
						])
						.split(f.area());

					let status = Paragraph::new(status.as_str())
						.block(Block::default().title("Status").borders(Borders::ALL))
						.style(Style::default().fg(Color::Green).add_modifier(Modifier::ITALIC));
					f.render_widget(status, chunks[1]);

					match state {
						AppState::Ready => {
							f.render_widget("SSLang, Compiler Debugger", chunks[0]);
						}

						AppState::Lexing(_) => {
							let token_list = List::new(
								tokens
									.iter()
									.rev()
									.map(|t| ListItem::new(format!("{t}")))
									.collect::<Vec<_>>(),
							)
							.block(Block::default().title("Tokens").borders(Borders::ALL));

							f.render_widget(token_list, chunks[0]);
						}

						AppState::Parsing(None) => {
							f.render_widget("Missing parser state", chunks[0]);
						}
						AppState::Parsing(Some(parser)) => {
							let top = Layout::default()
								.direction(Direction::Horizontal)
								.margin(1)
								.constraints([
									Constraint::Percentage(20),
									Constraint::Percentage(20),
									Constraint::Percentage(20),
									Constraint::Percentage(20),
									Constraint::Percentage(20),
								])
								.split(chunks[0]);

							let (head, _) = parser.input.split_at(parser.index);
							let start = head.last()
								.map(|t| t.range().start)
								.unwrap_or(0);
							let source = Paragraph::new(Line::from(vec![
									Span::styled(&parser.source[..start], Style::new().add_modifier(Modifier::BOLD)),
									Span::raw(&parser.source[start..]),
							]))
								.wrap(Wrap { trim: true })
								.scroll((1, 1))
								.block(Block::default().title("Source").borders(Borders::ALL));
							f.render_widget(source, top[0]);

							let node_data = parser.nodes.iter()
								.map(|(nx,n)| {
									(nx, &n.expr)
								})
								.fold(String::new(), |out,(nx,expr)| {
									format!("{out}\n[{nx:>3}] {expr}")
								});
							let nodes = Paragraph::new(node_data)
								.block(Block::default().title("Nodes").borders(Borders::ALL));
							f.render_widget(nodes, top[1]);

							let scopes = Paragraph::new(format!("{:#?}", parser.scopes))
								.block(Block::default().title("Scopes").borders(Borders::ALL));
							f.render_widget(scopes, top[2]);

							let value_data = parser.values.iter()
								.rev()
								.map(|value| format!("{value:?}"))
								.collect::<Vec<_>>();
							let values = Paragraph::new(value_data.join("\n"))
								.block(Block::default().title("Values").borders(Borders::ALL));
							f.render_widget(values, top[3]);

							let stack_data = parser.stack.iter()
								.rev()
								.map(|op| format!("{op:?}"))
								.collect::<Vec<_>>();
							let stack = Paragraph::new(stack_data.join("\n"))
								.block(Block::default().title("Stack").borders(Borders::ALL));
							f.render_widget(stack, top[4]);

							// stepper.program;
						}

						AppState::Done(start_nx, nodes) => {
							let mut out = vec![];
							parser::nodes_to_string(*start_nx, nodes, 0, &mut out);
							let output = List::new(out)
								.block(Block::default().title("Output").borders(Borders::ALL));
							f.render_widget(output, chunks[0]);
						}
					}
				})?;
			}

			{
				if event::poll(std::time::Duration::from_millis(200))? {
					if let Event::Key(key) = event::read()? {
						if key.kind != event::KeyEventKind::Press {
							continue;
						}

						match key.code {
							KeyCode::Char('q') => return Ok(()),

							// Move to the [n]ext stage
							KeyCode::Char('n') => self.state = match self.state {
								AppState::Ready => {
									let lexer = lexer::Lexer::new(&self.source);
									AppState::Lexing(lexer)
								}
								AppState::Lexing(lexer) => {
									for result in lexer {
										match result {
											Ok(token) => self.tokens.push(token),
											Err(e) => self.status = format!("ERROR: {e:?}"),
										}
									}
									let parser = parser::Parser::new(&self.source, &self.tokens);
									AppState::Parsing(Some(Box::new(parser)))
								}
								AppState::Parsing(mut parser) => {
									if let Some(mut parser) = parser.take() {
										loop {
											match parser.step() {
												Ok(Step::Next(msg)) => self.status = format!("[step] {msg}"),
												Err(Error::Report(e)) => self.status = format!("[erep] {e}"),
												Err(Error::Fatal(e)) => {
													self.status = format!("[exit] {e}");
													break;
												}
												Ok(Step::Done) => {
													self.status = "[done] Finished".into();
													break;
												}
											}
										}

										match parser.finish() {
											Ok(out) => AppState::Done(out.start, out.store),
											Err(e) => {
												self.status = format!("[exit] {e}");
												AppState::Done(0, parser::NodeStore::default())
											}
										}
									} else {
										self.status = "missing Parser state".into();
										AppState::Parsing(None)
									}
								}
								AppState::Done(..) => return Ok(()),
							},

							// [S]tep one item at a time
							KeyCode::Char('s') => match self.state {
								AppState::Ready => {
									let lexer = lexer::Lexer::new(&self.source);
									self.state = AppState::Lexing(lexer);
								}

								AppState::Lexing(mut lexer) => {
									match lexer.next() {
										Some(Ok(token)) => {
											self.tokens.push(token);
											self.state = AppState::Lexing(lexer);
										}
										Some(Err(e)) => {
											self.status = format!("[exit] {e}");
											self.state = AppState::Done(0, parser::NodeStore::default());
										}
										None => {
											let parser = parser::Parser::new(&self.source, &self.tokens);
											self.state = AppState::Parsing(Some(Box::new(parser)));
										}
									}
								}

								AppState::Parsing(ref mut parser) => {
									if let Some(mut parser) = parser.take() {
										match parser.step() {
											Ok(Step::Next(msg)) => {
												self.status = format!("[step] {msg}");
												self.state = AppState::Parsing(Some(parser));
												continue;
											}
											Err(Error::Report(e)) => {
												self.status = format!("[erep] {e}");
												self.state = AppState::Parsing(Some(parser));
												continue;
											}
											Err(Error::Fatal(e)) => self.status = format!("[exit] {e}"),
											Ok(Step::Done) => self.status = "[done] Finished".into(),
										}

										self.state = match parser.finish() {
											Ok(out) => AppState::Done(out.start, out.store),
											Err(e) => {
												self.status = format!("[exit] {e}");
												AppState::Done(0, parser::NodeStore::default())
											}
										};
									} else {
										self.status = "missing Parser state".into();
										self.state = AppState::Parsing(None);
									}
								}

								AppState::Done(..) => return Ok(()),
							}

							_ => {}
						}
					}
				}
			}
		}
	}
}

