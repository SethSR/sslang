
use std::rc::Rc;

use miette::IntoDiagnostic;
use slotmap::Key;
use tracing::{info,debug};

mod tokens;
mod lexer;
mod parser;
mod visualizer;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
	Lexer,
	Parser,
}

impl From<String> for Stage {
	fn from(s: String) -> Self {
		match s.to_lowercase().as_str() {
			"lexer" => Self::Lexer,
			"parser" => Self::Parser,
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

	let in_file_name = options.source_file;

	let out_file_name = options.output_file;

	let source = std::fs::read_to_string(&in_file_name.trim())
		.into_diagnostic()?;

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

	// let source = TEST_INPUT;

	if let Err(e) = AppData::new(&source).start() {
		panic!("ERR: {e}");
	}

	info!("lexing");
	let tokens = lexer::eval(&source)?;
	if options.debug.contains(&Stage::Lexer) {
		let token_str = tokens.iter()
			.map(|t| t.to_string())
			.collect::<Vec<_>>()
			.join(", ");
		debug!("Tokens: [{token_str}]");
	}

	info!("parsing");
	let mut stepper = parser::Parser::new(&source, &tokens);
	while stepper.step_and_continue(true) {}
	let out = stepper.finish()?;
	if options.debug.contains(&Stage::Parser) {
		debug!("Start ID: {:?}", out.start);
		for node in out.store.iter() {
			debug!("{node:?}");
		}
	}

	// let output = format!("AST: {out:?}");

	let output = visualizer::to_mermaid(&out.store);
	let out_file_name = format!("{}.md", out_file_name);

	// std::fs::write("test.out", output)
	std::fs::write(&out_file_name, output)
		.into_diagnostic()
}

use ratatui::{
	prelude::{CrosstermBackend, Terminal, Constraint, Direction, Layout, Style, Color, Modifier},
	widgets::{Block, Borders, List, ListItem, Paragraph},
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
									.map(|t| ListItem::new(t.to_string()))
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

							let tokens = &parser.input[parser.index..];
							let source = List::new(tokens.iter().map(|t| ListItem::new(t.to_string())).collect::<Vec<_>>())
								.block(Block::default().title("Tokens").borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM));
							f.render_widget(source, top[0]);

							let node_data = parser.nodes.iter()
								.fold(String::new(), |out,n| {
									format!("{out}\n{n:?}")
								});
							let nodes = Paragraph::new(node_data)
								.block(Block::default().title("Nodes").borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM));
							f.render_widget(nodes, top[1]);

							let scopes = Paragraph::new(format!("{:#?}", parser.scopes))
								.block(Block::default().title("Scopes").borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM));
							f.render_widget(scopes, top[2]);

							let value_data = parser.values.iter()
								.rev()
								.map(|value| format!("{value:?}"))
								.collect::<Vec<_>>();
							let values = Paragraph::new(value_data.join("\n"))
								.block(Block::default().title("Values").borders(Borders::TOP | Borders::LEFT | Borders::BOTTOM));
							f.render_widget(values, top[3]);

							let stack_data = parser.stack.iter()
								.rev()
								.map(|op| format!("{op:?}"))
								.collect::<Vec<_>>();
							let stack = Paragraph::new(stack_data.join("\n"))
								.block(Block::default().title("Stack").borders(Borders::ALL));
							f.render_widget(stack, top[4]);
						}

						AppState::Done(start_nx, nodes) => {
							let mut out = vec![];
							nodes.nodes_to_string(*start_nx, 0, &mut out);
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
										while parser.step_with_action(
											|m,s| { *s = m; true },
											|m,s| { *s = m; false },
											|m,s| { *s = m; true },
											&mut self.status,
										) {}

										match parser.finish() {
											Ok(out) => AppState::Done(out.start, out.store),
											Err(e) => {
												self.status = format!("[exit] {e}");
												AppState::Done(parser::NodeId::null(), parser::NodeStore::default())
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
											self.state = AppState::Done(parser::NodeId::null(), parser::NodeStore::default());
										}
										None => {
											let parser = parser::Parser::new(&self.source, &self.tokens);
											self.state = AppState::Parsing(Some(Box::new(parser)));
										}
									}
								}

								AppState::Parsing(ref mut parser) => {
									if let Some(mut parser) = parser.take() {
										if parser.step_with_action(
											|m,out| { *out = m; true },
											|m,out| { *out = m; false },
											|m,out| { *out = m; true },
											&mut self.status,
										) {
											self.state = AppState::Parsing(Some(parser));
											continue
										}

										self.state = match parser.finish() {
											Ok(out) => AppState::Done(out.start, out.store),
											Err(e) => {
												self.status = format!("[exit] {e}");
												AppState::Done(parser::NodeId::null(), parser::NodeStore::default())
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

