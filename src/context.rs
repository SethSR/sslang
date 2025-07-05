
pub(crate) struct Context {
	frames: Vec<String>,
}

impl Context {
	pub fn push(&mut self, s: &str) {
		self.frames.push(s);
	}
}
