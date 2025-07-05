
#[derive(Debug, Default)]
pub(crate) struct Data {
	frames: Vec<String>,
}

pub(crate) struct Guard<'a> {
	ctx: &'a mut Data,
}

impl<'a> Guard<'a> {
	pub fn new(ctx: &'a mut Data, s: impl Into<String>) -> Self {
		ctx.frames.push(s.into());
		Self { ctx }
	}
}

impl Drop for Guard<'_> {
	fn drop(&mut self) {
		self.ctx.frames.pop();
	}
}
