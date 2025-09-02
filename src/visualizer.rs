
use crate::parser::{Expr, NodeStore};

pub(crate) fn to_mermaid(graph: &NodeStore) -> String {
	format!(r#"
```mermaid
flowchart BT
{}

{}
```
"#,
		nodes(graph),
		edges(graph),
	)
}

fn nodes(graph: &NodeStore) -> String {
	let mut out = vec![];
	for node in graph.iter() {
		match node.expr() {
			Expr::Start => {
				out.push(format!("{}[Start]", node.id()));
			}
			Expr::Return => {
				out.push(format!("{}[Return]", node.id()));
			}
			Expr::Id => {
				out.push(format!("{}[Id_{}]", node.id(), node.name()));
			}
			Expr::Num => {
				out.push(format!("{}([#{}])", node.id(), node.number()));
			}
			Expr::Bool => {
				out.push(format!("{}[{}]", node.id(), node.boolean()));
			}
			Expr::Block => {
				out.push(format!("{}[Region]", node.id()));
			}
			Expr::Rec => todo!("record -> mermaid"),
			Expr::Fun => todo!("function -> mermaid"),
			Expr::Var => unreachable!("no variables in graph"),
			Expr::If => {
				out.push(format!("{}[If]", node.id()));
			}
			Expr::While => {
				out.push(format!("{}[While]", node.id()));
			}
			Expr::RecInit => todo!("record initializer -> mermaid"),
			Expr::Unary => {
				out.push(format!("{}[\"\\{}\"]", node.id(), node.unary_op()));
			}
			Expr::Binary => {
				out.push(format!("{}[\"\\{}\"]", node.id(), node.binary_op()));
			}
			Expr::FnCall => todo!("function call -> mermaid"),
			Expr::Phi => {
				out.push(format!("{}[Phi]", node.id()));
			}
		}
	}
	out.join("\n")
}

fn edges(graph: &NodeStore) -> String {
	let mut out = vec![];
	for node in graph.iter() {
		for nx in node.inputs() {
			let in_node = node.input(*nx);
			let arrow = match (in_node.expr(), node.expr()) {
				(Expr::Start, Expr::Num) => "~~~",
				(Expr::Start, _) | (_, Expr::Start) |
				(Expr::Block, _) | (_, Expr::Block) |
				(Expr::Return, _) | (_, Expr::Return) => "==>",
				_ => "-->",
			};
			out.push(format!("{}{}{}", node.id(), arrow, in_node.id()));
		}
	}
	out.join("\n")
}

