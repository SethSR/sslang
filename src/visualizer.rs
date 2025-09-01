
use crate::parser::{Expr, NodeRef};

pub(crate) fn to_mermaid(node: &NodeRef) -> String {
	return format!("\n```mermaid\nflowchart TD\n{}\n```\n",
		nodes(node),
		// edges(node),
	);

	fn nodes(node: &NodeRef) -> String {
		let mut out = vec![];
		match node.expr() {
			Expr::Id => {
				out.push(format!("{}{{ Id_{} }}", node.id(), node.name()));
			}
			Expr::Num => {
				out.push(format!("{}[#{}]", node.id(), node.number()));
			}
			Expr::Bool => {
				out.push(format!("{}[{}]", node.id(), node.boolean()));
			}
			Expr::Block => {
				out.push(format!("{}[Region]", node.id()));
				for input in node.inputs() {
					let in_node = node.store.get(*input);
					out.push(nodes(&in_node));
					out.push(format!("{}-->{}", in_node.id(), node.id()));
				}
			}
			Expr::Rec => todo!("record -> mermaid"),
			Expr::Fun => todo!("function -> mermaid"),
			Expr::Var => unreachable!("no variables in graph"),
			Expr::If => {
				out.push(format!("{}[If]", node.id()));
				for input in node.inputs() {
					out.push(nodes(&node.store.get(*input)));
				}
			}
			Expr::While => {
				out.push(format!("{}[While]", node.id()));
				for input in node.inputs() {
					out.push(nodes(&node.store.get(*input)));
				}
			}
			Expr::RecInit => todo!("record initializer -> mermaid"),
			Expr::Unary => {
				out.push(format!("{}[\"\\{}\"]", node.id(), node.unary_op()));
				for input in node.inputs() {
					let in_node = node.store.get(*input);
					out.push(nodes(&in_node));
					out.push(format!("{}-->{}", in_node.id(), node.id()));
				}
			}
			Expr::Binary => {
				out.push(format!("{}[\"\\{}\"]", node.id(), node.binary_op()));
				for input in node.inputs() {
					let in_node = node.store.get(*input);
					out.push(nodes(&in_node));
					out.push(format!("{}-->{}", in_node.id(), node.id()));
				}
			}
			Expr::FnCall => todo!("function call -> mermaid"),
			Expr::Phi => {
				out.push(format!("{}[Phi]", node.id()));
				for input in node.inputs() {
					let in_node = node.store.get(*input);
					out.push(nodes(&in_node));
					out.push(format!("{}-->{}", in_node.id(), node.id()));
				}
			}
		}
		out.join("\n")
	}
}

