
use std::collections::HashSet;
use std::rc::Rc;

use crate::parser::{
	BinaryOp,
	Expr,
	Int,
	Node,
	NodeId,
	NodeStore,
	ScopeTracker,
	UnaryOp,
	ValueType,
};

pub(crate) fn eval(mut data: crate::parser::Output) -> crate::parser::Output {
	update_types(data.start, &data.records, &data.functions, &data.scopes, &mut data.store, 0);
	data
}

fn update_types(
	nx: NodeId,
	r_store: &HashSet<Rc<str>>,
	f_store: &HashSet<Rc<str>>,
	scopes: &ScopeTracker,
	n_store: &mut NodeStore,
	indent: usize,
) {
	let Node { kind, expr, ..} = n_store.get(nx)
		.cloned()
		.expect("missing node in node_store");

	eprintln!("[{nx:>3}]{:>1$}({kind}): {expr}", ' ', indent);
	eprintln!("SCOPE: {scopes:?}");

	match expr {
		Expr::Num(_) | Expr::Id(_) | Expr::Bool(_) => {}

		Expr::Phi { lhs, rhs } => {
			update_types(lhs, r_store, f_store, scopes, n_store, indent + 2);
			update_types(rhs, r_store, f_store, scopes, n_store, indent + 2);
			let lkind = n_store.get(lhs)
				.map(|n| n.kind.clone())
				.expect("missing left-node-kind in node_store");
			let rkind = n_store.get(rhs)
				.map(|n| n.kind.clone())
				.expect("missing right-node-kind in node_store");
			if lkind != rkind {
				eprintln!("- phi branches no longer match: Left {lkind}, Right {rkind}");
			}
		}

		Expr::Block { body, scope } => {
			let mut block_scopes = scopes.clone();
			block_scopes.add(scope);
			for bx in body {
				update_types(bx, r_store, f_store, &block_scopes, n_store, indent + 2);
			}
		}

		Expr::While { cond, body } => {
			update_types(cond, r_store, f_store, scopes, n_store, indent + 2);
			update_types(body, r_store, f_store, scopes, n_store, indent + 2);
		}

		Expr::RecInit { name, field_inits, ..} => {
			if !r_store.contains(&name) {
				eprintln!("- unknown Record type '{name}'");
				return;
			};

			let _def = scopes.find(&name)
				.unwrap_or_else(|| panic!("- missing Record '{name}' in scopes\n\n{scopes:?}"));

			for (f_name, fnx) in field_inits {
				update_types(fnx, r_store, f_store, scopes, n_store, indent + 2);

				let fx = scopes.find(&f_name)
					.unwrap_or_else(|| panic!("- missing '{f_name}' in scopes\n\n{scopes:?}"));
				let rec = n_store.get(fx).cloned()
					.unwrap_or_else(|_| panic!("- missing index {fx} in node_store"));
				let Expr::Rec {..} = rec.expr else {
					eprintln!("- expected a Record for '{f_name}' identifier, found {rec:?}");
					return;
				};

				let fkind = n_store.get_mut(fnx)
					.map(|n| &mut n.kind)
					.unwrap_or_else(|_| panic!("missing field-kind for '{f_name}' in node_store"));

				if *fkind == ValueType::Any {
					*fkind = rec.kind.clone();
				}

				if *fkind != rec.kind {
					eprintln!("- '{f_name}' in Record Initializer has the wrong type. Expected {}, found {fkind}", rec.kind);
					return;
				}
			}
		}

		// TODO - srenshaw - Add initializers to Variable declarations.
		Expr::Var { name: _, body } => {
			// TODO - srenshaw - Ensure initializer and declaration match.
			if let Some(bx) = body {
				update_types(bx, r_store, f_store, scopes, n_store, indent + 2);
			}
		}

		Expr::If { cond, bt, bf } => {
			update_types(cond, r_store, f_store, scopes, n_store, indent + 2);

			update_types(bt, r_store, f_store, scopes, n_store, indent + 2);

			let tnode = n_store.get(bt)
				.expect("missing true-node in node_store")
				.clone();

			let Expr::Block { body: tnodes, ..} = tnode.expr else {
				panic!("expected block-type for true-node in IF branch");
			};

			if let Some(bf) = bf {
				update_types(bf, r_store, f_store, scopes, n_store, indent + 2);

				let fnode = n_store.get(bf)
					.expect("missing false-node in node_store")
					.clone();

				let Expr::Block { body: fnodes, ..} = fnode.expr else {
					panic!("expected block-type for false-node in IF branch");
				};

				assert_eq!(tnodes.len(), fnodes.len());

				assert_eq!(tnode.kind, fnode.kind, "- IF branches have different types: {} != {}", tnode.kind, fnode.kind);
			} else {
				assert_eq!(tnode.kind, ValueType::Unit, "THEN branch has a non-unit value-type");
			}
		}

		Expr::FnCall { name, args } => {
			if !f_store.contains(&name) {
				eprintln!("- unknown Record type '{name}'");
				return;
			}

			let cx = scopes.find(&name)
				.unwrap_or_else(|| panic!("- missing '{name}' in scopes"));
			let call = n_store.get(cx).cloned()
				.unwrap_or_else(|_| panic!("- missing index {cx} in node_store"));
			let Expr::Fun { params, rtype, ..} = &call.expr else {
				return;
			};

			for (idx, anx) in args.into_iter().enumerate() {
				update_types(anx, r_store, f_store, scopes, n_store, indent + 2);
				let akind = n_store.get(anx)
					.map(|n| n.kind.clone())
					.expect("missing arg-kind in node_store");

				if idx >= params.len() {
					eprintln!("- extra argument to call. Found argument at index {idx}, but there are only {} parameters.", params.len());
					return;
				}

				let px = params[idx];
				let param = n_store.get(px)
					.unwrap_or_else(|_| panic!("- missing index {px} in node_store"));
				let Expr::Id(pname) = &param.expr else {
					return;
				};

				if akind != param.kind {
					eprintln!("- '{pname}' in Function Call has the wrong type. Expected {}, found {akind}",
						param.kind,
					);
					return;
				}
			}

			if kind == ValueType::Any {
				if let Ok(node) = n_store.get_mut(nx) {
					node.kind = rtype.clone();
				}
			}
		}

		Expr::Rec { name, fields } => {
			assert!(r_store.contains(&name), "Compiler Error: Found unknown Record Definition");

			for fx in fields {
				let field = n_store.get(fx)
					.unwrap_or_else(|_| panic!("- missing index {fx} in node_store"));
				if let ValueType::Udt(fudt) = &field.kind {
					if !r_store.contains(fudt) {
						eprintln!("- unknown type '{fudt}' in Record Definition");
					}
				}
			}
		}

		Expr::Fun { name, params, rtype, body } => {
			assert!(f_store.contains(&name), "Compiler Error: Found unknown Function Definition");

			for px in params {
				let param = n_store.get(px)
					.unwrap_or_else(|_| panic!("- missing index {px} in node_store"));
				if let ValueType::Udt(pudt) = &param.kind {
					if !r_store.contains(pudt) {
						eprintln!("- unknown type '{pudt}' in Function Definition");
					}
				}
			}

			update_types(body, r_store, f_store, scopes, n_store, indent + 2);
			let bnode = n_store.get(body)
				.expect("missing body-node in node_store");

			if rtype != bnode.kind {
				eprintln!("- function body returns the wrong type: Expected {rtype}, found {}", bnode.kind);
			}
		}

		Expr::Unary { op, rhs } => {
			update_types(rhs, r_store, f_store, scopes, n_store, indent + 2);
			let Node { kind: rkind, ..} = n_store.get(rhs)
				.cloned()
				.expect("missing right-node-kind in node_store");

			match op {
				// TODO - srenshaw - This should only be applicable to 'pointer' types.
				UnaryOp::Deref => {
					eprintln!("- cannot dereference {rkind}");
				}
				UnaryOp::Neg => {
					if !matches!(rkind, ValueType::Int(Int::Signed(_)) | ValueType::Fix(_)) {
						eprintln!("- cannot negate {rkind}");
					}

					if let Ok(node) = n_store.get_mut(nx) {
						node.kind = rkind.clone();
					}
				}
				UnaryOp::Not => {
					if !matches!(rkind, ValueType::Int(_) | ValueType::Bool) {
						eprintln!("- cannot invert {rkind}");
					}

					if let Ok(node) = n_store.get_mut(nx) {
						node.kind = rkind.clone();
					}
				}
				UnaryOp::Pos => {
					if !matches!(rkind, ValueType::Int(_) | ValueType::Fix(_)) {
						eprintln!("- unary '+' does not apply to {kind}");
					}

					if let Ok(node) = n_store.get_mut(nx) {
						node.kind = rkind.clone();
					}
				}
				// TODO - srenshaw - Implement reference operator.
				UnaryOp::Ref => {
					eprintln!("- implement reference operator");
				}
			}
		}

		Expr::Binary { op, lhs, rhs } => {
			update_types(lhs, r_store, f_store, scopes, n_store, indent + 2);
			let Node { kind: lkind, expr: lexpr, ..} = n_store.get(lhs)
				.cloned()
				.expect("missing left-node-kind in node_store");

			update_types(rhs, r_store, f_store, scopes, n_store, indent + 2);
			let Node { kind: rkind, expr: rexpr, ..} = n_store.get(rhs)
				.cloned()
				.expect("missing right-node-kind in node_store");

			match op {
				BinaryOp::Add => {
					match (&lkind, &rkind) {
						// TODO - srenshaw - Would it be useful to add auto-casting between ints and
						// fixed-points?
						(ValueType::Int(_), ValueType::Int(_)) => {}
						(ValueType::Fix(_), ValueType::Fix(_)) => {}
						_ => {
							eprintln!("- cannot ADD {lkind} and {rkind}");
						}
					}
				}

				BinaryOp::Accessor => {
					match (lexpr, rexpr) {
						(Expr::Id(lname), Expr::Id(rname)) => {
							let Some(rec) = r_store.get(&lname) else {
								eprintln!("- unknown Record type '{lname}'");
								return;
							};
							
							if !rec.contains(&*rname) {
								eprintln!("- Record type '{lname}' has no field named '{rname}'");
							}
						}
						(Expr::Id(_), _) => {
							eprintln!("- expected Identifier on right-hand side of {op} operator");
						}
						_ => {
							eprintln!("- expected Identifier on left-hand side of {op} operator");
						}
					}
				}

				/*
				BinaryOp::AndB => {}
				BinaryOp::AndL => {}
				BinaryOp::Assign => {}
				BinaryOp::CmpEq => {}
				BinaryOp::CmpGE => {}
				BinaryOp::CmpGT => {}
				BinaryOp::CmpLE => {}
				BinaryOp::CmpLT => {}
				BinaryOp::CmpNE => {}
				BinaryOp::Comma => {}
				BinaryOp::Div => {}
				BinaryOp::DivMod => {}
				BinaryOp::LRot => {}
				BinaryOp::LShift => {}
				BinaryOp::Mod => {}
				*/
				BinaryOp::Mul => {
					match (&lkind, &rkind) {
						// TODO - srenshaw - Would it be useful to add auto-casting between ints and
						// fixed-points?
						(ValueType::Int(_), ValueType::Int(_)) => {}
						(ValueType::Fix(_), ValueType::Fix(_)) => {}
						_ => {
							eprintln!("- cannot MUL {lkind} and {rkind}");
						}
					}
				}
				/*
				BinaryOp::OrB => {}
				BinaryOp::OrL => {}
				BinaryOp::RRot => {}
				BinaryOp::RShift => {}
				BinaryOp::Sub => {}
				BinaryOp::XorB => {}
				BinaryOp::XorL => {}
				*/
				_ => eprintln!("- implement type-checking for {op} operator"),
			}
		}
	};
}

