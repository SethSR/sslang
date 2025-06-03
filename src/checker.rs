
use crate::parser::{
	BinaryOp,
	Expr,
	FuncStore,
	Int,
	Node,
	NodeId,
	NodeStore,
	RecStore,
	UnaryOp,
	ValueType,
};

pub(crate) fn eval(mut data: crate::parser::Output) -> crate::parser::Output {
	update_types(data.start, &data.records, &data.functions, &mut data.store);
	data
}

fn update_types(
	nx: NodeId,
	r_store: &RecStore,
	f_store: &FuncStore,
	n_store: &mut NodeStore,
) {
	let Node { kind, expr, ..} = n_store.get(nx)
		.cloned()
		.expect("missing node in node_store");

	println!("[{nx:3}] ({kind}): {expr}");

	match expr {
		Expr::Num(_) => return,
		Expr::Id(_) => return,
		Expr::Bool(_) => return,
		Expr::Phi { lhs, rhs } => {
			update_types(lhs, r_store, f_store, n_store);
			update_types(rhs, r_store, f_store, n_store);
			let lkind = n_store.get(lhs)
				.map(|n| n.kind.clone())
				.expect("missing lnode in node_store");
			let rkind = n_store.get(rhs)
				.map(|n| n.kind.clone())
				.expect("missing rnode in node_store");
			if lkind != rkind {
				eprintln!("- phi branches no longer match: Left {lkind}, Right {rkind}");
			}
		}

		Expr::Block { body } => {
			for bx in body {
				update_types(bx, r_store, f_store, n_store);
			}
		}

		Expr::While { cond, body } => {
			update_types(cond, r_store, f_store, n_store);
			update_types(body, r_store, f_store, n_store);
		}

		Expr::RecInit { name, field_inits, ..} => {
			let Some(rec) = r_store.get(&name) else {
				eprintln!("- unknown Record type '{name}'");
				return;
			};

			for (f_name, fnx) in field_inits {
				update_types(fnx, r_store, f_store, n_store);
				let fkind = n_store.get_mut(fnx)
					.map(|n| &mut n.kind)
					.expect("missing fnode in node_store");

				let Some(expected_kind) = rec.get(&f_name) else {
					eprintln!("- unknown Field '{f_name}' in Record Initializer");
					return;
				};

				if *fkind == ValueType::Any {
					*fkind = expected_kind.clone();
				}

				if *fkind != *expected_kind {
					eprintln!("- '{f_name}' in Record Initializer has the wrong type. Expected {expected_kind}, found {fkind}");
					return;
				}
			}
		}

		// TODO - srenshaw - Add initializers to Variable declarations.
		Expr::Var { name: _, body } => {
			// TODO - srenshaw - Ensure initializer and declaration match.
			update_types(body, r_store, f_store, n_store);
		}

		Expr::If { cond, bt, bf } => {
			update_types(cond, r_store, f_store, n_store);

			update_types(bt, r_store, f_store, n_store);
			let tkind = n_store.get(bt)
				.map(|n| n.kind.clone())
				.expect("missing tnode in node_store");

			if let Some(bf) = bf {
				update_types(bf, r_store, f_store, n_store);
				let fkind = n_store.get(bf)
					.map(|n| n.kind.clone())
					.expect("missing fnode in node_store");

				if tkind != fkind {
					eprintln!("- if branches have differing types: {tkind} != {fkind}");
					return;
				}
			} else if tkind != ValueType::Unit {
				eprintln!("- missing else-branch, then-branch returns type {tkind}");
				return;
			}
		}

		Expr::FnCall { name, args } => {
			let Some((params, rtype)) = f_store.get(&name) else {
				eprintln!("- unknown Record type '{name}'");
				return;
			};

			for (idx, anx) in args.into_iter().enumerate() {
				update_types(anx, r_store, f_store, n_store);
				let akind = n_store.get(anx)
					.map(|n| n.kind.clone())
					.expect("missing anode in node_store");

				if idx >= params.len() {
					eprintln!("- extra argument to call. Found argument at index {idx}, but there are only {} parameters.", params.len());
					return;
				}

				let (pname, expected_kind) = &params[idx];

				if akind != *expected_kind {
					eprintln!("- '{pname}' in Function Call has the wrong type. Expected {expected_kind}, found {akind}");
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
			assert!(r_store.contains_key(&name), "Compiler Error: Found unknown Record Definition");

			for (_, fkind) in fields {
				if let ValueType::UDT(fudt) = fkind {
					if !r_store.contains_key(&fudt) {
						eprintln!("- unknown type '{fudt}' in Record Definition");
					}
				}
			}
		}

		Expr::Fun { name, params, rtype, body } => {
			assert!(f_store.contains_key(&name), "Compiler Error: Found unknown Function Definition");

			for (_, pkind) in params {
				if let ValueType::UDT(pudt) = pkind {
					if !r_store.contains_key(&pudt) {
						eprintln!("- unknown type '{pudt}' in Function Definition");
					}
				}
			}

			update_types(body, r_store, f_store, n_store);
			let bkind = n_store.get(body)
				.map(|n| n.kind.clone())
				.expect("missing bnode in node_store");

			if rtype != bkind {
				eprintln!("- function body returns the wrong type: Expected {rtype}, found {bkind}");
			}
		}

		Expr::Unary { op, rhs } => {
			update_types(rhs, r_store, f_store, n_store);
			let Node { kind: rkind, ..} = n_store.get(rhs)
				.cloned()
				.expect("missing rnode in node_store");

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
			update_types(lhs, r_store, f_store, n_store);
			let Node { kind: lkind, expr: lexpr, ..} = n_store.get(lhs)
				.cloned()
				.expect("missing lnode in node_store");

			update_types(rhs, r_store, f_store, n_store);
			let Node { kind: rkind, expr: rexpr, ..} = n_store.get(rhs)
				.cloned()
				.expect("missing rnode in node_store");

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
							
							if !rec.contains_key(&rname) {
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

