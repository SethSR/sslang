
# Task List

- FIXME - Update expression parsing to use the stack-parser. Currently, due to an issue with identifier parsing, we fall into an infinite loop when trying to parse assignment statements.
- Clean up `eprintln` and `panic` macros in 'node.rs'. We should generate actual, useful error messages whenever possible, and strive to make it possible more often.
- Clean up debugging `println` macros throughout 'mod.rs', 'node.rs', and 'parser.rs'.
- Look into removing the block node type. I think it may be possible to just use the last node as the "block", as the last node will reference any others that are needed, and anything else should be elligible for dead-code-elimination (when we get around to that).
- Move `Parser::nodes`, `Parser::records`, and `Parser::functions` into `ScopeTracker`, since they're all scope related.
- Modify `StackOp::Ident` to save unknown Identifiers, so we can handle out-of-order definitions, instead of just throwing an error.

