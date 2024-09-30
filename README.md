# Feature Overview

- Simplified Rust-like syntax.
- No semi-colons (may add later for clarity).
- Syntax for variable, record, and function declarations (`var`, `rec`, `fn`).
- Syntax for `while` loops.
- Intrinsic (built-in) types specific to the Saturn.

### Intrinsic Types

- `u8`, `u16`, `u32` -- unsigned 1, 2, and 4-byte integers.
- `s8`, `s16`, `s32` -- signed 1, 2, and 4-byte integers.
- `fw`, `fl` -- signed 2 and 4-byte fixed-point decimals. The number of bits used for the integer value can be specified with an integer at the end of the type (ex: `fw4` -- 4 integer bits, 12 decimal-bits; `fl23` -- 23 integer bits, 9 decimal bits).

# Planned Features
- Syntactic sugar for `for` loops.
- Compiler declaratives for targeting specific chips in the Saturn in the same code-base.
- Generics / Higher-kinded types (potentially).
