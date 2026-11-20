//! Compile-fail tests for the proc macros.
//!
//! The fixtures live in `tests/ui/`; `trybuild` compiles each one and compares
//! the diagnostic with the committed `.stderr` snapshot. Run with
//! `TRYBUILD=overwrite cargo test -p oxide-ml --test compile_fail` after an
//! intentional diagnostic change.

#[test]
fn invalid_macro_usage_is_rejected() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("../../tests/ui/*.rs");
}
