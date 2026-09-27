#!/usr/bin/env python3
"""Check that two source trees differ only in comments and docstrings.

Two static comparisons must pass:

  TOKEN  Tokenises both versions and compares the token stream with
         comments and string statements dropped. Catches any change to
         names, literals, or operators.

  AST    Parses both versions and compares the syntax tree with string
         statements dropped. Independent of formatting.

An optional JSON rename map lets the same check cover a mechanical
rename. The map rewrites the old version's names and string literals
before the comparison, so a rename gets verified the same way as a
comment edit.

This check excludes docstring introspection and does not establish runtime
equivalence for arbitrary renames.

Usage:
    python3 scripts/check_equivalence.py BEFORE AFTER [--map renames.json]
"""

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path


def python_files(root):
    """Sorted .py paths under root, skipping caches and virtualenvs."""
    skip = {"__pycache__", ".venv", "node_modules", ".ipynb_checkpoints"}
    out = []
    for path in sorted(Path(root).rglob("*.py")):
        if any(part in skip for part in path.parts):
            continue
        out.append(path.relative_to(root))
    return out


def read(root, rel):
    path = Path(root) / rel
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def compile_map(mapping):
    """Whole-word substitutions. Keys are identifiers, so word
    boundaries keep ARBITER_DIAG out of ARBITER_DIAG_PATH."""
    return [(re.compile(r"\b%s\b" % re.escape(old)), new)
            for old, new in sorted(mapping.items())]


def compile_literals(pairs):
    """Plain substitutions for text inside string literals. Longest
    first, so 'arbiter_rl save failed' is not caught by 'arbiter_rl '."""
    return [(old, new) for old, new in
            sorted(pairs.items(), key=lambda kv: -len(kv[0]))]


def rename(text, rules, literals=()):
    for pattern, new in rules:
        text = pattern.sub(new, text)
    for old, new in literals:
        text = text.replace(old, new)
    return text


def text_spans(source):
    """(lineno, col_offset) starts of every non-semantic string literal.

    Includes docstrings and bare string statements. Docstrings may still
    affect introspection or CLI help when code reads __doc__.
    """
    spans = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return spans
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        for statement in body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                if isinstance(statement.value.value, str):
                    value = statement.value
                    spans.add((value.lineno, value.col_offset))
    return spans


def token_fingerprint(source, rules=(), literals=()):
    """Token sequence with comments and string statements dropped.

    Both are non-semantic, so their count may differ between versions.
    Every other string literal, including ones used in logic, stays in
    the fingerprint and is compared exactly.
    """
    spans = text_spans(source)
    fingerprint = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                        tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        if tok.type == tokenize.STRING and (tok.start[0], tok.start[1]) in spans:
            continue
        # Apply the map to every token, not just names and whole strings.
        # F-string middles carry text too and would otherwise slip through.
        fingerprint.append((tok.type, rename(tok.string, rules, literals)))
    return fingerprint


def strip_text_statements(tree):
    """Drop docstrings and orphan string statements from the tree."""
    for node in ast.walk(tree):
        for attr in ("body", "orelse", "finalbody"):
            block = getattr(node, attr, None)
            if not isinstance(block, list):
                continue
            kept = []
            for statement in block:
                value = statement.value if isinstance(statement, ast.Expr) else None
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    continue
                kept.append(statement)
            if len(kept) != len(block):
                block[:] = kept
    return tree


def ast_fingerprint(source, rules=(), literals=()):
    tree = strip_text_statements(ast.parse(source))
    return rename(ast.dump(tree, annotate_fields=True, include_attributes=False),
                  rules, literals)


def compare(before, after, rules=(), literals=()):
    """Failure messages, empty when the two versions agree."""
    problems = []
    try:
        tok_a = token_fingerprint(before, rules, literals)
        tok_b = token_fingerprint(after)
    except (tokenize.TokenError, IndentationError, SyntaxError) as err:
        return ["tokenising failed: %s" % err]
    if tok_a != tok_b:
        for i, (a, b) in enumerate(zip(tok_a, tok_b)):
            if a != b:
                problems.append("token %d differs: %r -> %r" % (i, a, b))
                break
        else:
            problems.append("token counts differ: %d -> %d"
                            % (len(tok_a), len(tok_b)))

    try:
        ast_a = ast_fingerprint(before, rules, literals)
        ast_b = ast_fingerprint(after)
    except SyntaxError as err:
        return problems + ["parsing failed: %s" % err]
    if ast_a != ast_b:
        problems.append("syntax trees differ")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--map", dest="map_path",
                    help="JSON file mapping old names to new names")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    mapping, literals = {}, {}
    if args.map_path:
        raw = json.loads(Path(args.map_path).read_text(encoding="utf-8"))
        if "names" in raw or "literals" in raw:
            mapping, literals = raw.get("names", {}), raw.get("literals", {})
        else:
            mapping = raw
    rules = compile_map(mapping)
    literal_rules = compile_literals(literals)

    before_root, after_root = Path(args.before), Path(args.after)
    names = sorted(set(python_files(before_root)) | set(python_files(after_root)))

    changed, failures = 0, []
    for rel in names:
        raw_before = read(before_root, rel)
        raw_after = read(after_root, rel)
        if raw_before is None or raw_after is None:
            which = "after" if raw_before is None else "before"
            failures.append("%s: missing from the %s tree" % (rel, which))
            continue
        if rename(raw_before, rules, literal_rules) == raw_after:
            continue
        changed += 1
        for problem in compare(raw_before, raw_after, rules, literal_rules):
            failures.append("%s: %s" % (rel, problem))

    if not args.quiet:
        print("files compared : %d" % len(names))
        print("files changed  : %d" % changed)
        if mapping or literals:
            print("rename entries : %d names, %d literals"
                  % (len(mapping), len(literals)))
    for failure in failures:
        print("FAIL %s" % failure)
    if failures:
        print("\n%d file(s) differ in more than comments and docstrings."
              % len({f.split(":")[0] for f in failures}))
        return 1
    print("\nPASS: every difference is a comment or a docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
