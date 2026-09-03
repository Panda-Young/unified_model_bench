"""Detect unescaped parentheses in echo/REM lines that sit *inside* a
parenthesized block in Windows batch files (.bat/.cmd).

Why this matters
----------------
cmd.exe parses a whole parenthesized block (an `if (...)` / `for (...) do (...)`
body) up front. Inside such a block, ANY unescaped `(` or `)` - including the
ones you merely wanted to print - is treated as block syntax:

    if "%FORCE_PUSH%"=="1" (
        echo Pushing custom model (forced)...
    )

The `)` after "forced" closes the block early, the trailing `...` is then parsed
as a command, and cmd dies with `... was unexpected at this time.`
(Chinese locale: `此时不应有 ...。`) - even when the branch is NOT taken,
because the error happens at parse time.

Escaping with `^(` / `^)` (or quoting) fixes it.

Scope (verified empirically against cmd.exe, not guessed)
---------------------------------------------------------
Only `echo` is flagged. Measured behaviour inside a block:

| line                                        | result                |
|---------------------------------------------|-----------------------|
| `REM foo (bar) baz`                         | safe  - comments are stripped before paren parsing |
| `:: foo (bar) baz`                          | safe  - same as REM   |
| `title My Bench (Release)`                  | safe                  |
| `echo "Path (x86)\foo"`                     | safe  - quoted        |
| `echo Pushing ^(forced^)...`                | safe  - escaped       |
| `echo Pushing custom model (forced)...`     | FATAL - `...` parsed as a command |
| `echo Pushing (forced)`                     | FATAL - silently prints `Pushing (forced` (drops the `)`) |

`echo` is the only commonly-used command that both takes free-form prose and
is actually broken by it, so `REM`/`::`/`title` are deliberately NOT reported.

Usage: python tools/utils/check_bat_parens.py [path...]
Exit code: 0 = compliant, 1 = violations found, 2 = scan error (nothing was
actually checked). Mirrors tools/utils/check_braces.py.
"""
import os
import re
import sys

# Resolved relative to this script (tools/utils/ -> repo root), so the check
# works on any machine / checkout path instead of a hardcoded absolute dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
ROOTS = [_REPO_ROOT]
EXTS = (".bat", ".cmd")

# Commands actually broken by an unescaped paren inside a block (see table in
# the module docstring - this list is empirical, not aspirational).
TEXT_CMDS = ("echo", "echo.")

BLOCK_OPEN_RE = re.compile(
    r"^(?:\s*)(?:"
    r"(?:if|for)\b(?!/\*).*"          # if <cond> (   /  for ... (
    r"|\)"                             # plain )  and  ) else (  /  ) && ...
    r"|else\b"
    r")",
    re.IGNORECASE,
)


def logical_lines(text):
    """Join '^' continued lines. Yields (start_line_no, joined_text).

    A trailing '^' is cmd's line-continuation marker, distinct from '^(' which
    merely escapes a parenthesis, so only a caret at end-of-line continues.
    """
    out = []
    buf = ""
    start = 1
    lineno = 0
    for raw in text.splitlines():
        lineno += 1
        if not buf:
            start = lineno
        stripped = raw.rstrip()
        if stripped.endswith("^") and not stripped.endswith("^^"):
            buf += stripped[:-1] + " "
            continue
        buf += raw
        out.append((start, buf))
        buf = ""
    if buf:
        out.append((start, buf))
    return out


def strip_quoted_and_escaped(line):
    """Blank out double-quoted spans and ^X escapes, keeping other chars.

    Returns (masked_line, had_paren_when_unmasked) - the masked line has
    quoted/escaped regions replaced by spaces so remaining parens are the ones
    cmd would see as syntax.
    """
    out = []
    in_q = False
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == "^":
            # ^X escapes the next char (including parenthesis)
            out.append("  ")
            i += 2
            continue
        if c == '"':
            in_q = not in_q
            out.append(c)
            i += 1
            continue
        out.append(" " if in_q else c)
        i += 1
    return "".join(out)


def is_structural(line):
    """True if this line's parens are block syntax rather than text to print."""
    s = line.strip()
    if not s:
        return True
    if s.startswith(")"):
        return True
    if re.match(r"(?i)^(if|for)\b", s):
        return True
    if re.match(r"(?i)^else\b", s):
        return True
    if re.match(r"(?i)^(goto|endlocal|setlocal|exit|shift|call)\b", s):
        # These may legitimately carry a block: `call :sub (arg)` is rare, but
        # treat as structural to avoid false positives.
        return True
    return False


def find_issues(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    issues = []
    depth = 0
    for lineno, line in logical_lines(text):
        masked = strip_quoted_and_escaped(line)
        stripped = masked.strip()

        # Determine the command word (first token) for reporting/eligibility.
        m = re.match(r"(?i)^([a-z]+(?:\.)?)(\s|$|:)", stripped)
        cmd = m.group(1).lower() if m else ""

        if depth > 0 and cmd in TEXT_CMDS:
            body = masked
            if "(" in body or ")" in body:
                issues.append((lineno, cmd, line.strip()[:70]))

        # Update block depth using the structural parens only.
        if is_structural(stripped) or depth > 0:
            opens = masked.count("(")
            closes = masked.count(")")
            depth += opens - closes
        if depth < 0:
            depth = 0
    return issues


def expand_paths(argv_paths):
    paths = []
    errors = []
    for p in argv_paths:
        if os.path.isdir(p):
            for dirpath, _, files in os.walk(p):
                for fn in files:
                    if fn.lower().endswith(EXTS):
                        paths.append(os.path.join(dirpath, fn))
        elif os.path.isfile(p):
            paths.append(p)
        else:
            errors.append("no such file or directory: %s" % p)
    return paths, errors


def main():
    argv_paths = sys.argv[1:] if len(sys.argv) > 1 else []
    arg_errors = []
    if argv_paths:
        paths, arg_errors = expand_paths(argv_paths)
    else:
        paths = []
        for root in ROOTS:
            if not os.path.isdir(root):
                arg_errors.append("default source dir not found: %s" % root)
                continue
            for dirpath, dirnames, files in os.walk(root):
                # Skip VCS/build dirs: their .bat files are not ours to lint.
                dirnames[:] = [
                    d for d in dirnames
                    if d not in (".git", "build", "deps", "__pycache__", ".github")
                ]
                for fn in files:
                    if fn.lower().endswith(EXTS):
                        paths.append(os.path.join(dirpath, fn))

    total = 0
    scanned = 0
    for p in sorted(paths):
        try:
            issues = find_issues(p)
        except Exception as e:
            print("ERROR %s: %s" % (p, e))
            arg_errors.append("could not scan %s: %s" % (p, e))
            continue
        scanned += 1
        if issues:
            print("\n=== %s ===" % p)
            for lineno, cmd, body in issues:
                print("  L%-4d [%s] %s" % (lineno, cmd, body))
            total += len(issues)

    if arg_errors:
        print("\nSCAN INCOMPLETE - %d error(s):" % len(arg_errors))
        for e in arg_errors:
            print("  %s" % e)
        print("\nTOTAL: %d unescaped paren(s) in block (scanned %d file(s))"
              % (total, scanned))
        return 2

    if scanned == 0:
        print("SCAN INCOMPLETE - no .bat/.cmd files found.")
        print("  looked in: %s" % ", ".join(ROOTS if not argv_paths else argv_paths))
        print("\nTOTAL: %d unescaped paren(s) in block (scanned 0 files)" % total)
        return 2

    print("\nTOTAL: %d unescaped paren(s) in block (scanned %d file(s))"
          % (total, scanned))
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
