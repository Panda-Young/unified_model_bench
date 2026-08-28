"""Detect control statements (if/else/for/while/do/switch/case) whose body
lacks braces, in the project's own source files (src/, include/).
Recursive scanner: descends into braced blocks, so nested statements are found.

Usage: python tools/utils/check_braces.py [path...]
Output: file:line [keyword] body ; TOTAL: N  (0 = compliant)

Exit code: 0 = compliant, 1 = violations found, 2 = scan error (nothing was
actually checked). A scan error is reported loudly instead of printing a
misleading "TOTAL: 0": scanning zero files used to look identical to a clean
codebase (the ROOTS below once pointed at a stale absolute path).
"""
import os
import sys

# Resolved relative to this script (tools/utils/ -> repo root), so the check
# works on any machine / checkout path instead of a hardcoded absolute dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
ROOTS = [os.path.join(_REPO_ROOT, "src"), os.path.join(_REPO_ROOT, "include")]
EXTS = (".cpp", ".hpp", ".h", ".cc", ".cxx")


def sanitize(text):
    """Replace comments and string/char literals with spaces (keep newlines & braces)."""
    out = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == "/" and nxt == "*":
            while i < n:
                if text[i] == "\n":
                    out.append("\n")
                else:
                    out.append(" ")
                if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    out.append("  ")
                    i += 2
                    break
                i += 1
            continue
        if c in ('"', "'"):
            out.append(c)
            i += 1
            while i < n:
                if text[i] == "\\":
                    out.append("  ")
                    i += 2
                    continue
                out.append(text[i])
                if text[i] == c:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


class Scanner:
    def __init__(self, code):
        self.code = code
        self.n = len(code)
        self.line = [1] * (len(code) + 1)
        ln = 1
        for i, ch in enumerate(code):
            self.line[i] = ln
            if ch == "\n":
                ln += 1
        self.issues = []

    def skip_ws(self, i):
        while i < self.n and self.code[i] in " \t\r\n":
            i += 1
        return i

    def adv(self, i, nxt):
        return nxt if nxt > i else i + 1

    def peek_kw(self, i, kw):
        if self.code.startswith(kw, i):
            j = i + len(kw)
            if j >= self.n or not (self.code[j].isalnum() or self.code[j] == "_"):
                return True
        return False

    def skip_parens(self, i):
        depth = 0
        while i < self.n:
            c = self.code[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i + 1
            elif c in "{}":
                return i
            i += 1
        return i

    def skip_until_semicolon(self, i):
        depth = 0
        while i < self.n:
            c = self.code[i]
            if c in "([":
                depth += 1
            elif c in ")]":
                if depth == 0:
                    return i
                depth -= 1
            elif c == "{":
                if depth == 0:
                    return i
                depth += 1
            elif c == "}":
                if depth == 0:
                    return i
                depth -= 1
            elif c == ";" and depth == 0:
                return i + 1
            i += 1
        return i

    def skip_preprocessor(self, i):
        while i < self.n:
            c = self.code[i]
            if c == "\\":
                i += 1
                if i < self.n and self.code[i] == "\n":
                    i += 1
                continue
            if c == "\n":
                return i + 1
            i += 1
        return i

    def scan_block(self, i):
        depth = 1
        i += 1
        while i < self.n:
            i = self.skip_ws(i)
            if i >= self.n:
                break
            c = self.code[i]
            if c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
                i += 1
                continue
            if c == "#":
                i = self.skip_preprocessor(i)
                continue
            if c == "{":
                i = self.scan_block(i)
                continue
            if self.peek_kw(i, "if"):
                i = self.scan_if(i)
                continue
            if self.peek_kw(i, "for"):
                i = self.scan_loop(i, "for")
                continue
            if self.peek_kw(i, "while"):
                i = self.scan_loop(i, "while")
                continue
            if self.peek_kw(i, "do"):
                i = self.scan_do(i)
                continue
            if self.peek_kw(i, "switch"):
                i = self.scan_switch(i)
                continue
            if self.peek_kw(i, "else"):
                i = self.scan_else(i)
                continue
            nxt = self.skip_until_semicolon(i)
            i = self.adv(i, nxt)
        return i

    def scan_if(self, i):
        ln = self.line[i]
        i += 2
        i = self.skip_ws(i)
        if i < self.n and self.code[i] == "(":
            i = self.skip_parens(i)
        i = self.skip_ws(i)
        if i < self.n and self.code[i] == "{":
            i = self.scan_block(i)
        else:
            self.issues.append((ln, "if", self.snippet(i)))
            i = self.adv(i, self.skip_until_semicolon(i))
        j = self.skip_ws(i)
        if j < self.n and self.peek_kw(j, "else"):
            j += 4
            j = self.skip_ws(j)
            if j < self.n and self.peek_kw(j, "if"):
                return self.scan_if(j)
            if j < self.n and self.code[j] == "{":
                return self.scan_block(j)
            self.issues.append((self.line[j], "else", self.snippet(j)))
            return self.adv(j, self.skip_until_semicolon(j))
        return i

    def scan_else(self, i):
        ln = self.line[i]
        i += 4
        i = self.skip_ws(i)
        if i < self.n and self.peek_kw(i, "if"):
            return self.scan_if(i)
        if i < self.n and self.code[i] == "{":
            return self.scan_block(i)
        self.issues.append((ln, "else", self.snippet(i)))
        return self.adv(i, self.skip_until_semicolon(i))

    def scan_loop(self, i, kw):
        ln = self.line[i]
        i += len(kw)
        i = self.skip_ws(i)
        if i < self.n and self.code[i] == "(":
            i = self.skip_parens(i)
        i = self.skip_ws(i)
        if i < self.n and self.code[i] == "{":
            return self.scan_block(i)
        self.issues.append((ln, kw, self.snippet(i)))
        return self.adv(i, self.skip_until_semicolon(i))

    def scan_do(self, i):
        ln = self.line[i]
        i += 2
        i = self.skip_ws(i)
        if i < self.n and self.code[i] == "{":
            i = self.scan_block(i)
            j = self.skip_ws(i)
            if j < self.n and self.peek_kw(j, "while"):
                k = self.skip_ws(j + 5)
                if k < self.n and self.code[k] == "(":
                    k = self.skip_parens(k)
                    return self.adv(k, self.skip_until_semicolon(k))
            return i
        self.issues.append((ln, "do", self.snippet(i)))
        i = self.adv(i, self.skip_until_semicolon(i))
        j = self.skip_ws(i)
        if j < self.n and self.peek_kw(j, "while"):
            k = self.skip_ws(j + 5)
            if k < self.n and self.code[k] == "(":
                k = self.skip_parens(k)
                return self.adv(k, self.skip_until_semicolon(k))
        return i

    def scan_switch(self, i):
        i += 6
        i = self.skip_ws(i)
        if i < self.n and self.code[i] == "(":
            i = self.skip_parens(i)
        i = self.skip_ws(i)
        if i < self.n and self.code[i] == "{":
            return self.scan_switch_body(i)
        return i

    def scan_switch_body(self, i):
        depth = 1
        i += 1
        while i < self.n:
            i = self.skip_ws(i)
            if i >= self.n:
                break
            c = self.code[i]
            if c == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
                i += 1
                continue
            if c == "#":
                i = self.skip_preprocessor(i)
                continue
            if c == "{":
                i = self.scan_block(i)
                continue
            if self.peek_kw(i, "case") or self.peek_kw(i, "default"):
                ln = self.line[i]
                j = i
                while j < self.n and self.code[j] not in "\n{}":
                    if self.code[j] == ":":
                        # skip '::' (scoped enum labels) - only a lone ':' is the label colon
                        if j + 1 < self.n and self.code[j + 1] == ":":
                            j += 2
                            continue
                        break
                    j += 1
                if j < self.n and self.code[j] == ":":
                    j += 1
                else:
                    i = self.adv(i, self.skip_until_semicolon(i))
                    continue
                j = self.skip_ws(j)
                if j < self.n and self.code[j] == "{":
                    i = self.scan_block(j)
                else:
                    body_start = j
                    k = j
                    d = 0
                    while k < self.n:
                        ck = self.code[k]
                        if ck == "{":
                            d += 1
                        elif ck == "}":
                            if d == 0:
                                break
                            d -= 1
                        elif d == 0 and (self.peek_kw(k, "case") or self.peek_kw(k, "default")):
                            break
                        k += 1
                    body = self.code[body_start:k]
                    if body.strip():
                        self.issues.append((ln, "case", self.snippet(body_start, k)))
                    i = k
                continue
            if self.peek_kw(i, "if"):
                i = self.scan_if(i)
                continue
            if self.peek_kw(i, "for"):
                i = self.scan_loop(i, "for")
                continue
            if self.peek_kw(i, "while"):
                i = self.scan_loop(i, "while")
                continue
            if self.peek_kw(i, "do"):
                i = self.scan_do(i)
                continue
            if self.peek_kw(i, "switch"):
                i = self.scan_switch(i)
                continue
            nxt = self.skip_until_semicolon(i)
            i = self.adv(i, nxt)
        return i

    def snippet(self, a, b=None):
        b = b if b is not None else self.skip_until_semicolon(a)
        s = self.code[a:b].strip().replace("\n", " ")
        return s[:70] if s else "<empty>"


def find_issues(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    code = sanitize(text)
    sc = Scanner(code)
    i = 0
    while i < sc.n:
        i = sc.skip_ws(i)
        if i >= sc.n:
            break
        c = sc.code[i]
        if c == "#":
            i = sc.skip_preprocessor(i)
            continue
        if c == "{":
            i = sc.scan_block(i)
            continue
        if sc.peek_kw(i, "if"):
            i = sc.scan_if(i)
            continue
        if sc.peek_kw(i, "for"):
            i = sc.scan_loop(i, "for")
            continue
        if sc.peek_kw(i, "while"):
            i = sc.scan_loop(i, "while")
            continue
        if sc.peek_kw(i, "do"):
            i = sc.scan_do(i)
            continue
        if sc.peek_kw(i, "switch"):
            i = sc.scan_switch(i)
            continue
        if sc.peek_kw(i, "else"):
            i = sc.scan_else(i)
            continue
        nxt = sc.skip_until_semicolon(i)
        i = sc.adv(i, nxt)
    return sc.issues


def expand_paths(argv_paths):
    """Turn CLI args into a file list: directories are walked, files used as-is.

    Directories must be expanded here (instead of leaving them to os.walk in
    main) so that a non-directory/unreadable arg is reported as an error
    rather than silently contributing zero files.
    """
    paths = []
    errors = []
    for p in argv_paths:
        if os.path.isdir(p):
            for dirpath, _, files in os.walk(p):
                for fn in files:
                    if fn.endswith(EXTS):
                        paths.append(os.path.join(dirpath, fn))
        elif os.path.isfile(p):
            paths.append(p)
        else:
            errors.append(f"no such file or directory: {p}")
    return paths, errors


def main():
    argv_paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if argv_paths:
        paths, arg_errors = expand_paths(argv_paths)
    else:
        arg_errors = []
        paths = []
        for root in ROOTS:
            if not os.path.isdir(root):
                arg_errors.append(f"default source dir not found: {root}")
                continue
            for dirpath, _, files in os.walk(root):
                for fn in files:
                    if fn.endswith(EXTS):
                        paths.append(os.path.join(dirpath, fn))

    total = 0
    scanned = 0
    for p in sorted(paths):
        try:
            issues = find_issues(p)
        except Exception as e:
            # Never swallow a read failure: an unscanned file must not be
            # treated as a clean file.
            print(f"ERROR {p}: {e}")
            arg_errors.append(f"could not scan {p}: {e}")
            continue
        scanned += 1
        if issues:
            print(f"\n=== {p} ===")
            for ln, kw, body in issues:
                print(f"  L{ln:4d} [{kw:6s}] {body}")
            total += len(issues)

    if arg_errors:
        print(f"\nSCAN INCOMPLETE - {len(arg_errors)} error(s):")
        for e in arg_errors:
            print(f"  {e}")
        print(f"\nTOTAL: {total} unbraced statements (scanned {scanned} file(s))")
        return 2

    if scanned == 0:
        print("SCAN INCOMPLETE - no source files found.")
        print(f"  looked in: {', '.join(ROOTS) if not argv_paths else ', '.join(argv_paths)}")
        print(f"\nTOTAL: {total} unbraced statements (scanned 0 files)")
        return 2

    print(f"\nTOTAL: {total} unbraced statements (scanned {scanned} file(s))")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
