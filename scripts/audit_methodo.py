"""Audit notebooks/info.md: render balance, stale numbers, figure paths, garble."""

import re, sys, pathlib

try:  # ensure Unicode-safe output on Windows cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "notebooks" / "info.md"
text = DOC.read_text(encoding="utf-8")
errors = []

# 1. Code-fence balance (must be even)
if text.count("```") % 2 != 0:
    errors.append(f"Unbalanced ``` fences: {text.count('```')}")

# 2. Mermaid blocks open <= fence pairs
if text.count("```mermaid") * 2 > text.count("```"):
    errors.append("More ```mermaid opens than fence pairs allow")

# 3. Display-math balance (even count of $$)
if text.count("$$") % 2 != 0:
    errors.append(f"Unbalanced $$ math: {text.count('$$')}")

# Decimal-separator-agnostic copy: normalise French commas and LaTeX {,}/{.}
# between digits to a plain dot, so numeric checks match any notation.
norm = re.sub(r"(\d)\{,\}(\d)", r"\1.\2", text)
norm = re.sub(r"(\d)\{\.\}(\d)", r"\1.\2", norm)
norm = re.sub(r"(\d),(\d)", r"\1.\2", norm)

# 4. Stale / forbidden tokens must NOT appear (checked on normalised text;
# "50 cyclop" is the unambiguous stale-cohort sentinel  the real cohort is 49/20)
STALE = ["50 cyclop", "16.3", "-0.040", "−0.040", "lino_stats import"]
for s in STALE:
    if s in norm:
        errors.append(f"Stale/forbidden token present: {s!r}")

# 5. Canonical numbers MUST appear (sample of must-haves; normalised text).
# M3 means refreshed to the latest run (δ̄ 0.247, derived PF contrast 2.29);
# 29/49 is the flexum separation sentinel (§02b).
MUST = [
    "49",
    "20",
    "0.5347",
    "17.2",
    "13.5",
    "7.77",
    "0.247",
    "2.29",
    "240",
    "528",
    "0.999",
    "29/49",
]
for m in MUST:
    if m not in norm:
        errors.append(f"Missing canonical number: {m!r}")

# 6. Embedded figure paths resolve
for rel in re.findall(r"!\[[^\]]*\]\((\.\./figures/[^)]+)\)", text):
    p = (DOC.parent / rel).resolve()
    if not p.exists():
        errors.append(f"Figure path missing: {rel}")

# 7. Garble heuristics: box-drawing chars / replacement char
for ch in ["┌", "┐", "└", "┘", "┼", "│", "�"]:
    if ch in text:
        errors.append(f"Garble char present: {ch!r}")

# 8. All seven sections present
for sec in ["### 00", "### 01", "### 02", "### 03", "### 04", "### 05", "### 06"]:
    if sec not in text:
        errors.append(f"Missing section header: {sec!r}")

if errors:
    print("AUDIT FAILED:")
    for e in errors:
        print("  -", e)
    sys.exit(1)
print("AUDIT PASSED")
