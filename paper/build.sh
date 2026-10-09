#!/bin/bash
# Assemble paper.tex from section files (inlining auto-generated tables) and compile with pdflatex+bibtex.
set -e
cd "$(dirname "$0")"
python3 - <<'PY'
import re
parts = "".join(open(f).read() for f in ["paper_head.tex", "paper_abstract.tex", "sec_intro_related.tex", "sec_method.tex", "sec_results.tex"])
parts = re.sub(r"\\input\{(auto_[a-z_]+\.tex)\}", lambda m: open(m.group(1)).read().replace("\\hline", "\\midrule").rstrip(), parts)
open("paper.tex", "w").write(parts)
PY
pdflatex -interaction=nonstopmode paper.tex > build.log 2>&1 || true
bibtex paper >> build.log 2>&1 || true
pdflatex -interaction=nonstopmode paper.tex >> build.log 2>&1 || true
pdflatex -interaction=nonstopmode paper.tex >> build.log 2>&1 || true
grep -E "^!|undefined" build.log || echo "no LaTeX errors"
pdfinfo paper.pdf | grep Pages
