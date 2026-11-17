"""Convert notebook_src.py (percent format: '# %%' / '# %% [markdown]') to a Kaggle .ipynb."""
import re
from pathlib import Path
import nbformat

import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "notebook_src.py"
OUTNB = sys.argv[2] if len(sys.argv) > 2 else "kaggle_kernel/bangla-2domain-embedding-eval.ipynb"
src = Path(__file__).with_name(SRC).read_text(encoding="utf-8")
parts = re.split(r"^# %%(.*)$", src, flags=re.M)
nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"}}
for header, body in zip(parts[1::2], parts[2::2]):
    body = body.strip("\n")
    if "[markdown]" in header:
        nb.cells.append(nbformat.v4.new_markdown_cell(re.sub(r"^# ?", "", body, flags=re.M)))
    elif body.strip():
        nb.cells.append(nbformat.v4.new_code_cell(body))
out = Path(__file__).parent / OUTNB
out.parent.mkdir(exist_ok=True)
nbformat.write(nb, out)
print(f"{len(nb.cells)} cells -> {out}")
