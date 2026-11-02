"""Reflow comment blocks longer than 100 columns that are not in a base file (stream hfield-epa).
Usage: python reflow.py <base file> <file>"""
import sys
import textwrap

base = set(open(sys.argv[1]).read().splitlines())
lines = open(sys.argv[2]).read().splitlines()
out, i = [], 0
while i < len(lines):
  ln = lines[i]
  st = ln.lstrip()
  if st.startswith("#") and ln not in base:
    j = i
    ind = ln[: len(ln) - len(st)]
    blk = []
    while j < len(lines) and lines[j].lstrip().startswith("#") and lines[j][: len(lines[j]) - len(lines[j].lstrip())] == ind \
        and lines[j] not in base:
      blk.append(lines[j].lstrip()[1:].strip())
      j += 1
    if any(len(lines[k]) > 100 for k in range(i, j)):
      out += textwrap.wrap(" ".join(blk), width=100, initial_indent=ind + "# ", subsequent_indent=ind + "# ",
                           break_long_words=False, break_on_hyphens=False)
    else:
      out += lines[i:j]
    i = j
    continue
  out.append(ln)
  i += 1
open(sys.argv[2], "w").write("\n".join(out) + "\n")
