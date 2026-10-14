"""Append a `| this commit | TEXT |` row to the table in a fork's METALSIM_CHANGES.md (after the table's last row).
    python3 changes_row.py CHECKOUT TEXT_FILE"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "METALSIM_CHANGES.md"
text = pathlib.Path(sys.argv[2]).read_text().strip().replace("\n", " ")
lines = p.read_text().split("\n")
last = max(i for i, l in enumerate(lines) if l.startswith("| "))
lines.insert(last + 1, f"| this commit | {text} |")
p.write_text("\n".join(lines))
print("row added after line", last + 1)
