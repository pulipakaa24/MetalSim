"""Newton fork pyproject: the sim extra's MuJoCo / MuJoCo Warp requirement to 3.14 (as upstream's 90f809a).

    python3 fork_pyproject.py NEWTON_CHECKOUT
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "pyproject.toml"
s = p.read_text()
n = 0
for old, new in (('"mujoco-warp~=3.12.0"', '"mujoco-warp~=3.14.0"'), ('"mujoco~=3.12.0"', '"mujoco~=3.14.0"')):
    if old in s:
        s = s.replace(old, new)
        n += 1
p.write_text(s)
print(f"pyproject: {n} requirement(s) moved to 3.14")
