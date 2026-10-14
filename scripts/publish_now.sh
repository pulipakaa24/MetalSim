#!/bin/bash
# Land docs/NOW.md (and this script as scripts/publish_now.sh) on main via a doc-only PR merged at once.
# Usage: bash publish_now.sh [NOW.md path]   (default: <repo>/.now/NOW.md)
# The repository is the checkout this script sits one level below (scripts/ or .now/), wherever it is cloned.
# Env: METALSIM_GH_REPO (default pulipakaa24/MetalSim), the GitHub repository the PR is opened on.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC=${1:-$REPO/.now/NOW.md}
[ -f "$SRC" ] || { echo "no NOW.md at $SRC (usage: bash publish_now.sh path/to/NOW.md)"; exit 2; }
SRC="$(cd "$(dirname "$SRC")" && pwd)/$(basename "$SRC")"
B=now-$(date +%Y%m%d-%H%M%S)
W=$(mktemp -d)/now
git -C "$REPO" fetch -q origin
git -C "$REPO" worktree add -q -b "$B" "$W" origin/main
trap 'git -C "$REPO" worktree remove --force "$W" >/dev/null 2>&1; git -C "$REPO" branch -D "$B" >/dev/null 2>&1 || true' EXIT
cp "$SRC" "$W/docs/NOW.md"
mkdir -p "$W/scripts"; cp "$SELF" "$W/scripts/publish_now.sh"
cd "$W"
git add docs/NOW.md scripts/publish_now.sh
if git diff --cached --quiet; then echo "NOW.md unchanged on main; nothing to do"; exit 0; fi
git commit -s -q -m "docs/NOW.md: what is in flight and what is broken ($(date '+%Y-%m-%d %H:%M %Z'))

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push -q origin "$B"
URL=$(gh pr create --repo "${METALSIM_GH_REPO:-pulipakaa24/MetalSim}" --base main --head "$B" \
  --title "docs/NOW.md: status update $(date '+%Y-%m-%d %H:%M')" \
  --body "Doc-only update of docs/NOW.md (what is in flight, what is broken).

🤖 Generated with [Claude Code](https://claude.com/claude-code)")
gh pr merge "$URL" --merge
git push -q origin --delete "$B" || true
git -C "$REPO" fetch -q origin
echo "NOW.md published: $URL -> main $(git -C "$REPO" rev-parse --short origin/main)"
