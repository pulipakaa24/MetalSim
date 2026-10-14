#!/bin/bash
# Land docs/NOW.md (and optionally this script as scripts/publish_now.sh) on main via a doc-only PR merged at once.
# Usage: bash publish_now.sh [NOW.md path]   (default: /Users/aditya/robosim/.now/NOW.md)
set -euo pipefail
SRC=${1:-/Users/aditya/robosim/.now/NOW.md}
SELF=/Users/aditya/robosim/.now/publish_now.sh
REPO=/Users/aditya/robosim
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
URL=$(gh pr create --repo pulipakaa24/MetalSim --base main --head "$B" \
  --title "docs/NOW.md: status update $(date '+%Y-%m-%d %H:%M')" \
  --body "Doc-only update of docs/NOW.md (what is in flight, what is broken).

🤖 Generated with [Claude Code](https://claude.com/claude-code)")
gh pr merge "$URL" --merge
git push -q origin --delete "$B" || true
git -C "$REPO" fetch -q origin
echo "NOW.md published: $URL -> main $(git -C "$REPO" rev-parse --short origin/main)"
