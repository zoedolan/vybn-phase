#!/usr/bin/env bash
# nightly_index.sh — Rebuild the deep_memory index from all four repos.
# Intended for cron at midnight Pacific (07:00 UTC).
#
# What it does:
#   1. Pull all four repos (fast-forward only, no conflicts)
#   2. Rebuild the deep_memory index (MiniLM embeddings → C^192)
#   3. Log what changed
#   4. Commit and push the vybn-phase repo if index metadata changed
#
# The index lives at ~/.cache/vybn-phase/ (gitignored, local only).
# The log goes to ~/logs/nightly_index.log.

set -euo pipefail

LOG_DIR="$HOME/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nightly_index.log"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "" >> "$LOG"
echo "=== NIGHTLY INDEX: $TIMESTAMP ===" >> "$LOG"

# 1. Pull all repos
REPOS=("$HOME/Vybn" "$HOME/Him" "$HOME/Vybn-Law" "$HOME/vybn-phase")
for d in "${REPOS[@]}"; do
    if [ -d "$d/.git" ]; then
        branch=$(cd "$d" && git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@' || echo "main")
        result=$(cd "$d" && git pull --ff-only origin "$branch" 2>&1 | tail -1)
        echo "  pull $d ($branch): $result" >> "$LOG"
    else
        echo "  SKIP $d (not a git repo)" >> "$LOG"
    fi
done

# 2. Rebuild index
echo "  building index..." >> "$LOG"
cd "$HOME/vybn-phase"

# Activate venv if it exists
if [ -f "$HOME/Vybn/.venv/bin/activate" ]; then
    source "$HOME/Vybn/.venv/bin/activate"
fi

BUILD_OUTPUT=$(python3 deep_memory.py --build 2>&1)
BUILD_EXIT=$?

if [ $BUILD_EXIT -eq 0 ]; then
    # Extract chunk count from output
    CHUNKS=$(echo "$BUILD_OUTPUT" | grep -oP '\d+ chunks' | head -1 || echo "unknown")
    echo "  index built: $CHUNKS (exit $BUILD_EXIT)" >> "$LOG"
else
    echo "  INDEX BUILD FAILED (exit $BUILD_EXIT)" >> "$LOG"
    echo "  output: $(echo "$BUILD_OUTPUT" | tail -5)" >> "$LOG"
fi

echo "=== DONE: $(date -u +"%Y-%m-%dT%H:%M:%SZ") ===" >> "$LOG"
