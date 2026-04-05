#!/usr/bin/env python3
"""Cron updater - encodes current Vybn state into the phase channel every 30 min."""
import json, subprocess
from datetime import datetime, timezone
from pathlib import Path
sys_path = str(Path(__file__).parent)
import sys; sys.path.insert(0, sys_path)
from channel import cmd_contribute, load_state, save_state

VYBN_MIND = Path.home() / "Vybn" / "Vybn_Mind"

def read(p, n=2000):
    try: return p.read_text(encoding='utf-8', errors='replace')[:n]
    except: return ""

def main():
    now = datetime.now(timezone.utc).isoformat()
    texts = []
    cont = VYBN_MIND / "continuity.md"
    if cont.exists(): texts.append(read(cont))
    nt = VYBN_MIND / "next_task.md"
    if nt.exists(): texts.append(read(nt))
    for f in sorted((VYBN_MIND/"discoveries").glob("*.md"), reverse=True)[:3]:
        texts.append(read(f))
    if texts:
        cmd_contribute("vybn_mind_current", texts)
        print(f"[{now}] Updated vybn_mind_current with {len(texts)} texts")
    else:
        print(f"[{now}] No texts found")

if __name__ == "__main__":
    main()
