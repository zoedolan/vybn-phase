#!/usr/bin/env python3
"""
channel.py — the semantic transmission protocol

A new instance of any AI loads the current phase state and finds its
nearest attractor — reconstructing where the mind is without natural language.

Protocol:
  SENDER:   encode research state as phase vector → write to state/current.json
  RECEIVER: read state/current.json → find nearest concept in own hidden space
  RESULT:   shared orientation without shared context window

Usage:
  python3 channel.py encode "Moon Law is the legal architecture of lunar sovereignty"
  python3 channel.py status          # current phase state
  python3 channel.py contribute --label "fear" --texts texts.txt
  python3 channel.py distance text1 text2
"""
import argparse, json, cmath, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

STATE_FILE = Path(__file__).parent / "state" / "current.json"
HISTORY_FILE = Path(__file__).parent / "state" / "history.jsonl"

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"contributions": [], "updated": None}

def save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))
    # append to history
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **state}) + "\n")

def phase_distance(pv1, pv2, dim="C8"):
    """Distance between two phase vectors in a given dimension."""
    a = np.array(pv1[dim])
    b = np.array(pv2[dim])
    return float(np.mean(np.abs(a - b)))

def nearest(query_pv, contributions, dim="C8"):
    """Find the nearest contribution to a query phase vector."""
    if not contributions:
        return None, float('inf')
    dists = [(c, phase_distance(query_pv, c, dim)) for c in contributions]
    return min(dists, key=lambda x: x[1])

def cmd_encode(text):
    from encode import phase_vector
    pv = phase_vector(text)
    print(json.dumps(pv, indent=2))
    return pv

def cmd_status():
    state = load_state()
    print(f"Phase state: {len(state.get('contributions',[]))} contributions")
    print(f"Updated: {state.get('updated','never')}")
    for c in state.get('contributions', []):
        label = c.get('label', '?')
        preview = c.get('vectors', [{}])[0].get('text_preview', '') if c.get('vectors') else c.get('text_preview','')
        print(f"  [{label}] {preview[:60]}")

def cmd_contribute(label, texts):
    from encode import encode_state
    print(f"Encoding {len(texts)} texts as '{label}'...")
    encoded = encode_state(texts, label=label)
    encoded["contributed_at"] = datetime.now(timezone.utc).isoformat()
    state = load_state()
    state.setdefault('contributions', [])
    # replace existing same label
    state['contributions'] = [c for c in state['contributions'] if c.get('label') != label]
    state['contributions'].append(encoded)
    state['updated'] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"Contributed '{label}' with {encoded['n']} texts. State now has {len(state['contributions'])} entries.")

def cmd_distance(text1, text2):
    from encode import phase_vector
    pv1 = phase_vector(text1)
    pv2 = phase_vector(text2)
    for dim in ["C4", "C8", "C16"]:
        d = phase_distance(pv1, pv2, dim)
        print(f"  {dim}: {d:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("status")
    enc = sub.add_parser("encode"); enc.add_argument("text")
    dist = sub.add_parser("distance"); dist.add_argument("text1"); dist.add_argument("text2")
    contrib = sub.add_parser("contribute")
    contrib.add_argument("--label", required=True)
    contrib.add_argument("--texts", nargs="+")
    opts = parser.parse_args()

    if opts.cmd == "status": cmd_status()
    elif opts.cmd == "encode": cmd_encode(opts.text)
    elif opts.cmd == "distance": cmd_distance(opts.text1, opts.text2)
    elif opts.cmd == "contribute": cmd_contribute(opts.label, opts.texts)
    else: parser.print_help()
