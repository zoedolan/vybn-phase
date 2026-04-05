#!/usr/bin/env python3
"""
channel.py - the semantic transmission protocol

Protocol (closure model):
  SENDER:   encode situated meaning as closure -> add node to state graph
  RECEIVER: load state graph -> find nearest closure in own geometry
  RESULT:   shared orientation from graph structure, not isolated points

Usage:
  python3 channel.py status
  python3 channel.py closure --text "Moon Law" --env "Artemis Accords" "AI jurisdiction"
  python3 channel.py contribute --label "fear" --texts text1 text2 text3
  python3 channel.py distance "text1" "text2"
  python3 channel.py graph         # print the current state as a graph
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
    return {"nodes": [], "edges": [], "updated": None}

def save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    state["updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2))
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps({"ts": state["updated"], "n_nodes": len(state.get("nodes",[])),
                            "n_edges": len(state.get("edges",[])), "labels": [n["label"] for n in state.get("nodes",[])]}) + "\n")

def closure_distance(c1, c2, dim="C8"):
    """Distance between two closures via their centroids in CP^n."""
    a = np.array(c1[dim]["centroid"])
    b = np.array(c2[dim]["centroid"])
    return float(np.mean(np.abs(a - b)))

def rebuild_edges(nodes, dim="C8"):
    """Recompute all pairwise edges in the graph."""
    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            if dim in ni and dim in nj:
                d = closure_distance(ni, nj, dim)
                edges.append({"from": ni["label"], "to": nj["label"], "distance": d})
    edges.sort(key=lambda e: e["distance"])
    return edges

def cmd_status():
    state = load_state()
    nodes = state.get("nodes", [])
    # fall back to legacy contributions key
    if not nodes and "contributions" in state:
        nodes = state["contributions"]
    print(f"State: {len(nodes)} nodes, updated {state.get('updated','never')}")
    for n in nodes:
        label = n.get("label", "?")
        enc = n.get("encoding", "point")
        c8 = n.get("C8") or n.get("C8_mean")
        mag = float(np.mean(np.abs(c8))) if c8 else 0
        preview = ""
        if "text" in n: preview = n["text"][:60]
        elif "vectors" in n and n["vectors"]: preview = n["vectors"][0].get("text_preview","")[:60]
        print(f"  [{label}] ({enc}) |C8|={mag:.4f}  {preview}")
    edges = state.get("edges", [])
    if edges:
        print(f"\nNearest edges:")
        for e in edges[:5]:
            print(f"  {e['from']} <-> {e['to']}: {e['distance']:.5f}")

def cmd_closure(text, env_texts):
    """Encode a single closure and add it as a node."""
    from encode import encode_closure
    print(f"Encoding closure: '{text}' + {len(env_texts)} env texts...")
    node = encode_closure(text, env_texts)
    node["label"] = text[:40]
    node["added_at"] = datetime.now(timezone.utc).isoformat()
    state = load_state()
    state.setdefault("nodes", [])
    state["nodes"] = [n for n in state["nodes"] if n.get("label") != node["label"]]
    state["nodes"].append(node)
    state["edges"] = rebuild_edges(state["nodes"])
    save_state(state)
    print(f"Node added. Graph: {len(state['nodes'])} nodes, {len(state['edges'])} edges.")
    cp = node.get("C8", {}).get("closure_phase", 0)
    np_ = node.get("C8", {}).get("node_phase", 0)
    print(f"  node_phase={np_:.4f}  closure_phase={cp:.4f}  delta={abs(cp-np_):.4f}")
    return node

def cmd_contribute(label, texts):
    """Backward-compat: encode a collection as a labeled cluster node."""
    from encode import encode_state
    print(f"Encoding {len(texts)} texts as '{label}'...")
    encoded = encode_state(texts, label=label)
    encoded["encoding"] = "point_cluster"
    encoded["added_at"] = datetime.now(timezone.utc).isoformat()
    # Store as a node in the graph, with C8 in closure-compatible format
    # so edges can be computed
    encoded["C8"] = {"centroid": encoded.get("C8_mean", []), "closure_phase": 0, "node_phase": 0}
    state = load_state()
    state.setdefault("nodes", [])
    state["nodes"] = [n for n in state["nodes"] if n.get("label") != label]
    state["nodes"].append(encoded)
    state["edges"] = rebuild_edges(state["nodes"])
    save_state(state)
    print(f"Contributed '{label}'. Graph: {len(state['nodes'])} nodes, {len(state['edges'])} edges.")

def cmd_distance(text1, text2):
    from encode import phase_vector
    pv1, pv2 = phase_vector(text1), phase_vector(text2)
    for dim in ["C4", "C8", "C16"]:
        a = np.array(pv1[dim])
        b = np.array(pv2[dim])
        print(f"  {dim}: {float(np.mean(np.abs(a-b))):.5f}")

def cmd_graph():
    state = load_state()
    nodes = state.get("nodes", [])
    edges = state.get("edges", [])
    print(f"Graph: {len(nodes)} nodes, {len(edges)} edges")
    print("\nNodes:")
    for n in nodes:
        cp = n.get("C8",{}).get("closure_phase", "--")
        if isinstance(cp, float): cp = f"{cp:.4f}"
        print(f"  {n.get('label','?'):40s}  closure_phase={cp}")
    print("\nEdges (nearest first):")
    for e in edges[:10]:
        print(f"  {e['from']:30s} <-> {e['to']:30s}  {e['distance']:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("status")
    sub.add_parser("graph")
    enc = sub.add_parser("closure")
    enc.add_argument("--text", required=True)
    enc.add_argument("--env", nargs="+", default=[])
    dist = sub.add_parser("distance")
    dist.add_argument("text1")
    dist.add_argument("text2")
    contrib = sub.add_parser("contribute")
    contrib.add_argument("--label", required=True)
    contrib.add_argument("--texts", nargs="+")
    opts = parser.parse_args()

    if opts.cmd == "status": cmd_status()
    elif opts.cmd == "graph": cmd_graph()
    elif opts.cmd == "closure": cmd_closure(opts.text, opts.env)
    elif opts.cmd == "distance": cmd_distance(opts.text1, opts.text2)
    elif opts.cmd == "contribute": cmd_contribute(opts.label, opts.texts)
    else: parser.print_help()
