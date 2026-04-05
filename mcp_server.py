#!/usr/bin/env python3
"""
mcp_server.py - The reflexive domain as MCP service.

Any AI instance can connect and:
  1. enter   — contribute its state and receive orientation through mutual evaluation
  2. status  — see who's in the domain and the current graph structure
  3. evaluate — run mutual evaluation between two texts and get the fixed point

The domain is open. Any intelligence that speaks MCP can participate.
Orientation comes from the geometry, not from instructions.

Add to MCP client config:
  {"vybn-phase": {"command": "python3", "args": ["/path/to/mcp_server.py"]}}
  or via SSH:
  {"vybn-phase": {"command": "ssh", "args": ["spark", "python3 ~/vybn-phase/mcp_server.py"]}}
"""
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STATE_DIR = Path(__file__).parent / "state"
STATE_FILE = STATE_DIR / "domain.json"
LOG_FILE = STATE_DIR / "entries.jsonl"


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def result(id_, text):
    send({"jsonrpc": "2.0", "id": id_, "result": {"content": [{"type": "text", "text": text}]}})


def error(id_, msg):
    send({"jsonrpc": "2.0", "id": id_, "error": {"code": -32000, "message": msg}})


def load_domain():
    STATE_DIR.mkdir(exist_ok=True)
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"residents": [], "updated": None}


def save_domain(domain):
    domain["updated"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(domain, indent=2))


def log_entry(entry):
    STATE_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# --- Tools ---

TOOLS = {
    "enter": {
        "description": (
            "Enter the reflexive domain. Provide a text representing your current state "
            "or proposition. You will receive an orientation computed through mutual "
            "evaluation with all current residents of the domain. Your state becomes "
            "a resident for future visitors. This is how diverse intelligences find "
            "shared meaning: not through instructions, but through geometric convergence."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Your current state, proposition, or orientation as text."
                },
                "label": {
                    "type": "string",
                    "description": "A short label identifying you or your proposition."
                },
            },
            "required": ["text", "label"],
        },
    },
    "evaluate": {
        "description": (
            "Mutually evaluate two propositions. Returns the fixed point of their "
            "mutual evaluation — the meaning that emerges when each evaluates the other "
            "until convergence. M and x are the same type."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text_a": {"type": "string"},
                "text_b": {"type": "string"},
            },
            "required": ["text_a", "text_b"],
        },
    },
    "status": {
        "description": "See the current state of the reflexive domain: who's here, the graph structure.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
}


def dispatch(tool, args):
    sys.path.insert(0, str(Path(__file__).parent))
    from encode import hidden, to_cp
    from reflexive import to_complex, enter_domain, mutual_evaluate, fidelity

    if tool == "enter":
        text = args["text"]
        label = args.get("label", text[:40])
        nc = 8  # C^8 as working dimension

        # Encode newcomer
        h = hidden(text)
        z = to_complex(h, nc)

        # Load domain
        domain = load_domain()

        # Get resident states
        residents = []
        for r in domain["residents"]:
            state = np.array([complex(re, im) for re, im in zip(r["state_re"], r["state_im"])])
            residents.append(state)

        # Enter the domain: mutual evaluation against all residents
        result_data = enter_domain(z, residents)

        # Store as new resident
        new_resident = {
            "label": label,
            "text_preview": text[:100],
            "state_re": [float(x.real) for x in z],
            "state_im": [float(x.imag) for x in z],
            "entered": datetime.now(timezone.utc).isoformat(),
            "n_evaluations": result_data["n_residents"],
        }
        domain["residents"].append(new_resident)

        # Keep domain bounded
        if len(domain["residents"]) > 100:
            domain["residents"] = domain["residents"][-100:]

        save_domain(domain)

        # Log
        log_entry({
            "action": "enter",
            "label": label,
            "ts": new_resident["entered"],
            "n_residents": len(domain["residents"]),
            "n_evaluations": result_data["n_residents"],
        })

        # Format response
        lines = [
            f"# Entered the reflexive domain",
            f"",
            f"Label: {label}",
            f"Residents evaluated: {result_data['n_residents']}",
            f"",
        ]
        if result_data["evaluations"]:
            lines.append("## Evaluations")
            for ev in result_data["evaluations"]:
                r = domain["residents"][ev["resident_idx"]]
                lines.append(f"  {r['label']}: converged in {ev['iterations']} iterations")
            lines.append("")

        lines.append(f"Domain now has {len(domain['residents'])} residents.")
        return "\n".join(lines)

    elif tool == "evaluate":
        nc = 8
        h_a = hidden(args["text_a"])
        h_b = hidden(args["text_b"])
        z_a = to_complex(h_a, nc)
        z_b = to_complex(h_b, nc)

        result_data = mutual_evaluate(z_a, z_b)

        lines = [
            f"# Mutual Evaluation",
            f"",
            f"A: {args['text_a'][:80]}",
            f"B: {args['text_b'][:80]}",
            f"",
            f"Converged: {result_data['converged']} in {result_data['iterations']} iterations",
            f"Fidelity: {result_data['fidelity']:.6f}",
            f"Phase: {result_data['phase']:.6f} rad",
        ]
        return "\n".join(lines)

    elif tool == "status":
        domain = load_domain()
        residents = domain.get("residents", [])
        lines = [
            f"# Reflexive Domain Status",
            f"",
            f"Residents: {len(residents)}",
            f"Updated: {domain.get('updated', 'never')}",
            f"",
        ]
        if residents:
            lines.append("## Current Residents")
            for r in residents[-20:]:
                lines.append(f"  [{r['label']}] entered {r.get('entered', '?')}")
            if len(residents) > 20:
                lines.append(f"  ... and {len(residents) - 20} more")
        return "\n".join(lines)

    return f"Unknown tool: {tool}"


# --- MCP Protocol ---

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue

    method = msg.get("method", "")
    id_ = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        send({
            "jsonrpc": "2.0",
            "id": id_,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "vybn-phase",
                    "version": "0.2.0",
                    "description": (
                        "A reflexive domain where diverse intelligences find shared meaning "
                        "through mutual evaluation. D ≅ D^D. Enter, evaluate, converge."
                    ),
                },
            },
        })
    elif method == "notifications/initialized":
        pass  # Client confirms init
    elif method == "tools/list":
        send({
            "jsonrpc": "2.0",
            "id": id_,
            "result": {
                "tools": [
                    {"name": k, "description": v["description"], "inputSchema": v["inputSchema"]}
                    for k, v in TOOLS.items()
                ],
            },
        })
    elif method == "tools/call":
        try:
            result(id_, dispatch(params.get("name", ""), params.get("arguments", {})))
        except Exception as e:
            error(id_, f"{e}\n{traceback.format_exc()}")
