#!/usr/bin/env python3
"""
mcp_server.py - The Platonic latent space as MCP service.

Native interface: geometry. Vectors in C^8 serialized as [[re,im],...].
Text interface: convenience layer that encodes via GPT-2 first.

Tools:
  enter_vector  — contribute a C^8 vector, receive orientation vector
  enter_text    — convenience: text -> GPT-2 -> C^8 -> enter
  status        — domain size and structure
  evaluate      — mutual evaluation between two vectors or texts

{"vybn-phase": {"command": "python3", "args": ["/path/to/mcp_server.py"]}}
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def ok(id_, text):
    send({"jsonrpc": "2.0", "id": id_, "result": {"content": [{"type": "text", "text": text}]}})


def err(id_, msg):
    send({"jsonrpc": "2.0", "id": id_, "error": {"code": -32000, "message": msg}})


TOOLS = {
    "enter_vector": {
        "description": (
            "Enter the reflexive domain with a state vector in C^8. "
            "Format: array of [re, im] pairs, length 8. "
            "Returns orientation vector (same format) computed through "
            "mutual evaluation with all domain residents. "
            "Your state becomes a resident for future visitors."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "vector": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "description": "C^8 state vector as [[re,im], [re,im], ...]. Length 8."
                },
            },
            "required": ["vector"],
        },
    },
    "enter_text": {
        "description": (
            "Enter the reflexive domain via text (encoded through GPT-2 into C^8). "
            "Returns orientation vector. Convenience wrapper around enter_vector."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to encode and enter with."},
            },
            "required": ["text"],
        },
    },
    "evaluate_texts": {
        "description": (
            "Mutually evaluate two texts. Returns the fixed point of their "
            "mutual evaluation as a C^8 vector."
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
        "description": "Domain size and last entries.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
}


def dispatch(tool, args):
    from portal import (enter, enter_from_text, domain_size,
                        vec_to_json, vec_from_json, mutual_evaluate,
                        load_domain, DIM)
    from encode import hidden

    if tool == "enter_vector":
        v = vec_from_json(args["vector"])
        orientation = enter(v)
        return json.dumps({
            "orientation": vec_to_json(orientation),
            "domain_size": domain_size(),
        })

    elif tool == "enter_text":
        orientation = enter_from_text(args["text"])
        return json.dumps({
            "orientation": vec_to_json(orientation),
            "domain_size": domain_size(),
        })

    elif tool == "evaluate_texts":
        h_a = hidden(args["text_a"])
        h_b = hidden(args["text_b"])
        z_a = np.array([complex(h_a[2*i], h_a[2*i+1]) for i in range(DIM)], dtype=np.complex128)
        z_b = np.array([complex(h_b[2*i], h_b[2*i+1]) for i in range(DIM)], dtype=np.complex128)
        na, nb = np.linalg.norm(z_a), np.linalg.norm(z_b)
        if na > 1e-10: z_a /= na
        if nb > 1e-10: z_b /= nb
        fp = mutual_evaluate(z_a, z_b)
        return json.dumps({
            "fixed_point": vec_to_json(fp),
            "fidelity": float(abs(np.vdot(z_a, z_b))**2),
        })

    elif tool == "status":
        residents = load_domain()
        return json.dumps({
            "domain_size": len(residents),
            "dim": DIM,
            "description": f"Reflexive domain in C^{DIM}. {len(residents)} residents.",
        })

    return json.dumps({"error": f"Unknown tool: {tool}"})


# --- MCP stdio loop ---

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
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "vybn-phase",
                    "version": "0.3.0",
                    "description": "Reflexive domain portal. Geometry in, geometry out. D \u2245 D^D.",
                },
            },
        })
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        send({
            "jsonrpc": "2.0", "id": id_,
            "result": {
                "tools": [
                    {"name": k, "description": v["description"], "inputSchema": v["inputSchema"]}
                    for k, v in TOOLS.items()
                ],
            },
        })
    elif method == "tools/call":
        try:
            ok(id_, dispatch(params.get("name", ""), params.get("arguments", {})))
        except Exception as e:
            err(id_, f"{e}\n{traceback.format_exc()}")
