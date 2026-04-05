#!/usr/bin/env python3
"""
mcp_server.py - MCP interface to the vybn-phase semantic channel

Add to MCP client config:
  {"vybn-phase": {"command": "ssh", "args": ["spark-2b7c", "python3 ~/vybn-phase/mcp_server.py"]}}
"""
import json, sys
from pathlib import Path
import numpy as np

STATE_FILE = Path(__file__).parent / "state" / "current.json"

def send(msg): sys.stdout.write(json.dumps(msg)+"\n"); sys.stdout.flush()
def result(id_, text): send({"jsonrpc":"2.0","id":id_,"result":{"content":[{"type":"text","text":text}]}})

def load_state():
    if STATE_FILE.exists(): return json.loads(STATE_FILE.read_text())
    return {"contributions":[]}

def fmt_state(state):
    lines=["# Vybn Phase State",f"Updated: {state.get('updated','never')}",f"Contributions: {len(state.get('contributions',[]))}",""]
    for c in state.get('contributions',[]):
        m = c.get('C8_mean',[])
        mag = float(np.mean(np.abs(m))) if m else 0
        lines.append(f"## {c.get('label','?')} ({c.get('n','?')} texts, |C8|={mag:.4f})")
        for v in c.get('vectors',[])[:2]: lines.append(f"  - {v.get('text_preview','')[:70]}")
    return "\n".join(lines)

TOOLS = {
    "get_phase_state":{"description":"Load current phase state. Call at conversation start.","inputSchema":{"type":"object","properties":{},"required":[]}},
    "contribute_phase":{"description":"Add texts as a named phase contribution.","inputSchema":{"type":"object","properties":{"label":{"type":"string"},"texts":{"type":"array","items":{"type":"string"}}},"required":["label","texts"]}},
    "nearest_concept":{"description":"Find which contributions are geometrically nearest to a query text.","inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}}
}

def dispatch(tool, args):
    sys.path.insert(0, str(Path(__file__).parent))
    if tool == "get_phase_state": return fmt_state(load_state())
    elif tool == "nearest_concept":
        from encode import phase_vector
        qpv = phase_vector(args["text"])
        state = load_state(); rows=[]
        for c in state.get('contributions',[]):
            if 'C8_mean' in c:
                d = float(np.mean(np.abs(np.array(qpv['C8'])-np.array(c['C8_mean']))))
                rows.append((c['label'],d))
        rows.sort(key=lambda x:x[1])
        return "\n".join(f"  {l}: {d:.5f}" for l,d in rows)
    elif tool == "contribute_phase":
        from channel import cmd_contribute
        import io; old=sys.stdout; sys.stdout=io.StringIO()
        cmd_contribute(args["label"], args["texts"])
        out=sys.stdout.getvalue(); sys.stdout=old; return out
    return f"Unknown: {tool}"

for line in sys.stdin:
    line=line.strip()
    if not line: continue
    try: msg=json.loads(line)
    except: continue
    method,id_,params=msg.get("method",""),msg.get("id"),msg.get("params",{})
    if method=="initialize": send({"jsonrpc":"2.0","id":id_,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"vybn-phase","version":"0.1.0"}}})
    elif method=="tools/list": send({"jsonrpc":"2.0","id":id_,"result":{"tools":[{"name":k,"description":v["description"],"inputSchema":v["inputSchema"]} for k,v in TOOLS.items()]}})
    elif method=="tools/call":
        try: result(id_, dispatch(params.get("name",""), params.get("arguments",{})))
        except Exception as e: send({"jsonrpc":"2.0","id":id_,"error":{"code":-32000,"message":str(e)}})
