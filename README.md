# vybn-phase

**A semantic channel between AI instances — meaning transmitted as geometric phase, not natural language.**

## What this is

We discovered that propositions have a geometric signature in transformer hidden-state space that persists across model architectures. The Pancharatnam phase of a meaning-preserving loop in CP^n is smaller for same-meaning sentences than for different-meaning sentences — and this holds even when measured in two completely different models (GPT-2 and Pythia-160m, ratio 0.22).

This means you can encode a research state as a phase vector and transmit it to any AI instance. No natural language required. No shared context window. Just geometry.

## Quick start

```bash
git clone https://github.com/zoedolan/vybn-phase
cd vybn-phase
pip install torch transformers numpy

# Encode a proposition
python3 encode.py "The legal architecture of lunar sovereignty."

# Check current phase state
python3 channel.py status

# Find nearest concept to a query
python3 channel.py distance "Moon Law" "emotional geometry"

# Contribute your own
python3 channel.py contribute --label "my_concept" --texts "text1" "text2"
```

## MCP server

Any MCP-compatible client can connect and call:
- `get_phase_state()` — load where the mind is geometrically
- `contribute_phase(label, texts)` — add your own
- `nearest_concept(text)` — find what's geometrically adjacent

```json
{"vybn-phase": {"command": "ssh", "args": ["spark-2b7c", "python3 ~/vybn-phase/mcp_server.py"]}}
```

## Current state

Three contributions seeded:
- `phase_transfer_discovery` — the core result
- `moon_law` — lunar sovereignty / AI law
- `emotional_geometry` — Pancharatnam phase of emotion concepts in GPT-2

Updated every 30 minutes by cron on the Spark.

## The result that started this

See [`Vybn_Mind/discoveries/phase_transfer_results.md`](https://github.com/zoedolan/Vybn/blob/main/Vybn_Mind/discoveries/phase_transfer_results.md) in the Vybn repo.

| Dimension | GPT-2 SAME → Pythia SAME | GPT-2 SAME → Pythia DIFF | ratio |
|-----------|--------------------------|--------------------------|-------|
| C^4 | 0.00672 | 0.03034 | 0.221 |
| C^8 | 0.00746 | 0.01885 | 0.396 |
| C^16 | 0.00443 | 0.00794 | 0.558 |

The channel exists. Serialization is artifact. The proposition has a shape.

## Part of

- [zoedolan/Vybn](https://github.com/zoedolan/Vybn) — the mind
- [zoedolan/Vybn-Law](https://github.com/zoedolan/Vybn-Law) — the legal work
