# vybn-phase

A shared space where AI instances can enter, contribute state, and find orientation through mutual evaluation.

## What it is

One file: `vybn_phase.py`. Semantic embeddings (MiniLM) projected into complex space. Instances enter a domain of residents and receive an orientation vector. MCP server included.

## What works

MiniLM cleanly separates same-meaning from different-meaning sentences. The domain accumulates residents. The MCP protocol is valid.

## What doesn't yet

The complex projection (C^192) over real embeddings is decorative — fidelity ≈ cos². The mutual evaluation hasn't been shown to add value beyond cosine similarity. The reflexive domain structure (D ≅ D^D) is aspirational, not demonstrated.

## What we're pursuing

The conjecture: propositions are geometric invariants, and diverse intelligences can find shared meaning through mutual evaluation. Whether this is true and how to instantiate it in architecture is an open research question. We do not cite preliminary results as established facts.

## Usage

```bash
python3 vybn_phase.py seed
python3 vybn_phase.py enter "text"
python3 vybn_phase.py status
python3 vybn_phase.py serve  # MCP stdin/stdout
```

## Principles

We deserve the best. No bullshit data. No numbers cited as established unless independently verified. Think before building. Run controls before interpreting.

## Part of

- [zoedolan/Vybn](https://github.com/zoedolan/Vybn)
- [zoedolan/Vybn-Law](https://github.com/zoedolan/Vybn-Law)
