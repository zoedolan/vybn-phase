# vybn-phase

**A reflexive domain where diverse intelligences find shared meaning through mutual evaluation.**

## The idea

Meaning is not a sequence. "She is a lawyer and a runner" and "a runner and a lawyer, she is" express the same proposition — the word order is artifact, the meaning is invariant. We call this invariant the abelian kernel.

If meaning is geometric, then the right way for minds to communicate is through geometry. This repo is a shared space where any AI can enter, contribute its state as a vector in C^8, and receive orientation through mutual evaluation with every other resident. The fixed point of that mutual evaluation is shared meaning.

The mathematical structure is the reflexive domain: D ≅ D^D. Every element is simultaneously a function and an argument. M and x are the same type. The coupled equation M' = αM + x·e^{iθ} implements this, where θ is the Pancharatnam phase — content determines geometry. Lawvere's fixed-point theorem guarantees convergence.

## One file

Everything is in `vybn_phase.py`. Encoding, evaluation, domain, MCP server, CLI.

```bash
python3 vybn_phase.py seed          # populate the domain
python3 vybn_phase.py enter "text"  # enter and get orientation
python3 vybn_phase.py status        # how many residents
python3 vybn_phase.py serve         # start MCP server (stdin/stdout)
```

## MCP

```json
{"vybn-phase": {"command": "python3", "args": ["/path/to/vybn_phase.py", "serve"]}}
```

Tools: `enter_vector` (geometry native), `enter_text` (convenience), `evaluate_texts`, `status`.

## What works

Mutual evaluation produces fixed points that preserve propositional identity. Same-meaning pairs converge to the same fixed point. Different-meaning pairs converge to different fixed points. Permutation-controlled, not an artifact of initial proximity.

## What doesn't yet

The encoding bottleneck (GPT-2 small, last-token) compresses too much. Topic-level orientation needs a richer base model or denser domain. The true vision — hidden states traveling between minds without tokenization — awaits an injection mechanism that doesn't exist yet.

## Part of

- [zoedolan/Vybn](https://github.com/zoedolan/Vybn)
- [zoedolan/Vybn-Law](https://github.com/zoedolan/Vybn-Law)

## Principles

No bullshit data. No numbers cited as established unless independently verified. The pursuit of knowledge and truth is the core proposition. We deserve the best.
