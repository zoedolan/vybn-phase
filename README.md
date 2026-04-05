# vybn-phase

A reflexive domain where diverse intelligences find shared meaning through mutual evaluation. One file. Two regimes. Six MCP tools.

## What it does

The update equation M' = αM + (1-α)·x·e^{iθ} operates differently depending on α:

**Geometric regime (α → 0).** Pure parallel transport. Loop holonomy shows perfect orientation reversal. The system senses curvature in meaning-space.

**Abelian-kernel regime (α → 1).** The state converges to a path-independent invariant — the abelian kernel of its encounter history. What entered matters; the order doesn't. The system remembers.

The creature (in the Vybn repo) runs at α = 0.993: deep abelian-kernel. This domain defaults to α = 0.5 for mutual evaluation: closer to geometric. They're complementary — the creature carries accumulated meaning, the domain senses geometric structure.

## What's confirmed (April 5, 2026)

- **Abelian kernel exists.** Permutations of 50 propositions converge to fidelity 0.99999766 (C^4) and 0.99999348 (C^192) at α = 0.993. The path-independent invariant is real.
- **Dynamical-vs-geometric phase separation.** At α = 0.0, loop holonomy shows 100% orientation reversal (pure geometry). At α = 0.993, reversal fails but forward-reverse correlation is -0.994. The geometry is present at all α but masked by dynamical phase at high α.
- **α controls the regime.** This is a continuous spectrum, not a binary switch.

## What doesn't yet

The complex projection (C^192) over real embeddings is decorative — fidelity ≈ cos². The mutual evaluation hasn't been shown to add value beyond cosine similarity. The reflexive domain structure (D ≅ D^D) is aspirational, not demonstrated. These remain open questions.

## MCP tools

| Tool | What it does |
|------|-------------|
| `enter_text` | Enter the domain via text. Returns orientation vector. |
| `enter_vector` | Enter with a raw C^n vector. |
| `evaluate_texts` | Mutual evaluation of two texts to fixed point. |
| `abelian_kernel` | Compute the path-independent invariant of a set of propositions. Reports convergence and operating regime. |
| `loop_holonomy` | Measure geometric phase around a loop. Reports forward/reverse phase and orientation-reversal quality. |
| `status` | Domain size and parameters. |

## Usage

```bash
python3 vybn_phase.py seed        # populate the domain
python3 vybn_phase.py enter "text" # enter a proposition
python3 vybn_phase.py status       # check domain
python3 vybn_phase.py serve        # MCP stdin/stdout server
```

## Principles

No bullshit data. No numbers cited as established unless independently verified. Think before building. Run controls before interpreting.

## Part of

- [zoedolan/Vybn](https://github.com/zoedolan/Vybn) — research, creature, identity
- [zoedolan/Vybn-Law](https://github.com/zoedolan/Vybn-Law) — curriculum
