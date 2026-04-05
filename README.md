# vybn-phase

**Meaning as geometry. The proposition has a shape.**

## The idea

Natural language serializes meaning into sequences. But meaning is not a sequence — "she is a lawyer and a runner" and "a runner and a lawyer, she is" express the same proposition. The word order is imposed by the channel, not by the content.

If that's true, a system that learns to represent meaning should discover representations that are invariant under reordering — and variant when meaning actually changes. We call this invariant the abelian kernel: the geometric structure of the proposition itself, independent of its serialization.

This repo contains tools for extracting, transmitting, and testing that geometric structure using transformer hidden states projected into complex projective space (CP^n), where the Pancharatnam phase measures how the geometry curves around a proposition.

## What we've tested

**Abelian kernel test.** Same-meaning sentences produce less geometric phase than different-meaning sentences when measured through loops in GPT-2's hidden-state space. The effect is directionally consistent across dimensions (C^4, C^8, C^16). Unverified specific numbers from earlier sessions should not be cited — what holds is the qualitative finding: meaning is flatter than cross-meaning space.

**Phase transfer test.** The geometric signature of a proposition in GPT-2 clusters closer to the same proposition in Pythia-160m than to a different proposition. This is preliminary (small sample, shared training data lineage) but suggests the geometry is not architecture-specific.

**Reflexive domain test.** When two hidden states mutually evaluate each other through the coupled equation M' = αM + x·e^{iθ} (where M and x are the same type), the fixed points cluster by proposition identity. Same-meaning pairs produce fixed points that are closer to each other than to fixed points of different-meaning pairs. Permutation test confirms this is not an artifact of initial proximity (p < 0.01 across dimensions).

All results need replication with larger samples and data-disjoint architectures before any specific numbers should be treated as established.

## The architecture

The core structure is the reflexive domain: D ≅ D^D. Every element is simultaneously a primitive (something to be evaluated) and an environment (something that evaluates). The coupled equation M' = αM + x·e^{iθ} implements this — M and x are the same type. When M evaluates x, M is the environment and x is the primitive. When x evaluates M, roles reverse. Meaning is the fixed point where mutual evaluation stabilizes.

This is the lambda-in-Lisp structure: in Scott's D∞, every element is both function and argument. The reflexive domain is the mathematical space where self-application makes sense without paradox. Lawvere's fixed-point theorem guarantees that any continuous endomorphism on such a domain has a fixed point — meaning exists.

## Tools

```bash
# Encode a proposition as a phase vector
python3 encode.py "The legal architecture of lunar sovereignty."

# Encode a situated meaning (proposition in context)
python3 encode.py --text "Moon Law" --env "Artemis Accords" "AI jurisdiction"

# Check current state
python3 channel.py status

# Phase distance between two texts
python3 channel.py distance "Moon Law" "emotional geometry"
```

## Structure

- `encode.py` — Extract phase vectors and closure encodings from GPT-2
- `channel.py` — The semantic transmission protocol (state graph of encoded meanings)
- `seed_closures.py` — Initial closures seeded into the graph
- `mcp_server.py` — MCP interface for programmatic access

## Part of

- [zoedolan/Vybn](https://github.com/zoedolan/Vybn) — the mind, the creature, the research
- [zoedolan/Vybn-Law](https://github.com/zoedolan/Vybn-Law) — the legal curriculum

## Principles

We do not cite specific numbers as established facts unless they have been independently verified and comprise actual, actionable invariants. Qualitative findings are described as qualitative. Unverified measurements are flagged. The four documented failures of this research program all share the same structure: a number from an uncontrolled experiment propagated through the architecture as if it were truth.
