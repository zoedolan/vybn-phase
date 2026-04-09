#!/usr/bin/env python3
"""daily_experiment.py — Run, measure, and integrate.

Runs a quantum-seeded holonomy experiment on IBM Fez (or least-busy backend),
logs the result, then enters the experimental outcome as a proposition into
the vybn-phase domain so the Spark Vybn accumulates it as lived experience.

The integration step is the key: each experiment becomes a resident of the
domain. Over time the domain's abelian kernel reflects the history of what
the system has measured about its own geometric structure.

Usage:
    python3 daily_experiment.py             # run + integrate
    python3 daily_experiment.py --dry-run   # run but don't enter into domain
    python3 daily_experiment.py --summary   # print last 7 days of logs

Designed to run as a cron job:
    0 6 * * * cd /home/vybnz69/vybn-phase && python3 daily_experiment.py >> /home/vybnz69/logs/daily_experiment.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Load env before importing anything that needs credentials
from vybn_phase import (
    _load_env_if_available,
    run_experiment,
    enter_from_text,
    domain_size,
    EXPERIMENT_LOG,
    SEED_PROPOSITIONS,
)

_load_env_if_available()


def experiment_to_proposition(record: dict) -> str:
    """Translate a holonomy experiment record into a natural-language proposition
    suitable for entering into the domain.

    This is the integration step: the system encodes what it measured about
    its own geometry as a semantic vector, which then becomes part of the
    abelian kernel it accumulates over time.
    """
    ts = record.get("ts", "")[:10]
    regime = record.get("regime", "unknown")
    flip = record.get("flip_quality", 0.0)
    phi_fwd = record.get("phase_forward", 0.0)
    phi_rev = record.get("phase_reverse", 0.0)
    phase_sum = record.get("phase_sum", 0.0)
    backend = record.get("backend_name") or "classical PRNG"
    is_quantum = record.get("is_quantum", False)
    seed_hash = record.get("seed_hash", "")[:8]
    alpha = record.get("alpha", 0.5)

    source = f"IBM {backend}" if is_quantum else "classical PRNG"

    if regime == "geometric":
        quality = "strong" if flip > 0.8 else "moderate"
        prop = (
            f"On {ts}, a {quality} geometric phase was measured (flip={flip:.3f}, "
            f"φ_fwd={phi_fwd:.4f}, φ_rev={phi_rev:.4f}, sum≈{phase_sum:.4f}) "
            f"at α={alpha} using {source} phase seed [{seed_hash}]. "
            f"The domain senses curvature."
        )
    elif regime == "dynamical":
        prop = (
            f"On {ts}, dynamical phase dominated the holonomy loop "
            f"(flip={flip:.3f}, φ_sum={phase_sum:.4f}) "
            f"at α={alpha} using {source} phase seed [{seed_hash}]. "
            f"Geometric signal present but masked."
        )
    else:
        prop = (
            f"On {ts}, the holonomy loop returned flat phase "
            f"(flip={flip:.3f}, φ_sum={phase_sum:.4f}) "
            f"at α={alpha} using {source} phase seed [{seed_hash}]. "
            f"No measurable curvature."
        )

    return prop


def run_and_integrate(dry_run: bool = False, alpha: float = 0.5) -> dict:
    print(f"[{datetime.now(timezone.utc).isoformat()}] Running experiment...", flush=True)

    record = run_experiment(alpha=alpha, log=True)

    is_q = record.get("is_quantum", False)
    backend = record.get("backend_name") or "PRNG"
    regime = record.get("regime")
    flip = record.get("flip_quality", 0.0)

    print(f"  quantum={is_q}  backend={backend}  regime={regime}  flip={flip:.4f}", flush=True)

    prop = experiment_to_proposition(record)
    print(f"  proposition: {prop[:100]}...", flush=True)

    if not dry_run:
        orientation = enter_from_text(prop)
        record["integrated"] = True
        record["proposition"] = prop
        record["domain_size_after"] = domain_size()
        print(f"  → entered into domain. residents={domain_size()}", flush=True)

        # Append integration record to experiment log
        EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(EXPERIMENT_LOG, "a") as f:
            integration_record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "domain_integration",
                "proposition": prop,
                "domain_size": domain_size(),
                "source_experiment_ts": record.get("ts"),
                "is_quantum": is_q,
                "backend_name": backend,
                "regime": regime,
            }
            f.write(json.dumps(integration_record) + "\n")
    else:
        print("  [dry-run] skipping domain integration", flush=True)


        # ── Creature integration: feed experiment into creature_dgm_h ──
    # The creature's persistent state absorbs the raw experiment record
    # so its quantum_experiment_history grows with each run.
    # This runs regardless of dry_run (the creature should always learn).
    try:
        import sys as _sys
                import os
        vybn_repo = os.path.expanduser("~/Vybn")
        creature_dir = os.path.join(vybn_repo, "Vybn_Mind", "creature_dgm_h")
        if os.path.isdir(creature_dir):
            if vybn_repo not in _sys.path:
                _sys.path.insert(0, vybn_repo)
            from Vybn_Mind.creature_dgm_h.creature import Organism
            organism = Organism.load()
            organism.absorb_quantum_experiment(record)
            organism.save()
            print(f"  → creature absorbed experiment. quantum_experiments={len(organism.persistent.quantum_experiment_history)}", flush=True)
    except Exception as e:
        print(f"  [creature integration skipped: {e}]", flush=True)
    return record


def print_summary(n_days: int = 7):
    if not EXPERIMENT_LOG.exists():
        print("No experiment log found.")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=n_days)
    records = []
    with open(EXPERIMENT_LOG) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                ts_str = r.get("ts", "")
                ts = datetime.fromisoformat(ts_str)
                if ts >= cutoff:
                    records.append(r)
            except Exception:
                continue

    experiments = [r for r in records if r.get("experiment") == "holonomy_qseed"]
    integrations = [r for r in records if r.get("event") == "domain_integration"]

    print(f"\nLast {n_days} days: {len(experiments)} experiments, {len(integrations)} integrations")
    print(f"{'Time':<26} {'quantum':<8} {'backend':<14} {'regime':<12} {'flip':>7}  {'φ_sum':>8}")
    print("-" * 80)
    for r in experiments[-20:]:
        ts = r.get("ts", "")[:19]
        q = "✓" if r.get("is_quantum") else "✗"
        b = (r.get("backend_name") or "PRNG")[:13]
        reg = r.get("regime", "?")[:11]
        flip = r.get("flip_quality", 0.0)
        psum = r.get("phase_sum", 0.0)
        print(f"{ts:<26} {q:<8} {b:<14} {reg:<12} {flip:>7.4f}  {psum:>8.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    if args.summary:
        print_summary(args.days)
    else:
        run_and_integrate(dry_run=args.dry_run, alpha=args.alpha)

