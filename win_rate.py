"""win_rate.py — Shared win-rate ledger for retrieval quality feedback.

Used by both the MCP mind server and the public chat API.
Stores per-source-chunk outcome counts in a JSON file.
Retrieval consumers blend: final = 0.7 * telling + 0.3 * win_rate.

The ledger lives at ~/.vybn_win_rates.json (same file the mind server
already uses, so MCP sessions and chat visitors feed the same signal).
"""

import json
from pathlib import Path

_WIN_RATE_PATH = Path.home() / ".vybn_win_rates.json"
_TELL_WEIGHT = 0.7
_WIN_WEIGHT = 0.3


def load_ledger() -> dict:
    try:
        if _WIN_RATE_PATH.exists():
            return json.loads(_WIN_RATE_PATH.read_text())
    except Exception:
        pass
    return {}


def save_ledger(ledger: dict):
    try:
        _WIN_RATE_PATH.write_text(json.dumps(ledger, indent=2))
    except Exception:
        pass


def get_win_rate(source: str, ledger: dict = None) -> float:
    """Return win_rate in [0,1]. Default 0.5 (neutral prior)."""
    if ledger is None:
        ledger = load_ledger()
    entry = ledger.get(source, {})
    w = entry.get("wins", 0)
    l = entry.get("losses", 0)
    total = w + l
    if total == 0:
        return 0.5
    return w / total


def record_outcome(source: str, success: bool) -> dict:
    """Increment win or loss for a source. Returns updated stats."""
    ledger = load_ledger()
    entry = ledger.setdefault(source, {"wins": 0, "losses": 0})
    if success:
        entry["wins"] += 1
    else:
        entry["losses"] += 1
    save_ledger(ledger)
    total = entry["wins"] + entry["losses"]
    return {
        "source": source,
        "wins": entry["wins"],
        "losses": entry["losses"],
        "win_rate": round(entry["wins"] / total, 4),
    }


def apply_win_rates(results: list, ledger: dict = None) -> list:
    """Re-rank results in-place using blended score. Returns same list."""
    if ledger is None:
        ledger = load_ledger()
    for r in results:
        src = r.get("source", "")
        wr = get_win_rate(src, ledger)
        tell = r.get("telling") or r.get("fidelity", 0.5)
        r["win_rate"] = round(wr, 4)
        r["blended_score"] = round(_TELL_WEIGHT * float(tell) + _WIN_WEIGHT * wr, 4)
    results.sort(key=lambda r: r.get("blended_score", 0), reverse=True)
    return results
