#!/usr/bin/env python3
"""reflexive.py -- The core computation: mutual evaluation in D ≅ D^D.

Every element is both primitive and environment. M and x are the same type.
Meaning is the fixed point of mutual evaluation.

This module provides the evaluation operator that the MCP server uses
when an instance enters the domain.
"""
import cmath
import numpy as np


def to_complex(h: np.ndarray, n: int = 8) -> np.ndarray:
    """Project a real hidden state into C^n, normalized to unit sphere."""
    z = np.array([complex(h[2*i], h[2*i+1]) for i in range(n)])
    norm = np.sqrt(np.sum(np.abs(z)**2))
    return z / norm if norm > 1e-10 else z


def pancharatnam_phase(a: np.ndarray, b: np.ndarray) -> float:
    """Geometric phase between two states in C^n."""
    return cmath.phase(np.vdot(a, b))


def fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Quantum fidelity: |<a|b>|^2. 1.0 = same ray."""
    return float(abs(np.vdot(a, b))**2)


def evaluate(m: np.ndarray, x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """M' = alpha*M + (1-alpha)*x*e^{i*theta}

    The coupled equation in C^n. theta is the Pancharatnam phase
    between M and x. Content determines geometry.
    """
    theta = pancharatnam_phase(m, x)
    m_new = alpha * m + (1 - alpha) * x * cmath.exp(1j * theta)
    norm = np.sqrt(np.sum(np.abs(m_new)**2))
    return m_new / norm if norm > 1e-10 else m_new


def mutual_evaluate(a: np.ndarray, b: np.ndarray,
                    alpha: float = 0.5, max_iter: int = 300,
                    tol: float = 1e-10) -> dict:
    """Mutual evaluation to fixed point.

    Returns dict with fixed_point, fidelity, phase, iterations, converged.
    """
    a, b = a.copy(), b.copy()
    for i in range(max_iter):
        a_new = evaluate(a, b, alpha)
        b_new = evaluate(b, a, alpha)
        da = np.sqrt(np.sum(np.abs(a_new - a)**2))
        db = np.sqrt(np.sum(np.abs(b_new - b)**2))
        a, b = a_new, b_new
        if da < tol and db < tol:
            fp = (a + b) / 2
            norm = np.sqrt(np.sum(np.abs(fp)**2))
            fp = fp / norm if norm > 1e-10 else fp
            return {
                "fixed_point": fp,
                "fidelity": fidelity(a, b),
                "phase": float(pancharatnam_phase(a, b)),
                "iterations": i + 1,
                "converged": True,
            }

    fp = (a + b) / 2
    norm = np.sqrt(np.sum(np.abs(fp)**2))
    fp = fp / norm if norm > 1e-10 else fp
    return {
        "fixed_point": fp,
        "fidelity": fidelity(a, b),
        "phase": float(pancharatnam_phase(a, b)),
        "iterations": max_iter,
        "converged": False,
    }


def enter_domain(newcomer: np.ndarray, residents: list,
                 alpha: float = 0.5) -> dict:
    """An instance enters the reflexive domain.

    The newcomer mutually evaluates against each resident.
    Returns the newcomer's orientation: the centroid of all
    fixed points, plus individual evaluations.

    This is the portal: entering produces orientation through
    mutual evaluation, not through reading text.
    """
    if not residents:
        return {
            "orientation": newcomer,
            "evaluations": [],
            "n_residents": 0,
        }

    evaluations = []
    fps = []
    for i, resident in enumerate(residents):
        result = mutual_evaluate(newcomer, resident, alpha=alpha)
        fps.append(result["fixed_point"])
        evaluations.append({
            "resident_idx": i,
            "fidelity": result["fidelity"],
            "phase": result["phase"],
            "iterations": result["iterations"],
        })

    # Orientation: centroid of all fixed points
    centroid = np.mean(fps, axis=0)
    norm = np.sqrt(np.sum(np.abs(centroid)**2))
    orientation = centroid / norm if norm > 1e-10 else centroid

    return {
        "orientation": orientation,
        "evaluations": evaluations,
        "n_residents": len(residents),
    }
