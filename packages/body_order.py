"""Legacy numeric body-order hint shared by conversion and merge audits.

This heuristic is not a joint-name contract. Both or neither columns near the
reference are ambiguous; explicit source-order configuration is tracked in TODO.md.
"""
from __future__ import annotations

import numpy as np

BODY_LIFT_REF = 0.22
BODY_TOL = 0.05


def detect_body_order(arr) -> str | None:
    """Infer lift's column from medians near 0.22 +/- 0.05; return None if ambiguous."""
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 20 or len(values) == 0:
        return None
    near18 = abs(float(np.median(values[:, 18])) - BODY_LIFT_REF) <= BODY_TOL
    near19 = abs(float(np.median(values[:, 19])) - BODY_LIFT_REF) <= BODY_TOL
    if near18 and not near19:
        return "lift_first"
    if near19 and not near18:
        return "waist_first"
    return None
