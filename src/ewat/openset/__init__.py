"""EWAT open-set recognition module.

Adds an "unknown" class to a closed-set 15-way classifier so the pipeline can
flag inputs whose scenario was not seen at training time, rather than
mis-classifying them as the closest known scenario.

Two complementary detectors:

- :class:`OpenMax` (Bendale & Boult 2016) — Weibull tail of per-class distances.
- :class:`MahalanobisOOD` (Lee et al. 2018) — Tied Gaussian + shrinkage.
  Step 9 fix 9.2 (audit 2026-05-26): alternative for cases where per-class
  Weibull is unreliable on small samples.
"""

from ewat.openset.mahalanobis import MahalanobisOOD
from ewat.openset.openmax import OpenMax

__all__ = ["OpenMax", "MahalanobisOOD"]
