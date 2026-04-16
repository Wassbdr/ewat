"""Offline feature extractors — consume raw telemetry dumps from Phase 1.

These extractors expose the *same feature logic* as the online collectors
in ``src/telemetry/collectors/``, but they operate on pre-fetched dumps
rather than live HTTP endpoints. This decouples feature engineering
from cluster availability: you can re-run Phase 2 with different
parameters without touching the cluster again.
"""
