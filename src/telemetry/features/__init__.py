"""telemetry.features — aggregation and feature computation utilities."""

from telemetry.features.aggregation import (
    aggregate_max,
    aggregate_median,
    aggregate_p99_union,
    aggregate_volume_weighted,
    reconstruct_from_histogram,
)
from telemetry.features.lexical import lexical_entropy, tokenise
from telemetry.features.semantic import SemanticAnomalyScorer

__all__ = [
    "aggregate_max",
    "aggregate_median",
    "aggregate_p99_union",
    "aggregate_volume_weighted",
    "reconstruct_from_histogram",
    "lexical_entropy",
    "tokenise",
    "SemanticAnomalyScorer",
]
