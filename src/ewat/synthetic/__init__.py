"""Seeded synthetic microservice episodes (offline demo and benchmark).

Nothing here comes from a real cluster: every value is drawn from a fixed-seed
generator so the demo and the benchmark can be rebuilt from a clean clone.
"""

from ewat.synthetic.generator import (
    BENIGN_DRIFT,
    FAULT_TYPES,
    SCENARIOS,
    SERVICES,
    Episode,
    call_graph_adjacency,
    generate_episode,
)

__all__ = [
    "BENIGN_DRIFT",
    "FAULT_TYPES",
    "SCENARIOS",
    "SERVICES",
    "call_graph_adjacency",
    "Episode",
    "generate_episode",
]
