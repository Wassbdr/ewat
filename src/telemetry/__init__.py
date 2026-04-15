"""telemetry — EWAT signal extraction package.

Extracts S(t) ∈ ℝ^{N×17} = [M(t) | T(t) | L(t)] from live cluster telemetry.

    from telemetry.signal_builder import SignalBuilder, SignalSnapshot
"""

from telemetry.signal_builder import SignalBuilder, SignalSnapshot

__all__ = ["SignalBuilder", "SignalSnapshot"]
