"""Refresh the results in showcase/showcase.json from showcase/results/benchmark.json.

Narrative fields (title, problem, approach, limits, role, ...) are kept as they are in the
existing file; only `results` is rebuilt, so every number comes from the benchmark run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMAND = "ewat benchmark --episodes 50 --seed0 0"


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def build_results(bench: dict) -> list[dict]:
    setup = bench["setup"]
    eps = bench["episodes"]
    faults = [e for e in eps if e["fault_start"] is not None]
    drift = [e for e in eps if e["fault_start"] is None]
    detected = [e for e in faults if e["alert_step"] is not None and not e["false_alarm"]]
    correct = [e for e in detected if e["alert_type"] == e["scenario"]]
    false_alarms = [e for e in drift if e["false_alarm"]]
    leak = bench["per_type"].get("resource_leak", {})
    lat = bench["step_latency_ms"]
    base = (
        f"{setup['data']}, eval seeds {setup['eval_seeds']}, "
        f"train seeds {setup['train_seeds']}, {setup['hardware']}"
    )
    return [
        {
            "label": "False alarms on harmless drift (traffic ramp)",
            "value": f"{pct(bench['false_alarm_rate_drift_only'])} "
            f"({len(false_alarms)}/{len(drift)}); MMD alone without typing: "
            f"{pct(bench['mmd_only_false_alarm_rate_drift_only'])}",
            "command": COMMAND,
            "setup": f"{len(drift)} drift-only episodes, {base}",
        },
        {
            "label": f"Faults detected ({len(bench['per_type'])} fault types)",
            "value": f"{pct(bench['detection_rate'])} ({len(detected)}/{len(faults)})",
            "command": COMMAND,
            "setup": f"{len(faults)} fault episodes, {base}",
        },
        {
            "label": "Fault-type accuracy on detected faults",
            "value": f"{pct(bench['type_accuracy'])} ({len(correct)}/{len(detected)})",
            "command": COMMAND,
            "setup": base,
        },
        {
            "label": "Median / p90 detection delay",
            "value": f"{bench['median_delay_s']:.0f} s / {bench['p90_delay_s']:.0f} s"
            + (f" (resource leak median {leak['median_delay_s']:.0f} s)" if leak else ""),
            "command": COMMAND,
            "setup": base,
        },
        {
            "label": "Per-step latency (MMD gate + typing)",
            "value": f"p50 {lat['p50']:.2f} ms, p95 {lat['p95']:.1f} ms",
            "command": COMMAND,
            "setup": f"{setup['episodes']} episodes, {base}",
        },
    ]


def main() -> None:
    target = ROOT / "showcase.json"
    bench_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "benchmark.json"
    card = json.loads(target.read_text()) if target.exists() else {}
    card.setdefault("role", "TODO")
    card["results"] = build_results(json.loads(bench_path.read_text()))
    target.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {target} (role kept: {card['role']!r})")


if __name__ == "__main__":
    main()
