"""`ewat` command line: offline synthetic demo, benchmark, window prediction, API.

ewat demo                      # one synthetic episode end to end + poster
ewat benchmark --episodes 50   # seeded synthetic benchmark → JSON
ewat predict window.json       # metrics window → drift, fault type, latency
ewat serve                     # FastAPI on :8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_BAR = "█"


def _bar(x: float, width: int = 24) -> str:
    n = int(min(x, 3.0) / 3.0 * width)
    return (_BAR * n).ljust(width)


def cmd_train(a) -> int:
    from ewat.early_warning.workflow import train

    train(Path(a.model))
    return 0


def cmd_demo(a) -> int:
    from ewat.early_warning.plot import plot_timeline
    from ewat.early_warning.workflow import first_alert_outcome, get_pipeline
    from ewat.synthetic import generate_episode

    pipe = get_pipeline(Path(a.model), retrain=a.retrain)
    e = generate_episode(a.scenario, a.seed)
    print(
        f"\nSYNTHETIC episode '{e.scenario}' seed={e.seed}: 6 services × 17 features, "
        f"{len(e.phase)} steps of 30 s  (fault target: {e.target})"
    )
    print(
        f"phases: normal 0–{e.drift_start - 1} · harmless drift {e.drift_start}–"
        f"{(e.fault_start or len(e.phase)) - 1} · fault {e.fault_start}–{len(e.phase) - 1}\n"
    )
    t0 = time.perf_counter()
    run = pipe.run(e.signal, e.adjacency)
    wall = time.perf_counter() - t0
    shown_phase = None
    for s in run.steps:
        ph = e.phase[s.t]
        interesting = s.regime not in ("warmup", "normal") or ph != shown_phase or s.t % 10 == 0
        if not interesting or (s.regime == "warmup" and s.t > 0):
            continue
        shown_phase = ph
        score = s.mmd2 / s.epsilon
        typ = f"{s.predicted} p={s.probability:.2f}" if s.predicted else ""
        flag = "  ◀ ALERT" if s.alert and s is run.first_alert else ""
        print(f"t={s.t:3d} {ph:6s} MMD²/ε={score:5.2f} {_bar(score)} {s.regime:12s} {typ}{flag}")
        if a.delay:
            time.sleep(a.delay)
        if s.alert and s is run.first_alert:
            break
    out = first_alert_outcome(e, run)
    per_phase = {}
    for ph in ("normal", "drift", "fault"):
        alerts = [s for s in run.steps if e.phase[s.t] == ph and s.alert]
        per_phase[ph] = "ALERT " + str(alerts[0].alert_type) if alerts else "no alert"
    print("\nper phase:", " | ".join(f"{k}: {v}" for k, v in per_phase.items()))
    if out.get("detected"):
        print(
            f"fault detected {out['delay_steps'] * 30} s after onset, type = {out['alert_type']} "
            f"(truth: {e.scenario})"
        )
    lat = run.array("latency_ms")
    print(
        f"processed {len(run.steps)} steps: {wall * 1000:.0f} ms total, "
        f"{np.median(lat):.2f} ms/step median (CPU)"
    )
    if a.out:
        rec = {
            "synthetic": True,
            **out,
            "per_phase": per_phase,
            "epsilon": pipe.epsilon,
            "mmd2": run.array("mmd2").tolist(),
            "regime": run.array("regime").tolist(),
            "phase": e.phase.tolist(),
            "median_step_latency_ms": float(np.median(lat)),
        }
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=1))
    if a.plot:
        print("chart →", plot_timeline(e, run, a.plot))
    return 0


def cmd_benchmark(a) -> int:
    from ewat.early_warning.workflow import benchmark, get_pipeline

    pipe = get_pipeline(Path(a.model), retrain=a.retrain)
    t0 = time.perf_counter()
    res = benchmark(pipe, n_per_scenario=a.episodes, seed0=a.seed0)
    res["setup"]["wall_s"] = round(time.perf_counter() - t0, 1)
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    print(
        f"SYNTHETIC benchmark: {res['setup']['episodes']} episodes, "
        f"seeds {res['setup']['eval_seeds']}"
    )
    print(
        f"  false-alarm rate (harmless drift only) : {pct(res['false_alarm_rate_drift_only'])}"
        f"   (MMD-only baseline: {pct(res['mmd_only_false_alarm_rate_drift_only'])})"
    )
    print(f"  fault detection rate                   : {pct(res['detection_rate'])}")
    print(
        f"  median / p90 detection delay           : "
        f"{res['median_delay_s']} s / {res['p90_delay_s']} s"
    )
    print(f"  fault-type accuracy (detected faults)  : {pct(res['type_accuracy'])}")
    for k, v in res["per_type"].items():
        print(
            f"    {k:20s} detected {pct(v['detection_rate']):>6s}"
            f"  type acc {pct(v['type_accuracy']):>6s}"
            f"  median delay {v['median_delay_s']} s"
        )
    lat = res["step_latency_ms"]
    print(
        f"  per-step latency p50/p95: {lat['p50']:.2f} / {lat['p95']:.2f} ms"
        f"  ({res['setup']['hardware']})"
    )
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("→", a.out)
    return 0


def cmd_predict(a) -> int:
    from ewat.early_warning.serve import WindowRequest, predict

    raw = json.loads(Path(a.window).read_text() if a.window != "-" else sys.stdin.read())
    print(predict(WindowRequest(**raw)).model_dump_json(indent=1))
    return 0


def cmd_example_window(a) -> int:
    from ewat.synthetic import generate_episode

    e = generate_episode(a.scenario, a.seed)
    end = (e.fault_start or e.drift_start) + a.after
    json.dump({"signal": np.round(e.signal[:end], 5).tolist()}, sys.stdout)
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    uvicorn.run("ewat.early_warning.serve:app", host=a.host, port=a.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    from ewat.early_warning.workflow import DEFAULT_MODEL

    ap = argparse.ArgumentParser(
        prog="ewat", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="model path (trained on synthetic data if absent)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="train the typer on seeded synthetic episodes").set_defaults(
        fn=cmd_train
    )

    d = sub.add_parser("demo", help="replay one synthetic episode end to end")
    d.add_argument("--scenario", default="fail_slow_latency")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", help="write the run as JSON")
    d.add_argument("--plot", help="write the timeline PNG")
    d.add_argument("--delay", type=float, default=0.0, help="pause between printed steps (s)")
    d.add_argument("--retrain", action="store_true")
    d.set_defaults(fn=cmd_demo)

    b = sub.add_parser("benchmark", help="seeded synthetic benchmark")
    b.add_argument("--episodes", type=int, default=50, help="episodes per scenario")
    b.add_argument("--seed0", type=int, default=0)
    b.add_argument("--out")
    b.add_argument("--retrain", action="store_true")
    b.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("predict", help="metrics window JSON → drift, fault type, latency")
    p.add_argument("window", help="JSON file with {'signal': [T][6][17]} or '-' for stdin")
    p.set_defaults(fn=cmd_predict)

    w = sub.add_parser("example-window", help="print a synthetic metrics window as JSON")
    w.add_argument("--scenario", default="fail_slow_latency")
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--after", type=int, default=8, help="steps after the fault onset")
    w.set_defaults(fn=cmd_example_window)

    s = sub.add_parser("serve", help="FastAPI endpoint POST /predict")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
