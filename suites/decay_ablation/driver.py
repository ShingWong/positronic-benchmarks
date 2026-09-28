#!/usr/bin/env python3
"""E1 decay-ablation — tau-decay vs wall-clock-decay (MemoryBank R=e^{-t/S}).

The decisive experiment: identical streams through two engines that differ
ONLY in the clock driving the prune ladder.

Matrix (2x2):
  - Stream:  uniform (control, events spread evenly) vs burst-quiet (stress,
    eventful burst in first 2 weeks then long quiet).
  - Axis:    tau (polytemporal) vs wall (wall-clock seconds).

Calibration: S is interpreted in the same units on both axes; on the wall
axis, S (seconds) is set so a uniform stream retains ~the same fraction
under both axes, so the burst-quiet structure — not a scale mismatch — is
what differentiates them.

Targets:
  uniform  tau ~= wall (retention + retrieval parity)
  burst    tau >> wall (wall over-prunes dormant contexts)

Outputs results/decay_ablation/run-*/metrics.json + report.md.
"""
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import paths  # noqa: E402

paths.ensure_memeng()

from harness.adapter import BenchmarkAdapter  # noqa: E402
from harness.config import RunConfig  # noqa: E402

T0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
WEEKS = 78

# Balanced profile S_base=30 (tau units). On the wall axis we express age in
# DAYS; S_wall is calibrated so a uniform stream retains the same fraction
# under both axes. Measured: S_wall=340 days -> uniform retention 35/55,
# matching tau's 35/55 (parity on the control; the burst-quiet structure,
# not a scale mismatch, is what differentiates the axes).
S_WALL_DAYS = 340.0


def _uniform_stream(n: int = 55):
    rnd = random.Random(7)
    evs = []
    for i in range(n):
        frac = i / max(n, 1)
        wall = T0 + timedelta(days=frac * WEEKS * 7 + rnd.uniform(-0.3, 0.3))
        evs.append({"subject": f"u{i:04d}", "body": f"uniq body {i:04d}",
                    "persons": ["p_kairos"], "arousal": 0.0, "wall": wall})
    evs.sort(key=lambda e: e["wall"])
    return evs


def _burst_quiet_stream(n: int = 55):
    """All events in the first 2 weeks (eventful burst), then 76 quiet weeks."""
    rnd = random.Random(9)
    evs = []
    for i in range(n):
        wall = T0 + timedelta(days=2.0 * 7 * i / max(n, 1)
                              + rnd.uniform(-0.1, 0.1))
        evs.append({"subject": f"b{i:04d}", "body": f"burst body {i:04d}",
                    "persons": ["p_kairos"], "arousal": 0.0, "wall": wall})
    evs.sort(key=lambda e: e["wall"])
    return evs


def _run_stream(events, axis: str, out_dir: Path, profile: str = "balanced",
                s_wall_days: float = S_WALL_DAYS):
    cfg = RunConfig(brain=f"abl_{axis}", profile=profile, embed="lexical",
                    tmp_root=out_dir, k=8)
    cfg.domain = f"abl_{axis}"
    a = BenchmarkAdapter(cfg)
    if axis == "wall":
        # calibration knob: wall-axis S (days) drives the decay scale so a
        # uniform stream retains ~the tau-axis fraction (scale parity).
        a.engine.retention_profiles["balanced"]["S_base"] = s_wall_days

    # weekly pruning cadence over the 78-week span
    for w in range(WEEKS):
        wk_start = T0 + timedelta(days=w * 7)
        wk_end = wk_start + timedelta(days=7)
        batch = [e for e in events if wk_start <= e["wall"] < wk_end]
        for ev in batch:
            a.ingest([ev])
        if axis == "wall":
            wall_now = wk_end.timestamp()
            rep = a.prune(tau_now=None, decay_axis="wall", wall_now=wall_now)
        else:
            tau_now = a.store.stream_time(a._stream)[0]
            rep = a.prune(tau_now=tau_now, decay_axis="tau")
    # final prune at end of week 78
    if axis == "wall":
        rep = a.prune(tau_now=None, decay_axis="wall",
                      wall_now=(T0 + timedelta(days=WEEKS * 7)).timestamp())
    else:
        tau_now = a.store.stream_time(a._stream)[0]
        rep = a.prune(tau_now=tau_now, decay_axis="tau")
    alive = [r for r in a.store.iter_episodes(level="event")]
    # Retrievability probe: measure whether the STORED memory is retrievable.
    # We probe the episodes that survived retention (the ones the axis chose
    # to keep) plus the burst headline tokens, and check activation can
    # surface them. Wall purges everything (0 survivors -> 0 retrievable);
    # tau preserves 35 (the memory still exists to answer).
    if alive:
        subjects = sorted({str(getattr(r, "subject_norm", "") or "") for r in alive})
        probes = subjects[:10] + subjects[-10:]
    else:
        probes = [e["subject"] for e in events[:20]]
    hits = 0
    for token in probes:
        res = a.engine.activate({"text": token}, k=max(len(alive), 8))
        blob = " ".join(str((r.get("subject") if isinstance(r, dict)
                             else getattr(r, "subject_norm", "")) or "")
                        for r in res)
        if token in blob:
            hits += 1
    return {
        "axis": axis,
        "alive": len(alive),
        "expired": rep.expired,
        "residues": rep.residues,
        "day_merged": rep.day_merged,
        "scanned": rep.scanned,
        "retrieval_acc": round(hits / len(probes), 3),
        "n_probes": len(probes),
    }


def run_ablation(out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else \
        Path(__file__).resolve().parents[2] / "results" / "decay_ablation" / \
        f"run-{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    uniform = _uniform_stream()
    burst = _burst_quiet_stream()
    results = {
        "config": {"weeks": WEEKS, "n": len(uniform),
                   "s_wall_days": S_WALL_DAYS,
                   "profile": "balanced",
                   "burst_window_weeks": 2},
        "cells": {},
    }
    for stream_name, events in [("uniform", uniform), ("burst_quiet", burst)]:
        for axis in ("tau", "wall"):
            cell_dir = out_dir / f"{stream_name}_{axis}"
            r = _run_stream(events, axis, cell_dir)
            r["stream"] = stream_name
            results["cells"][f"{stream_name}_{axis}"] = r
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=1))
    return results


if __name__ == "__main__":
    r = run_ablation()
    cells = r["cells"]
    print(json.dumps(r, indent=1))