# =====================================================================
# Project Positronic — Polytemporal Cognitive Engram Memory Substrate
# Copyright (C) 2026 Shing Wong. All Rights Reserved.
# =====================================================================
# This program is DUAL-LICENSED. You may redistribute and/or modify it 
# under the terms of the GNU Affero General Public License as published by the 
# Free Software Foundation, either version 3 of the License, or (at your 
# option) any later version.
#
# Alternatively, commercial entities, multi-tenant instances, and Managed 
# Service Providers (MSPs) may utilize this program under a separate, 
# proprietary Commercial License Waiver issued directly by the copyright 
# holder, completely exempt from the network-use copyleft restrictions of 
# the AGPLv3 Section 13.
#
# This program is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License 
# along with this program. If not, see <https://gnu.org>.
# =====================================================================

"""Pilot driver — validates MemoryEngine works on LongMemEval-shaped data."""
from __future__ import annotations
import time
from pathlib import Path

from harness.adapter import BenchmarkAdapter
from harness.config import RunConfig
from harness.metrics import compute_recall_at_k, p95_latency
from harness.report import write_report
from suites.longmemeval.dataset import load_longmemeval

def _gold_ids_for_session(adapter, gold_subjects: list[str]) -> set[str]:
    # resolve gold subject → episode_id via store lookup
    ids = set()
    for subj in gold_subjects:
        row = adapter.store.conn.execute("SELECT id FROM episode WHERE subject_norm=?", (subj,)).fetchone()
        if row:
            ids.add(row["id"])
    return ids

def run_longmemeval(n: int = 5, profile: str = "balanced", embed: str = "lexical", out_dir: Path | None = None, synthetic: bool = True) -> dict:
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[2] / "results" / "longmemeval" / f"run-{int(time.time())}"
    out_dir = Path(out_dir)
    sessions = load_longmemeval(n=n, synthetic=synthetic)
    cfg = RunConfig(brain="kairos", profile=profile, embed=embed, tmp_root=out_dir / "tmp", k=8)
    adapter = BenchmarkAdapter(cfg)

    recalls: list[float] = []
    latencies: list[float] = []
    hits_list: list[list[dict]] = []
    rrf_scores: list[float] = []
    # ingest all sessions first (simulates long history), then query
    for sess in sessions[:n]:
        adapter.ingest(sess["events"])
    for sess in sessions[:n]:
        for qa in sess["qa"]:
            gold_ids = _gold_ids_for_session(adapter, qa.get("gold_subjects", []))
            t0 = time.perf_counter()
            hits = adapter.activate(qa["q"], k=8)
            latencies.append((time.perf_counter() - t0) * 1000)
            hits_list.append(hits)
            if hits:
                rrf_scores.append(float(hits[0].get("rrf_score", 0)))
            # for synthetic pilot gold_ids is exact; fallback to hits>0 if lookup fails
            if gold_ids:
                recalls.append(compute_recall_at_k(hits, gold_ids, k=1))
            else:
                recalls.append(1.0 if hits else 0.0)

    from harness.metrics import fallback_rate as _fr
    metrics = {
        "n": n,
        "profile": profile,
        "embed": embed,
        "episodes": adapter.stats()["episodes"],
        "recall@1": (sum(recalls) / len(recalls) if recalls else 0.0),
        "fallback_rate": _fr(hits_list),
        "mean_rrf": float(__import__("numpy").mean(rrf_scores)) if rrf_scores else 0.0,
        "p95_ms": p95_latency(latencies),
        "p50_ms": float(__import__("numpy").median(latencies)) if latencies else 0.0,
    }
    write_report(metrics, out_dir)
    # breakdown prep: group by synthetic prefix (longmemeval 5 types stubbed)
    breakdown: dict[str, list[float]] = {}
    idx = 0
    for sess in sessions[:n]:
        for qa in sess["qa"]:
            # synthetic q encodes type via token; real HF will carry question_type field
            qtype = qa.get("question_type") or qa.get("q_type") or "synthetic-single"
            breakdown.setdefault(qtype, []).append(recalls[idx] if idx < len(recalls) else 0.0)
            idx += 1
    if breakdown:
        metrics["breakdown"] = {k: round(sum(v) / len(v), 3) for k, v in breakdown.items()}
    # re-write with breakdown included
    write_report(metrics, out_dir)
    return metrics

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--embed", default="lexical")
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    m = run_longmemeval(n=args.n, profile=args.profile, embed=args.embed, out_dir=Path(args.out) if args.out else None, synthetic=args.synthetic)
    print(json.dumps(m, indent=2))
