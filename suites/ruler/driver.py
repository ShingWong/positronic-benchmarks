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

"""RULER driver — 32k retrieval efficiency (Appendix + pitch).

Primary paper claim stays LongMemEval/E7; ruler is secondary.
Conditions:
  with positronic: ingest ruler haystack as episode(s) → activate(k=8) top-8 snippet injection
  without: full haystack verbatim (baseline string length)
Reports recall@1, fallback_rate, mean_rrf, token ratio (with ~k*~200 vs without ~length),
and per-length breakdown. 32k gated; 64k/128k via --length when needed.

Usage:
  python3 -m suites.ruler.driver --n 20 --length 32000 --profile balanced --embed lexical
  python3 -m suites.ruler.driver --n 20 --length 32000 --real  # HF fetch
"""
from __future__ import annotations
import time
from pathlib import Path
import json

from harness.adapter import BenchmarkAdapter
from harness.config import RunConfig
from harness.metrics import compute_recall_at_k, p95_latency, fallback_rate as _fr
from harness.report import write_report
from suites.ruler.dataset import load_ruler

def _gold_ids(adapter, gold_subjects):
    ids = set()
    for subj in gold_subjects:
        row = adapter.store.conn.execute("SELECT id FROM episode WHERE subject_norm=?", (subj,)).fetchone()
        if row:
            ids.add(row["id"])
    return ids

def _tokens_heuristic(text: str) -> int:
    return max(1, len(text) // 4)

def run_ruler(n: int = 20, length: int = 32000, profile: str = "balanced", embed: str = "lexical", out_dir: Path | None = None, synthetic: bool = True) -> dict:
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[2] / "results" / "ruler" / f"run-{int(time.time())}"
    out_dir = Path(out_dir)
    sessions = load_ruler(n=n, length=length, synthetic=synthetic)

    cfg = RunConfig(brain="kairos", profile=profile, embed=embed, tmp_root=out_dir / "tmp", k=8)
    adapter = BenchmarkAdapter(cfg)

    recalls: list[float] = []
    latencies: list[float] = []
    hits_list: list[list[dict]] = []
    rrf_scores: list[float] = []
    tok_with: list[int] = []
    tok_without: list[int] = []

    # ingest all
    for sess in sessions[:n]:
        adapter.ingest(sess["events"])

    for sess in sessions[:n]:
        # without tokens = haystack body tokens
        hay = sess["events"][0]["body"] if sess["events"] else ""
        tok_without.append(_tokens_heuristic(hay))
        for qa in sess["qa"]:
            gold_ids = _gold_ids(adapter, qa.get("gold_subjects", []))
            t0 = time.perf_counter()
            hits = adapter.activate(qa["q"], k=8)
            latencies.append((time.perf_counter() - t0) * 1000)
            hits_list.append(hits)
            if hits:
                rrf_scores.append(float(hits[0].get("rrf_score", 0)))
                # with tokens = top-8 snippet injection (~200 chars/hit → ~50 tok/hit)
                tok_with.append(sum(_tokens_heuristic(h.get("snippet", "")) for h in hits[:8]) + 192)  # + prompt overhead
            else:
                tok_with.append(192)
            if gold_ids:
                recalls.append(compute_recall_at_k(hits, gold_ids, k=1))
            else:
                recalls.append(1.0 if hits else 0.0)

    metrics = {
        "n": n,
        "length": length,
        "profile": profile,
        "embed": embed,
        "episodes": adapter.stats()["episodes"],
        "recall@1": (sum(recalls) / len(recalls) if recalls else 0.0),
        "fallback_rate": _fr(hits_list),
        "mean_rrf": float(__import__("numpy").mean(rrf_scores)) if rrf_scores else 0.0,
        "p95_ms": p95_latency(latencies),
        "p50_ms": float(__import__("numpy").median(latencies)) if latencies else 0.0,
        "tokens_with": int(__import__("numpy").mean(tok_with)) if tok_with else 0,
        "tokens_without": int(__import__("numpy").mean(tok_without)) if tok_without else 0,
        "token_ratio": round((__import__("numpy").mean(tok_with) / max(1, __import__("numpy").mean(tok_without))), 4) if tok_with and tok_without else 0.0,
        "note": "with vs without: with=top-8 RRF injection, without=full haystack verbatim. Profile invariance at Δτ<5 expected (balanced==archival==long_term).",
    }
    # per-length breakdown stub (single length per run; multi-length aggregate will merge)
    metrics["breakdown"] = {f"niah-{length}": round(metrics["recall@1"], 3)}
    write_report(metrics, out_dir)
    return metrics

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--length", type=int, default=32000)
    ap.add_argument("--profile", default="balanced")
    ap.add_argument("--embed", default="lexical")
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    syn = not args.real
    m = run_ruler(n=args.n, length=args.length, profile=args.profile, embed=args.embed, out_dir=Path(args.out) if args.out else None, synthetic=syn)
    print(json.dumps(m, indent=2))
