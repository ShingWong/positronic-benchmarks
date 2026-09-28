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

"""E7 replication — same life, four forgetting policies.

Uses `harness.adapter.BenchmarkAdapter` (isolated tmp DBs, per-domain
retention) to feed the SAME chronological event stream into four profiles
(archival/long_term/balanced/short_term, engine.py:48) with the same weekly
prune cadence as the original private E7 experiment. Validates the
application-knob thesis: identical experience → different memory by policy.

Two modes:
  synthetic (default, no PII) — deterministic 10k-or-55 synthetic episodes
    spread over 78 weeks, unique tokens so gate stays ~0.9 novel.
  replay (opt-in, requires a private corpus you supply) — replays a real
    55-message window when POSITRONIC_PRIVATE_DIR points at a directory
    containing state/index.jsonl; otherwise skips.

Pilot gates with n=55 synthetic; full run with n=10000 synthetic.
"""
from __future__ import annotations
import time
import json
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone

from harness.adapter import BenchmarkAdapter
from harness.config import RunConfig
from harness.report import write_report

PROFILES = ["archival", "long_term", "balanced", "short_term"]
WEEKS = 78
T0 = datetime(2007, 8, 9, tzinfo=timezone.utc)

def _synthetic_events(n: int = 55, weeks: int = WEEKS) -> list[dict]:
    """Deterministic synthetic stream mimicking E7's 55 msgs over 78 weeks.
    Each event has a unique subject so novelty stays high and tau advances
    ~0.9 per event (so ladder sees Δτ comparable to E7). Arousal left at 0
    to match the real E7's `ingest.to_engine_event(..., brain=None)` which
    yields S=S_base (archival 1e6, long_term 120, balanced 30, short_term 6).
    """
    rnd = random.Random(42)
    events = []
    for i in range(n):
        frac = i / max(n, 1)
        wall = T0 + timedelta(days=frac * weeks * 7 + rnd.uniform(-0.4, 0.4))
        token = f"syn{i:05d}"
        arousal = 0.0
        events.append({"subject": f"memo {token}", "body": f"body {token} with unique token {token}", "persons": ["p_kairos"], "arousal": arousal, "wall": wall})
    events.sort(key=lambda e: e["wall"])
    return events

def _wall_for_week(w: int) -> datetime:
    return T0 + timedelta(days=w * 7)

def run_synthetic_e7(n: int = 55, weeks: int = WEEKS, out_dir: Path | None = None, synthetic: bool = True) -> dict:
    """Run E7 over n events / weeks. Returns metrics dict and writes report."""
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[2] / "results" / "synthetic_e7" / f"run-{int(time.time())}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if synthetic:
        events = _synthetic_events(n=n, weeks=weeks)
    else:
        # try real replay; fall back to synthetic if index missing
        try:
            from pathlib import Path as _P
            import os
            import sys
            # Optional real replay from a private corpus. Override with
            # POSITRONIC_PRIVATE_DIR; unset means synthetic-only, which is the
            # correct default for anyone without access to the private repo.
            _priv = os.environ.get("POSITRONIC_PRIVATE_DIR", "")
            pp = _P(_priv) if _priv else None
            if pp and (pp / "state/index.jsonl").exists():
                sys.path.insert(0, str(pp.parent / "positronic-engram/engine/src"))
                sys.path.insert(0, str(pp))
                import pull as P, ingest
                idx = P.load_index()
                t0 = datetime.fromisoformat(idx[0]["date"])
                t_end = t0 + timedelta(weeks=weeks)
                window = [m for m in idx if m.get("epoch") and datetime.fromisoformat(m["date"]) < t_end][:n]
                events = [dict(subject=ingest.to_engine_event(m, None).features.get("subject_norm", ""), body=ingest.to_engine_event(m, None).features.get("body_text", ""), persons=ingest.to_engine_event(m, None).persons or ["p_kairos"], arousal=float(ingest.to_engine_event(m, None).features.get("arousal", 0.5)), wall=datetime.fromisoformat(m["date"])) for m in window]
            else:
                events = _synthetic_events(n=n, weeks=weeks)
        except Exception:
            events = _synthetic_events(n=n, weeks=weeks)

    # bucket events by week
    weekly_buckets: list[list[dict]] = [[] for _ in range(weeks)]
    for ev in events:
        w = int((ev["wall"] - T0).total_seconds() // (7 * 86400))
        w = max(0, min(weeks - 1, w))
        weekly_buckets[w].append(ev)

    finals: dict[str, dict] = {}
    for prof in PROFILES:
        cfg = RunConfig(brain=f"e7_{prof}", profile=prof, embed="lexical", tmp_root=out_dir / f"tmp-{prof}", k=8)
        # use same domain as brain so prune scopes correctly
        cfg.domain = f"e7_{prof}"
        adapter = BenchmarkAdapter(cfg)
        for w in range(weeks):
            for ev in weekly_buckets[w]:
                adapter.ingest([ev])
            # weekly consolidation pass (same cadence as E7)
            tau_now = adapter.store.stream_time(adapter._stream)[0]  # type: ignore[attr-defined]
            adapter.prune(tau_now=tau_now)
        # final prune
        tau_now = adapter.store.stream_time(adapter._stream)[0]  # type: ignore[attr-defined]
        rep = adapter.prune(tau_now=tau_now)
        # Post-prune: only surviving level="event" episodes (demoted
        # episodes are re-levelled to day_token/week_token, not deleted,
        # so this count shrinks with demotion while expired stays 0).
        alive = len(adapter.store.iter_episodes(level="event"))
        finals[prof] = {"alive": alive, "expired": rep.expired, "day_merged": rep.day_merged, "scanned": rep.scanned}

    # control: object formation would be identical if we formed objects (synthetic uses entity extraction only)
    # for synthetic pilot we just report that gate produced comparable episode_counts pre-prune
    metrics = {
        "n": n,
        "weeks": weeks,
        "synthetic": synthetic,
        "finals": finals,
        # E7 headline numbers from 45-pilot-mail-cognition.md: archival 55, long_term 55, balanced 35, short_term 7
        "profile_order_ok": finals["archival"]["alive"] >= finals["long_term"]["alive"] >= finals["balanced"]["alive"] >= finals["short_term"]["alive"],
    }
    write_report(metrics, out_dir)
    # also write a human markdown supplement
    md_path = out_dir / "report.md"
    extra = "\n\n## E7 final survival (ask: does policy change forgetting?)\n\n| profile | alive | expired | day_merged |\n|---|---:|---:|---:|\n"
    for p in PROFILES:
        f = finals[p]
        extra += f"| {p} | {f['alive']} | {f['expired']} | {f['day_merged']} |\n"
    extra += f"\nOrder archival≥long_term≥balanced≥short_term: {metrics['profile_order_ok']}\n"
    md_path.write_text(md_path.read_text() + extra)
    return metrics

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=55)
    ap.add_argument("--weeks", type=int, default=78)
    ap.add_argument("--out", default=None)
    ap.add_argument("--real", action="store_true", help="try replaying private index instead of synthetic")
    args = ap.parse_args()
    m = run_synthetic_e7(n=args.n, weeks=args.weeks, out_dir=Path(args.out) if args.out else None, synthetic=not args.real)
    print(json.dumps(m, indent=2))
