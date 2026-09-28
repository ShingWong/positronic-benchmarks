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

from suites.longmemeval.driver import run_longmemeval

def test_longmemeval_pilot_5_runs(tmp_path):
    metrics = run_longmemeval(n=5, profile="balanced", embed="lexical", out_dir=tmp_path / "out", synthetic=True)
    assert "recall@1" in metrics
    assert "p95_ms" in metrics
    assert "fallback_rate" in metrics
    assert "mean_rrf" in metrics
    assert "breakdown" in metrics
    assert metrics["recall@1"] == 1.0  # synthetic tokens are unique, should hit
    assert metrics["fallback_rate"] == 0.0  # lexical FTS must fire, not recency fallback
    assert metrics["mean_rrf"] > 0
    assert metrics["breakdown"]["synthetic-single"] == 1.0
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "report.md").exists()

def test_longmemeval_pilot_k_truncation_and_rrf(tmp_path):
    from harness.adapter import BenchmarkAdapter
    from harness.config import RunConfig
    # 3 epis with overlapping term; k=1 vs k=2 recall differs
    cfg = RunConfig(brain="kairos", profile="balanced", embed="lexical", tmp_root=tmp_path / "k")
    adapter = BenchmarkAdapter(cfg)
    adapter.ingest([
        {"subject": "alpha bravo", "body": "alpha bravo uniqueA", "persons": ["p_kairos"]},
        {"subject": "alpha charlie", "body": "alpha charlie uniqueB", "persons": ["p_kairos"]},
        {"subject": "other", "body": "other uniqueC", "persons": ["p_kairos"]},
    ])
    hits2 = adapter.activate("alpha", k=2)
    assert len(hits2) >= 1
    assert all(h["fallback"] is False for h in hits2)
    assert hits2[0]["rrf_score"] > 0
    hits1 = adapter.activate("alpha", k=1)
    assert len(hits1) == 1

def test_longmemeval_pilot_fallback_on_missing_term(tmp_path):
    from harness.adapter import BenchmarkAdapter
    from harness.config import RunConfig
    cfg = RunConfig(brain="kairos", profile="balanced", embed="lexical", tmp_root=tmp_path / "fb")
    adapter = BenchmarkAdapter(cfg)
    adapter.ingest([{"subject": "hello world", "body": "hello world", "persons": ["p_kairos"]}])
    hits = adapter.activate("nonexistent_xyz_123", k=3)
    # no lexical hit → fallback path (recency) if episodes exist; assert flag present
    assert all("fallback" in h for h in hits) or hits == []
