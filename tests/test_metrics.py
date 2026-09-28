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

def test_recall_at_k():
    from harness.metrics import compute_recall_at_k
    assert compute_recall_at_k([{"episode_id": "a"}, {"episode_id": "b"}], {"a"}, k=1) == 1.0
    assert compute_recall_at_k([{"episode_id": "a"}, {"episode_id": "b"}], {"c"}, k=2) == 0.0
    assert compute_recall_at_k([], {"a"}, k=1) == 0.0
    # k truncation: gold beyond k must be miss
    assert compute_recall_at_k([{"episode_id": "a"}, {"episode_id": "b"}], {"b"}, k=1) == 0.0
    assert compute_recall_at_k([{"episode_id": "a"}, {"episode_id": "b"}], {"b"}, k=2) == 1.0

def test_p95():
    from harness.metrics import p95_latency
    # 95th of [10,20,30,40,100] with linear interp = 88.0
    v = p95_latency([10, 20, 30, 40, 100])
    assert 80 <= v <= 100
    assert p95_latency([]) == 0.0

def test_fallback_rate():
    from harness.metrics import fallback_rate
    assert fallback_rate([]) == 0.0
    assert fallback_rate([[]]) == 1.0
    assert fallback_rate([[{"fallback": False}]]) == 0.0
    assert fallback_rate([[{"fallback": True}], [{"fallback": False}]]) == 0.5

def test_report_writes_files(tmp_path):
    from harness.report import write_report
    out = write_report({"recall@1": 0.8, "p95_ms": 120}, tmp_path)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "report.md").exists()

def test_report_is_single_source():
    # harness.metrics must NOT re-define write_report (single source is harness.report)
    import harness.metrics as m
    assert not hasattr(m, "write_report")
