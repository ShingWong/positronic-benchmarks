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

def test_adapter_ingest_and_recall(tmp_path):
    from harness.adapter import BenchmarkAdapter
    from harness.config import RunConfig
    cfg = RunConfig(brain="kairos", profile="balanced", embed="lexical", tmp_root=tmp_path)
    adapter = BenchmarkAdapter(cfg)
    adapter.ingest([{"subject": "web2 deploy", "body": "deployed on web2", "persons": ["p_kairos"], "arousal": 0.7}])
    hits = adapter.activate("web2", k=3)
    assert len(hits) > 0
    assert any(h["fallback"] is False for h in hits)
    assert hits[0]["subject"] == "web2 deploy"
    assert hits[0]["rrf_score"] > 0
    assert adapter.stats()["episodes"] == 1

def test_adapter_domain_retention_and_stream_wiring(tmp_path):
    from harness.adapter import BenchmarkAdapter
    from harness.config import RunConfig
    cfg = RunConfig(brain="kairos", profile="balanced", embed="lexical", tmp_root=tmp_path)
    adapter = BenchmarkAdapter(cfg)
    # domain retention must match RunConfig.profile
    did = adapter.store.get_domain_id(adapter._domain)  # type: ignore[attr-defined]
    assert did is not None
    dom = adapter.store.get_domain(did)
    assert dom is not None and dom["retention_profile"] == "balanced"
    # stream must be wired to that domain
    srow = adapter.store.get_stream(adapter._stream)  # type: ignore[attr-defined]
    assert srow is not None and int(srow["domain_id"]) == did
    # isolated tmp DB under tmp_root and distinct per adapter
    cfg2 = RunConfig(brain="kairos", profile="long_term", embed="lexical", tmp_root=tmp_path)
    adapter2 = BenchmarkAdapter(cfg2)
    assert str(tmp_path) in str(adapter.db_path)
    assert str(tmp_path) in str(adapter2.db_path)
    assert str(adapter.db_path) != str(adapter2.db_path)
    # second adapter's domain reflects its own profile
    did2 = adapter2.store.get_domain_id(adapter2._domain)  # type: ignore[attr-defined]
    assert adapter2.store.get_domain(did2)["retention_profile"] == "long_term"  # type: ignore[index]

def test_adapter_prune_scopes_by_domain(tmp_path):
    from harness.adapter import BenchmarkAdapter
    from harness.config import RunConfig
    cfg = RunConfig(brain="kairos", profile="short_term", embed="lexical", tmp_root=tmp_path)
    adapter = BenchmarkAdapter(cfg)
    # 10 synthetic memos spread thinly so some fall past 0.35 threshold at modest Δτ
    import datetime
    base = datetime.datetime(2007, 8, 9, tzinfo=datetime.timezone.utc)
    for i in range(10):
        adapter.ingest([{"subject": f"memo tmp{i:03d}", "body": f"body tmp{i:03d}", "persons": ["p_kairos"], "wall": base + datetime.timedelta(days=i), "arousal": 0.0}])
    tau_now = adapter.store.stream_time(adapter._stream)[0] + 5  # push horizon
    rep = adapter.prune(tau_now=tau_now)
    # short_term with horizon should have at least scanned; archival would not prune
    assert rep.scanned >= 10
