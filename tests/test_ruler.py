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

from suites.ruler.driver import run_ruler

def test_ruler_32k_pilot_synthetic(tmp_path):
    m = run_ruler(n=5, length=32000, profile="balanced", embed="lexical", out_dir=tmp_path / "ruler", synthetic=True)
    assert m["recall@1"] == 1.0
    assert m["fallback_rate"] == 0.0
    assert m["mean_rrf"] > 0
    assert m["tokens_with"] < m["tokens_without"]  # efficiency claim: with < without
    assert m["token_ratio"] < 0.5
    assert m["length"] == 32000
    assert "niah-32000" in m["breakdown"]
    assert (tmp_path / "ruler" / "metrics.json").exists()

def test_ruler_profiles_invariant_at_short_horizon(tmp_path):
    from suites.ruler.driver import run_ruler as run
    m_bal = run(n=5, length=8000, profile="balanced", embed="lexical", out_dir=tmp_path / "bal", synthetic=True)
    m_arch = run(n=5, length=8000, profile="archival", embed="lexical", out_dir=tmp_path / "arc", synthetic=True)
    # at Δτ<5, same events → same recall; profiles should not diverge here
    assert m_bal["recall@1"] == m_arch["recall@1"] == 1.0
    assert m_bal["fallback_rate"] == m_arch["fallback_rate"] == 0.0
