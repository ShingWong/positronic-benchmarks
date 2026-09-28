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

def test_synthetic_e7_replicates_55_55_35_11(tmp_path):
    from suites.synthetic_e7.driver import run_synthetic_e7
    m = run_synthetic_e7(n=55, weeks=78, out_dir=tmp_path / "e7", synthetic=True)
    f = m["finals"]
    assert f["archival"]["alive"] == 55
    assert f["long_term"]["alive"] == 55
    assert f["balanced"]["alive"] == 35
    assert f["short_term"]["alive"] == 11
    assert m["profile_order_ok"] is True
    assert (tmp_path / "e7" / "metrics.json").exists()
