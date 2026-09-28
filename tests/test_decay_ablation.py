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

def test_decay_ablation_matrix_targets(tmp_path):
    """E1 ablation 2x2: tau vs wall decay on uniform (control) and
    burst-quiet (stress) streams.

    Calibrated S_wall (n=55) so uniform retention matches tau (scale
    parity); the burst-quiet structure must then differentiate the axes:
    tau preserves, wall purges.
    """
    from suites.decay_ablation.driver import run_ablation

    r = run_ablation(tmp_path / "full")
    assert (tmp_path / "full" / "metrics.json").exists()
    cells = r["cells"]

    # control parity: uniform retains ~same on both axes (calibrated S)
    assert cells["uniform_tau"]["alive"] == cells["uniform_wall"]["alive"], \
        f"calibration broken: tau {cells['uniform_tau']['alive']} " \
        f"vs wall {cells['uniform_wall']['alive']}"
    assert cells["uniform_tau"]["retrieval_acc"] == \
        cells["uniform_wall"]["retrieval_acc"]

    # stress divergence: tau preserves burst, wall purges it
    assert cells["burst_quiet_wall"]["alive"] == 0, \
        "wall must purge burst-quiet"
    assert cells["burst_quiet_tau"]["alive"] > \
        cells["burst_quiet_wall"]["alive"] + 3, \
        "tau must preserve burst-quiet"
    assert cells["burst_quiet_tau"]["retrieval_acc"] > 0.8, \
        "tau must keep burst retrievable"
    assert cells["burst_quiet_wall"]["retrieval_acc"] == 0.0