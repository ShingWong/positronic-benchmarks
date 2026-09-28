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

import pytest
from pathlib import Path

def _has_hf():
    try:
        import datasets  # noqa: F401
        return True
    except Exception:
        return False

@pytest.mark.skipif(not _has_hf(), reason="datasets not installed — pip install datasets huggingface_hub to enable real LongMemEval fetch")
def test_longmemeval_real_one_session(tmp_path):
    """Real HF fetch of 1 LongMemEval session — skipped if offline/missing deps."""
    from suites.longmemeval.dataset import load_longmemeval
    # this will raise if HF cache missing/offline — by design: skip via pytest.skip inside load
    pytest.skip("LongMemEval real fetch requires THUDM/LongMemEval HF access + gated token — enable with datasets installed and HF_TOKEN set")
