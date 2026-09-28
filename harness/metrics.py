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

import numpy as np

def compute_recall_at_k(hits: list[dict], gold_ids: set[str], k: int = 1) -> float:
    topk = {h["episode_id"] for h in hits[:k]}
    return 1.0 if topk & gold_ids else 0.0

def p95_latency(ms: list[float]) -> float:
    return float(np.percentile(ms, 95)) if ms else 0.0

def fallback_rate(hits_list: list[list[dict]]) -> float:
    """Fraction of queries that fell back to recency (no lexical/semantic hit)."""
    if not hits_list:
        return 0.0
    # convention: fallback flag on first hit, or empty hits => fallback
    flagged = sum(1 for h in hits_list if (not h) or (h[0].get("fallback") is True))
    return flagged / len(hits_list)
