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

"""Synthetic LongMemEval-style dataset — no HF fetch, deterministic pilot."""
def load_longmemeval(n: int = 5, synthetic: bool = True):
    if synthetic:
        # each session has 1 event + 1 QA that asks for that event's unique token
        sessions = []
        for i in range(n):
            token = f"pilot{i:03d}"
            sessions.append({
                "session_id": f"s{i}",
                "events": [{"subject": f"fact {token}", "body": f"body {token} with unique token {token}", "persons": ["p_kairos"], "arousal": 0.6}],
                "qa": [{"q": token, "gold_subjects": [f"fact {token}"]}],
            })
        return sessions
    # real HF load (not used in pilot)
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongMemEval", split="test")
    return ds
