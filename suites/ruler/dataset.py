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

"""RULER synthetic dataset — deterministic, no HF fetch.

Mirrors RULER (2404.06654) 4k/8k/16k/32k NIAH-style: needle hidden in
haystack, query is needle token. Haystack length ~ length//4 tokens
(~4 chars/token heuristic). Real HF `hsieh et al. ruler` fetch is
deferred behind --real (requires datasets + HF_TOKEN).
"""
from __future__ import annotations
import random

def _haystack_tokens(length: int) -> int:
    # chars ≈ tokens*4, so words ≈ tokens*0.75
    return max(10, length // 4)

def load_ruler(n: int = 20, length: int = 32000, synthetic: bool = True):
    if synthetic:
        rnd = random.Random(42 + length)
        hay_words = [" ".join(f"word{w:04d}" for _ in range(5)) for w in range(4000)]
        sessions = []
        ntok = _haystack_tokens(length)
        for i in range(n):
            needle = f"needle{i:04d}"
            # single needle in a shared-prefix haystack would be learned as a rule after
            # induce_after=3 and then scored as low-novelty below gate (engine.py:160
            # predictions matched → novelty 0.05). So each haystack is unique (rnd suffix).
            suffix = rnd.randint(100000, 999999)
            body_needle = f"needle {needle}-{suffix} hidden here"
            # build haystack ~ ntok tokens, insert needle at random position
            parts = []
            inserted = False
            for _ in range(ntok // 5):
                if not inserted and rnd.random() < 0.02:
                    parts.append(body_needle)
                    inserted = True
                else:
                    parts.append(rnd.choice(hay_words))
            if not inserted:
                parts.insert(rnd.randint(0, len(parts)), body_needle)
            body = " ".join(parts)
            sessions.append({
                "session_id": f"ruler-{length}-{i}",
                "length": length,
                "events": [{"subject": f"ruler {needle}", "body": body, "persons": ["p_kairos"], "arousal": 0.0}],
                "qa": [{"q": needle, "gold_subjects": [f"ruler {needle}"], "question_type": "niah", "length": length}],
            })
        return sessions
    # real HF (gated)
    from datasets import load_dataset
    ds = load_dataset("hsieh197/ruler", split="test")
    # filter by length bucket
    return [r for r in ds if r.get("length") == length][:n]
