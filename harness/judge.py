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

"""LLM judge for LongMemEval/RULER — OpenRouter meta-llama 3.3-70b / 3.1-405b.

Wu 2410.10813 Table 3 uses GPT-4o + human; we replicate with Llama judge.
Cost: 70b ~$0.002 per 10 q, 405b ~6x more — pilot 70b, final 405b if budget allows.
"""
from __future__ import annotations
import json
import os
import urllib.request

DEFAULT_JUDGE = "meta-llama/llama-3.3-70b-instruct"

JUDGE_PROMPT = """You are a strict factual judge. Decide if the predicted answer correctly answers the question given the gold answer. Allow paraphrase and minor noise but require the core fact to match exactly. For temporal/knowledge-update questions, the updated value must be exact; stale values are wrong.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}

Is the predicted answer correct? Answer with exactly one word: YES or NO."""

def llm_judge(question: str, gold: str, pred: str, model: str = DEFAULT_JUDGE, timeout: int = 60) -> int:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": JUDGE_PROMPT.format(question=question, gold=gold, pred=pred)}],
        "max_tokens": 16,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
        msg = body["choices"][0]["message"]
        text = (msg.get("content") or msg.get("reasoning") or "") or ""
        # some models put answer in reasoning when max_tokens low
        if not text.strip() and msg.get("reasoning_details"):
            text = " ".join((d.get("summary") or d.get("text") or "") for d in msg["reasoning_details"])
        return 1 if text.strip().upper().startswith("YES") else 0
