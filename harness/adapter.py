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

from __future__ import annotations
import sys
from pathlib import Path
import tempfile
import uuid
from datetime import datetime, timezone

from harness import paths

paths.ensure_memeng()

from memeng.store import SQLiteStore
from memeng.engine import MemoryEngine
from memeng.models import Event

class BenchmarkAdapter:
    def __init__(self, cfg):
        from pathlib import Path as _P
        root = _P(cfg.tmp_root) if cfg.tmp_root else _P(tempfile.mkdtemp())
        root.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.db_path = root / f"{cfg.brain}-{uuid.uuid4().hex[:6]}.db"
        self.store = SQLiteStore(str(self.db_path))
        self.engine = MemoryEngine(self.store, config={"threshold": 0.55})
        self.engine.init_database()
        # federation parity: domain retention profile controls prune ladder
        domain = getattr(cfg, "domain", None) or cfg.brain
        try:
            self.engine.register_domain(domain, retention_profile=cfg.profile)
        except ValueError:
            pass
        stream = f"positronic:{cfg.brain}"
        try:
            self.engine.attach_stream(stream, domain)
        except Exception:
            pass
        self._stream = stream
        self._domain = domain
        self._embedder = None
        if getattr(cfg, "embed", "lexical") == "local":
            local_url = getattr(cfg, "local_url", "http://127.0.0.1:8090")
            def _emb(text: str) -> list[float]:
                import json, urllib.request
                # bge-m3 Q8 physical limit c=8192 but per-input 512 tokens ~1200 chars; truncate to 1200 to avoid HTTP 500 (see task-3 report batch32 500)
                t = text[:1200]
                payload = json.dumps({"input": t}).encode()
                req = urllib.request.Request(f"{local_url}/v1/embeddings", data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=10) as r:
                    j = json.loads(r.read().decode())
                    return j["data"][0]["embedding"]
            self._embedder = _emb
            self.engine.bind_embedder(_emb)

    def ingest(self, events: list[dict]):
        # batch embeds when local (network round-trip is the bottleneck at 500 msgs ~0.8s each)
        if self._embedder is not None and len(events) > 1:
            try:
                import json, urllib.request
                local_url = getattr(self.cfg, "local_url", "http://127.0.0.1:8090")
                # chunk batch to avoid 512-token limit and huge payloads
                BATCH = 8
                vecs_all: list[list[float]] = []
                for off in range(0, len(events), BATCH):
                    chunk = events[off:off+BATCH]
                    texts = [(ev["subject"] + " " + ev["body"])[:1200] for ev in chunk]
                    payload = json.dumps({"input": texts}).encode()
                    req = urllib.request.Request(f"{local_url}/v1/embeddings", data=payload, headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=30) as r:
                        j = json.loads(r.read().decode())
                        vecs_all.extend([d["embedding"] for d in j["data"]])
                # ingest with precomputed vecs
                for ev, vec in zip(events, vecs_all):
                    w = ev.get("wall") or datetime.now(timezone.utc)
                    res = self.engine.new_event(Event(
                        stream=ev.get("stream") or self._stream,
                        kind="message",
                        persons=ev.get("persons", ["p_kairos"]),
                        wall=w,
                        features={"subject_norm": ev["subject"], "body_text": ev["body"], "arousal": ev.get("arousal", 0.5)},
                    ))
                    if res.episode_id and res.verdict.encoded:
                        try:
                            self.store.set_embedding(str(res.episode_id), vec)
                            self.engine.vec_index.add(str(res.episode_id), vec)
                        except Exception:
                            pass
                return
            except Exception:
                pass
        for ev in events:
            w = ev.get("wall") or datetime.now(timezone.utc)
            self.engine.new_event(Event(
                stream=ev.get("stream") or self._stream,
                kind="message",
                persons=ev.get("persons", ["p_kairos"]),
                wall=w,
                features={"subject_norm": ev["subject"], "body_text": ev["body"], "arousal": ev.get("arousal", 0.5)},
            ))

    def activate(self, text: str, k: int | None = None,
                 context_window: int = 0):
        if self._embedder is not None:
            try:
                vec = self._embedder(text)
                return self.engine.activate(
                    {"text": text, "embedding": vec}, k=k or self.cfg.k,
                    context_window=context_window)
            except Exception:
                pass
        return self.engine.activate(
            {"text": text}, k=k or self.cfg.k,
            context_window=context_window)

    def prune(self, tau_now: float | None = None, *, decay_axis: str = "tau",
              wall_now: float | None = None):
        return self.engine.prune(tau_now=tau_now, domain=self._domain,
                                 decay_axis=decay_axis, wall_now=wall_now)

    def stats(self):
        c = self.store.conn.execute("SELECT COUNT(*) c FROM episode").fetchone()["c"]
        return {"episodes": c, "db": str(self.db_path)}
