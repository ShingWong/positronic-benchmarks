#!/usr/bin/env python3
"""Positronic dogfood test — run AFTER a compaction to verify post-compaction
health: content marker, recall modes, dossier, latency. Python (no shell-escaping
issues). Run: python3 dogfood-test.py"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Workspace layout is discovered, not hardcoded. Set POSITRONIC_WORKSPACE to
# point at the directory holding positronic-engram/ and .positronic/.
DIR = os.environ.get("POSITRONIC_WORKSPACE", "").rstrip("/") or str(
    Path(__file__).resolve().parent
)
BRAIN = os.environ.get("POSITRONIC_BRAIN", "default")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import paths  # noqa: E402

paths.ensure_memeng()

PASS = 0
FAIL = 0


def check(name, ok, detail):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} — {detail}")


def pai(verb, *args, **kw):
    cmd = ["python3", "-m", "positronic_ai", verb] + list(args)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=DIR)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"raw": r.stdout, "err": r.stderr}


print(f"=== DOGFOOD TEST — post-compaction ({time.strftime('%H:%M:%S')}) ===\n")

# 1. content-carrying marker
print("1) Compaction marker written?")
sql = ("SELECT round(tau,1) tau, json_extract(features_json,'$.body_text') t "
       "FROM episode WHERE kind='consolidation' ORDER BY tau DESC LIMIT 1")
d = pai("query", "--sql", sql, "--json")
rows = d.get("results") or []
if rows:
    t = rows[0].get("t") or ""
    check("content marker", len(t) > 80,
          f"tau={rows[0]['tau']} len={len(t)} {t[:70]!r}")
else:
    check("content marker", False, "no consolidations")

# 2. prune log
print("\n2) Prune report from log")
log = Path.home() / ".cache/positronic/prune.log"
if log.exists():
    tail = log.read_text().strip().splitlines()[-2:]
    check("prune log tail", bool(tail), tail[0][:120] if tail else "empty")

# 3. consolidation-only recall
print("\n3) Recall — consolidation-only mode")
d = pai("recall", "prism benchmark", "--consolidation", "only", "--k", "3", "--json")
h = d.get("results") or []
kinds = {x.get("kind") for x in h}
check("consolidation-only", bool(h) and kinds == {"consolidation"},
      f"hits={len(h)} kinds={kinds}")

# 4. object digest
print("\n4) Recall — object digest for known entity")
d = pai("recall", "opencode plugin", "--json")
obj = d.get("object")
if obj:
    v = obj.get("versions", {})
    check("object digest", True,
          f"{obj['canonical_name']} sightings={v.get('sighting_count')}")
else:
    check("object digest", False, "no object block")

# 5. ask dossier
print("\n5) Ask — dossier surfaces content")
d = pai("ask", "positronic-opencode-plugin", "--json")
s = d.get("sightings") or []
withtext = [x for x in s if (x.get("body_text") or "").strip()]
check("dossier content", bool(withtext), f"sightings={len(s)} with-content={len(withtext)}")

# 6. latency
print("\n6) Latency — sub-second")
from memeng.engine import MemoryEngine
from memeng.store import SQLiteStore
e = MemoryEngine(SQLiteStore(f"{DIR}/.positronic/brains/{BRAIN}/memory.db"))
lat = []
for q in ["alpha beta gamma delta", "epsilon zeta eta theta",
          "what did we decide about the rollout"]:
    t0 = time.perf_counter()
    e.activate({"text": q}, k=8)
    lat.append((time.perf_counter() - t0) * 1000)
    print(f"    {q[:40]:42s} {lat[-1]:6.1f}ms")
check("latency sub-second", max(lat) < 1000, f"max {max(lat):.0f}ms")

# 7. session summary recallable
print("\n7) Session consolidation recallable")
d = pai("recall", "positronic prism research paper", "--consolidation", "only",
        "--k", "2", "--json")
h = d.get("results") or []
check("session summary", bool(h), f"{len(h)} distilled hits")

print(f"\n=== DOGFOOD RESULT: {PASS} pass, {FAIL} fail ===")
sys.exit(1 if FAIL else 0)