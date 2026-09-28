#!/usr/bin/env bash
# Positronic dogfood test — run AFTER a compaction to verify the brain's
# post-compaction health: content marker, recall modes, dossier, latency.
# Usage: bash dogfood-test.sh

set -u
# Workspace layout is discovered, not hardcoded. Set POSITRONIC_WORKSPACE to
# point at the directory holding positronic-engram/ and .positronic/.
DIR="${POSITRONIC_WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BRAIN="${POSITRONIC_BRAIN:-default}"
ENGRAM_SRC="${POSITRONIC_ENGRAM_SRC:-$DIR/positronic-engram/engine/src}"
export PYTHONPATH="$ENGRAM_SRC${PYTHONPATH:+:$PYTHONPATH}"
# Run from the workspace so `python3 -m positronic_ai` resolves the project
# brain there, matching dogfood-test.py (which passes cwd=DIR). Without this
# the checks below silently inspect whatever brain the invoking directory
# happens to resolve, and report spurious failures.
cd "$DIR" || exit 1
echo "=== DOGFOOD TEST — post-compaction verification ($(date -u +%H:%M:%S)) ==="
echo

echo "1) Compaction marker written? (newest consolidation, must be content-carrying)"
python3 -m positronic_ai query --sql \
  "SELECT round(tau,1) tau, substr(json_extract(features_json,'\$.body_text'),1,100) t, length(json_extract(features_json,'\$.body_text')) ln FROM episode WHERE kind='consolidation' ORDER BY tau DESC LIMIT 1" \
  --json 2>/dev/null | python3 -c "
import sys,json
r=json.load(sys.stdin)['results'][0]
print(f\"  tau={r['tau']} len={r['ln']} text={r['t'][:90]!r}\")
print('  PASS' if r['ln'] > 80 else '  FAIL: marker not content-carrying')
"
echo

echo "2) Prune report from log"
tail -2 ~/.cache/positronic/prune.log | python3 -c "
import sys
for line in sys.stdin:
    print('  ' + line.strip()[:140])
"
echo

echo "3) Recall — consolidation-only mode (distilled memory)"
python3 -m positronic_ai recall "prism benchmark" --consolidation only --k 3 --json 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
h=d['results']
print(f\"  hits={len(h)} | kinds={ {x.get('kind') for x in h} }\")
for x in h[:3]:
    print(f\"    tau={x.get('tau',0):.0f} {str(x.get('subject',''))[:70]}\")
print('  PASS' if h and all(x.get('kind')=='consolidation' for x in h) else '  FAIL')
"
echo

echo "4) Recall — default mode returns object digest for a known entity"
python3 -m positronic_ai recall "opencode plugin" --json 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
obj=d.get('object')
if obj:
    v=obj['versions']
    print(f\"  object={obj['canonical_name']} sightings={v['sighting_count']} tau_span={[round(x,1) for x in v['tau_span']]}\")
    print('  PASS')
else:
    print('  FAIL: no object block')
"
echo

echo "5) Ask — dossier surfaces body_text (subject_norm NULL fallback)"
python3 -m positronic_ai ask "positronic-opencode-plugin" --json 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
s=d.get('sightings',[])
withtext=[x for x in s if (x.get('body_text') or '').strip()]
print(f\"  sightings={len(s)} with-content={len(withtext)}\")
print('  PASS' if withtext else '  FAIL: dossier empty')
"
echo

echo "6) Latency — event-defined interval query, sub-second?"
python3 -u -c "
import time
from memeng.engine import MemoryEngine
from memeng.store import SQLiteStore
e=MemoryEngine(SQLiteStore('$DIR/.positronic/brains/$BRAIN/memory.db'))
qs=['alpha beta gamma delta','epsilon zeta eta theta','what did we decide about the rollout']
for q in qs:
    t0=time.perf_counter(); e.activate({'text':q}, k=8); dt=(time.perf_counter()-t0)*1000
    print(f'  {q[:38]:40s} {dt:6.1f}ms')
" 2>/dev/null
echo

echo "7) Consolidated summary of this session is still recallable"
python3 -m positronic_ai recall "positronic prism research paper" --consolidation only --k 2 --json 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
for x in d['results'][:2]:
    print(f\"  tau={x.get('tau',0):.0f} {str(x.get('subject',''))[:75]}\")
"
echo
echo "=== DOGFOOD TEST COMPLETE ==="