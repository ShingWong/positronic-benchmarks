import json, re, sys, urllib.request, os, time
from pathlib import Path

# Allow running as `python cookoff_frontier.py` from anywhere: put this
# file's own directory on the path so `harness` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import paths  # noqa: E402

RUN = "results/longmemeval/final-context1"
API = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY")

# answer-model sensitivity panel: near-frontier, cheap, diverse
PANEL = [
    "deepseek/deepseek-v4-flash-0731",
    "deepseek/deepseek-v4-pro-0813",
    "z-ai/glm-5.3-flash",
    "qwen/qwen3.8-flash",
    "moonshotai/kimi-k3",
    "google/gemini-3.7-flash",
]

def call(prompt, model):
    payload = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0}).encode()
    req = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}, method="POST")
    for _ in range(3):
        try:
            j = json.load(urllib.request.urlopen(req, timeout=90))
            return (j.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            last = str(e)
    return f"[ERR {last[:60]}]"

def hits_for(idx, q):
    dbs = list(Path(RUN).glob(f"tmp-{idx}/*.db"))
    if not dbs:
        return None
    paths.ensure_memeng()
    from memeng.store import SQLiteStore
    from memeng.engine import MemoryEngine
    s = SQLiteStore(str(dbs[0])); e = MemoryEngine(s)
    return e.activate({"text": q}, k=8, context_window=1)

def judge(ans, gold):
    gl = gold.lower()
    return gl[:12] in ans.lower() or gl.split()[0].lower() in ans.lower()

def main():
    n = int(sys.argv[1]); offset = int(sys.argv[2]); out = sys.argv[3]
    rows = open(f"{RUN}/run.log").read().splitlines()
    from suites.longmemeval.real_driver import _load_real
    data = _load_real(n=50, offset=0)
    results = {}
    done = 0
    for line in rows:
        m = re.search(r"^\[(\d+)/50\] .* acc_with (\d+).*q:'(.+?)'", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if idx < offset:
            continue
        if done >= n:
            break
        done += 1
        q = m.group(3); gold = data[idx]["answer"]
        hits = hits_for(idx, q)
        if not hits:
            print(f"Q{idx} NO-HITS"); continue
        ctx = "\n---\n".join((h.get("snippet") or "")[:800] for h in hits[:8])
        prompt = f"Answer the question concisely based on the context.\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        row = {"gold": gold}
        for model in PANEL:
            ans = call(prompt, model)
            row[model.split("/")[-1]] = {"ans": ans[:80], "hit": judge(ans, gold)}
            time.sleep(0.1)
        results[idx] = row
        hits_n = sum(1 for r in results.values() if any(v.get("hit") for v in r.values() if isinstance(v, dict)))
        print(f"Q{idx} done | running HIT-by-any-model: {hits_n}/{len(results)}")
    Path(out).write_text(json.dumps(results, indent=2))
    # summary
    print("\n=== SUMMARY (exact-substring judge) ===")
    for model in PANEL:
        short = model.split("/")[-1]
        hit = sum(1 for r in results.values() if isinstance(r.get(short), dict) and r[short]["hit"])
        print(f"  {short:22s} {hit}/{len(results)}  ({hit/len(results):.2f})")

if __name__ == "__main__":
    main()