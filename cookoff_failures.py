import json, re, sys, urllib.request, os, time
from pathlib import Path

# Allow running as `python cookoff_frontier.py` from anywhere: put this
# file's own directory on the path so `harness` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import paths  # noqa: E402

RUN = "results/longmemeval/final-context1"
API = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY")

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
            content = ((j.get("choices") or [{}])[0].get("message", {}) or {}).get("content")
            if content:
                return str(content).strip()
            last = "empty content"
        except Exception as e:
            last = str(e)
        time.sleep(2)
    return f"[ERR {last[:60]}]"

def hits_for(idx, q):
    dbs = list(Path(RUN).glob(f"tmp-{idx}/*.db"))
    if not dbs:
        return None
    paths.ensure_memeng()
    from memeng.store import SQLiteStore
    from memeng.engine import MemoryEngine
    return MemoryEngine(SQLiteStore(str(dbs[0]))).activate({"text": q}, k=8, context_window=1)

def judge(ans, gold):
    gl = gold.lower()
    return gl[:12] in ans.lower() or gl.split()[0].lower() in ans.lower()

def main():
    idxs = [int(x) for x in sys.argv[1:]]
    from suites.longmemeval.real_driver import _load_real
    data = _load_real(n=50, offset=0)
    out = {}
    for idx in idxs:
        gold = data[idx]["answer"]; q = data[idx]["question"]
        hits = hits_for(idx, q)
        ctx = "\n---\n".join((h.get("snippet") or "")[:800] for h in hits[:8])
        prompt = f"Answer the question concisely based on the context.\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:"
        print(f"=== Q{idx} gold={gold!r} q={q[:40]!r} ===")
        row = {}
        for model in PANEL:
            ans = call(prompt, model)
            ok = "HIT" if judge(ans, gold) else "MISS"
            row[model.split('/')[-1]] = {"ans": ans[:60], "hit": ok == "HIT"}
            print(f"  {model.split('/')[-1]:22s} {ok}  {ans[:50]!r}")
        out[idx] = row
        print()
    Path("results/longmemeval/cookoff-failures.json").write_text(json.dumps(out, indent=2))
    print("saved results/longmemeval/cookoff-failures.json")

if __name__ == "__main__":
    main()