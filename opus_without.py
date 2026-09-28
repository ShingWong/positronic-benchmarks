import json, sys, urllib.request, os, time
from pathlib import Path

# Allow running as `python cookoff_frontier.py` from anywhere: put this
# file's own directory on the path so `harness` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import paths  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY")

MODEL = "anthropic/claude-opus-5"

def call(prompt):
    payload = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64, "temperature": 0}).encode()
    req = urllib.request.Request(API, data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}, method="POST")
    for _ in range(4):
        try:
            j = json.load(urllib.request.urlopen(req, timeout=120))
            content = ((j.get("choices") or [{}])[0].get("message", {}) or {}).get("content")
            if content:
                return str(content).strip()
            last = "empty"
        except Exception as e:
            last = str(e)
        time.sleep(2)
    return f"[ERR {last[:60]}]"

def judge(ans, gold):
    gl = gold.lower()
    return gl[:12] in ans.lower() or gl.split()[0].lower() in ans.lower()

def main():
    paths.ensure_memeng()
    from suites.longmemeval.real_driver import _load_real, _session_to_event
    idxs = [int(x) for x in sys.argv[1:]]
    data = _load_real(n=50, offset=0)
    for idx in idxs:
        row = data[idx]
        q, gold = row["question"], row["answer"]
        hay = row["haystack_sessions"]
        ids = row["haystack_session_ids"]
        full_hay = "\n\n".join(
            _session_to_event(s, sid)["body"][:1200]
            for s, sid in zip(hay[:30], ids[:30]))
        full_hay = full_hay[:30000]
        prompt = (f"Answer the question concisely based on the full chat history.\n\n"
                  f"History:\n{full_hay}\n\nQuestion: {q}\nAnswer:")
        ans = call(prompt)
        ok = "HIT" if judge(ans, gold) else "MISS"
        print(f"idx{idx}: gold={gold!r} -> {ok}  {ans[:60]!r}")

if __name__ == "__main__":
    main()