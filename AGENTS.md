# Benchmarks — AGENTS.md

Harness for credible retention tests. Uses `memeng` via `positronic-engram/engine/src`.

- No PII: tmp `*.db`, `datasets/` and `results/*/` are gitignored. Commit only `metrics.json` summaries if needed.
- Engine pin: `ENGRAM_TAG=v0.2.1`, `import memeng` via `sys.path` to `../../positronic-engram/engine/src` (or `pip install memeng`). **v0.2.1 is the first tag carrying the AGPL migration; v0.2.0 is GPL-3.0-or-later and declares no numpy**, so vector recall silently returns nothing there.
- Profiles: `balanced|archival|long_term|short_term` (`engine.py:48`, E7 55/55/35/7).
- Embed tiers: `lexical` (pilot, 0.5ms FTS), then `local :8090` BGE-M3.
