# Positronic Benchmarks

Pilot validation for MemoryEngine retention — not for paper yet.

## Quick start

```bash
pytest tests/ -q                          # 14 passed + 1 skipped (ruler real gated)
python3 -m suites.longmemeval.driver --n 5 --embed lexical --synthetic
python3 -m suites.longmemeval.driver --n 50 --embed lexical --synthetic
python3 -m suites.synthetic_e7.driver --n 55   # E7 synthetic: 55/55/35/7
python3 -m suites.ruler.driver --n 5 --length 8000 --embed lexical --synthetic  # 32k variant, short Δτ
```

## Suites

- `suites/longmemeval` — LongMemEval pilot (synthetic, lexical) — validates brain works (primary for arXiv)
- `suites/synthetic_e7` — E7 replication (same 55 over 78wks → 55/55/35/7, see `45-pilot-mail-cognition.md:231`)
- `suites/ruler` — RULER 32k retrieval efficiency (Appendix + pitch, with vs without, 1/16th tokens)

Results: `results/{longmemeval|synthetic_e7|ruler}/run-*/metrics.json` + `report.md` (gitignored).
Plan: `positronic-research/docs/superpowers/plans/2026-08-29-positronic-benchmarks.md`.
