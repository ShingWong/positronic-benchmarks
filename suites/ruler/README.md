# RULER suite — 32k retrieval efficiency (Appendix + pitch)

Mirrors `RULER (2404.06654)` NIAH at `4k/8k/16k/32k`. For arXiv this suite is
**secondary** — primary retention stays `longmemeval` + `synthetic_e7`.

- **With vs without positronic:** `with=top-8 RRF injection (~200 chars/hit → ~50 tok/hit)` vs `without=full haystack verbatim`. Reports `recall@1`, `fallback_rate`, `mean_rrf`, `token_ratio` (`with/without`). `32k`→`~0.05` ratio = `1/20` headline, `1/16th` at `32k` `top-8`.
- **Profiles on RULER:** At `Δτ<5` (`S_base 30→1e6`, `engine.py:48`) no `prune` hits `0.35→day_token`, so `balanced==archival==long_term` by design (gated by `tests/test_ruler.py:test_ruler_profiles_invariant_at_short_horizon`). Keep profile comparison for `synthetic_e7` (`55/55/35/7` at `wk78` `τ≈49.9`).
- **Run (synthetic, lexical):** `python3 -m suites.ruler.driver --n 5 --length 8000` (~8s) or `--length 32000` (~15-30s, `72k chars/episode`). Real HF: `--real` (`hsieh197/ruler`, needs `datasets` + `HF_TOKEN`, deferred).
- **Outputs:** `results/ruler/run-*/metrics.json` + `report.md`
