# LongMemEval suite — pilot (synthetic, lexical)

- **Source:** Wu et al. 2024 `arXiv:2410.10813`, HF `THUDM/LongMemEval` (500 sessions, avg 115k tok, 5 question types).
- **Pilot:** synthetic `n=5|50` via `dataset.py:load_longmemeval(synthetic=True)` — unique tokens, `recall@1` should be `1.0`, `p95 < 2ms` lexical. Real HF fetch is deferred (`--real` gated, needs `datasets` + full corpus, reuses `driver.py` `activate(k=8)` path).
- **Run:** `python3 -m suites.longmemeval.driver --n 50 --embed lexical --synthetic`
- **Outputs:** `results/longmemeval/run-*/metrics.json` + `report.md`
