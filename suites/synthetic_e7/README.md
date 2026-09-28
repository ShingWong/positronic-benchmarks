# Synthetic E7 suite — retention ladder

Replicates `papers/temporal-perception-in-AI/45-pilot-mail-cognition.md:231` E7:
same 55 synthetic events over 78 weeks, weekly `prune(tau_now)` per profile,
final survival `archival 55 / long_term 55 / balanced 35 / short_term 7`
(`engine.py:48` `S_base` 1e6/120/30/6, H15 ladder `0.35→day_token`, `0.05→expired`).

- **Run:** `python3 -m suites.synthetic_e7.driver --n 55` (pilot, ~1s) or `--n 10000 --weeks 78` (full synthetic).
- **Real replay:** `--real` replays a private corpus window via `ingest.to_engine_event` if available, otherwise synthetic fallback. Point `POSITRONIC_PRIVATE_DIR` at the private brain directory to enable it; leave it unset and the suite stays synthetic. The private index is never committed; the harness uses tmp DBs.
- **Outputs:** `results/synthetic_e7/run-*/metrics.json` + `report.md` (finals table + order check).

The sparse cadence (55 events / 78 weeks, 25-week gap) plus `short_term`'s low strength ceiling (S_base=6, max 10.0) drives `retain = exp(−Δτ/strength)` below the 0.05 threshold at ~Wk 36. Episodes are demoted to `week_token` state (not deleted), so `episodes_alive` plateaus at the 7 that remain above threshold. This is spec-conformant behavior, not a bug.
