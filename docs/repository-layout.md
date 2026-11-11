# Repository layout

> Moved from the top-level [README](../README.md) on 2026-11-12. The text is unchanged; only heading levels and relative link paths were adjusted.

| path | contents |
|---|---|
| `patches/sglang/` | 121-patch series (0001–0105 = September, 0106–0121 = 2026-10-02) against the base fork + squashed core diff + `apply.sh` + `SERIES.md` + `MODIFICATIONS.md`; `mtp-dump/` = the 9-patch training-data dump hook |
| `patches/flashinfer/` | 10 production patches to flashinfer-python 0.6.17 (01–07 September, 08–10 2026-10-02) + `apply.sh`; rejected ones in `experimental/` |
| `launch/` | `serve-fast.sh` / `serve-local.sh` (portable), `as-measured/` = the scripts used (commands unchanged; paths genericised and comments edited for publication) |
| `results/` | earlier stripped result files (timings, token counts, acceptance; no generated text), `make_tables.py`, `TABLES.md`, per-item quality-audit results; `runs-1002/` = the historical 2026-10-02 round; `runs-1027/genre/` = the 2026-10-27 per-genre runs of the final release (public and private components); the author's other 2026-10-27 run files are not published here |
| `bench/` | the benchmark / profiling harness (`fnbench`), synthetic workloads, analysis and experiment scripts |
| `kernels/` | readable snapshots of our Triton kernels + micro-benchmarks |
| `sim/` | offline n-gram acceptance simulator (no corpora) |
| `tokenmaps/` | how to build a reduced draft vocabulary; historical scripts in `as-used/`; `public/` = 49,152- and 32,768-token maps built only from public data, their source records and preparation scripts (both maps and both source records CC BY-SA 4.0, scripts Apache-2.0); rebuild with `build_hot_vocab.py` |
| `mtp-train/` | MTP draft-head fine-tuning pipeline: dump client, corpus readers, self-generation, trainer, renewal evaluator, write-back (no data or weights); `recipes/public-data/` retrains the head on public data only |
| `docs/` | optimisations, rejected experiments, roofline, measurement method, timeline, reproduction; `lab-notes/` = the original engineering logs |
