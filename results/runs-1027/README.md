# Result files of the 2026-10-27 per-genre runs

`genre/`: the final 2026-10-27 release and the previous best build (2026-10-22) on the 32 BN1 held-out prompts
of `../TABLES.md` section 8 (code-edit, prose-en, prose-ja, agent-loop; 8 each), greedy, thinking on, one request
at a time, one server start per run, screen off. The G runs used public components only (the checkpoint's
original MTP head and the fork's public token maps). The P runs repeat them with the author's private components
(the fine-tuned MTP head and private token maps, not published); nothing else changed. `../TABLES.md` sections 9
and 10 and the README's two 2026-10-27 per-genre tables are generated from these files.

| file | run |
|---|---|
| `G1-release-normal.jsonl` | the release, normal: the launcher's default settings (`CONC=4 bash serve-fast.sh wa`) except port 30000 and prebuilt FlashInfer modules (below) |
| `G2-release-hyperpro.jsonl` | the release, hyper pro (`FLASH_NEXT_EDITION=hyperpro`, locally built ThinkingCap overlay) |
| `G3-previous-normal.jsonl` | the previous best build (2026-10-22), normal, one running request |
| `P1-release-normal-private.jsonl` | G1 with the private head and maps |
| `P2-release-hyperpro-private.jsonl` | G2 with the private head (on the ThinkingCap overlay) and maps |
| `P3-previous-normal-private.jsonl` | G3 with the private head and maps, the files its production entry uses |
| `effective-G*.txt`, `effective-P*.txt` | environment as resolved by the launcher, and the server arguments |
| `conditions.json` | components, start times, screen state before and after each run, the unmeasured streaming warm-up, and the excluded first P2 attempt (stopped when the screen turned on; not published) |
| `table.md` | the per-genre cells |

Prebuilt FlashInfer modules (G1, G2, P1, P2): with `PREBUILT_CACHE=1` the server loaded FlashInfer modules built earlier
instead of compiling them. Each loaded module is byte-identical to an earlier build whose ninja records were kept.
Every C++/CUDA input in those records' dependency logs is byte-identical to this release's assembled FlashInfer
trees, except comment lines (notice headers) in three MoE source files, and the kept generated sources match what
the release generates. Compared with a `build.ninja` written by the release's own FlashInfer code, the MoE
(`fused_moe_120_vt_prologue_v1`, `fused_moe_120_vtoea_prologue_v1`), FP4 GEMM and FP4 quantization modules have
the same flags. Nine smaller modules (two prefill attention, two XQA decode, sampling, top-k, quantization, GEMM
and a utility module) were compiled for `sm_120a`, where a fresh build targets `sm_120f`. The remaining FP16
prefill module differs only in the compiler's name (the same g++). P1 and P2 loaded the same modules as G1 and G2.
G3 and P3 ran the previous build with its own caches.

What was removed before publishing (`../strip_runs.py` and a path rewrite):

- every generated text field and the raw `/metrics` scrapes; only token counts, timings and acceptance remain;
- absolute paths (rewritten to `/home/user/...`, `$REPO` and `<previous-build>`) and the cache, scratch and
  identity variables of the launch environment;
- the paths of the private head and maps (rewritten to `<private-mtp-head>`, `<private-mtp-head-overlay>`,
  `<private-token-map>` and `<private-token-map-32768>`);
- the run labels (rewritten to `release-normal`, `release-hyperpro`, `previous-normal` and the same with `-private`).
