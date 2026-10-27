# Qwen3.8-Flash-Next: fast single-stream decode on one RTX PRO 6000 Blackwell Max-Q

[日本語版 README](README.ja.md)

This repository documents the historical September and 2026-10-02 work that took
single-request (batch size 1) decoding of
**Qwen3.8-Flash-Next** (NVFP4 checkpoint) on **one NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB, 300–325 W)**
from about **130–210 tokens/s** to **about 260–670 tokens/s**, depending on the kind of text being
generated — and what did *not* work along the way. It contains the patch series, the FlashInfer
patches, the launch scripts, the benchmark harness, stripped result files for the earlier
September and 2026-10-02 runs, and the engineering notes. Of the 2026-10-27 results, only the
per-genre runs are published here (`results/runs-1027/genre/`); the author's other run files are not.
The new four-request release is described in the
[2026-10-27 update](#2026-10-27-update) below.

> **Built on [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000).**
> The Flash-Next model code in that fork (QSA, hyper-connections, PLE, native MTP, the memory pool)
> is an adapted import of the unmerged [SGLang](https://github.com/sgl-project/sglang) PR
> [#36497](https://github.com/sgl-project/sglang/pull/36497) (Qwen 3.8 Flash-Next; Qiaolin-Yu with
> co-authors ch-wan, ispobock, JustinTong0323, samuellees, YAMY1234, yhyang201, yizhang2077 and
> zijiexia), with the GDN MTP state adapted from the unmerged PR
> [#30967](https://github.com/sgl-project/sglang/pull/30967) (RecoverSSM). John Pezzulli's work in
> that fork is the SM120 integration and qualification: running it on this GPU, recovery CUDA
> graphs, HiCache/NIXL persistence and the validation behind it (see the fork's `CHANGES.md` and
> `PROVENANCE.md`). Our work is a
> 121-commit patch series on top of its `pennyroyal-main-sm120-final` branch (commit `16e5682aad`)
> plus ten patches to FlashInfer 0.6.17 (105 and seven of them for the September results below, the
> rest for the 2026-10-02 build). Please credit and read that project first.

## 2026-10-27 update

The new release is on the fork's [`flash-next-fast` branch](https://github.com/aiueo52/sglang-rtxpro6000/tree/d21289fb923ac65b7b913b73c6e32a2fa2601e7b), based on jpezzulli/sglang-rtxpro6000 at 16e5682 (2026-08-27) plus the flash-next-fast changes by aiueo52 and contributors. It adds support for four concurrent requests with long context while retaining the single-request optimizations. Performance comparisons apply only to the measured configurations below. It ships in two editions:

| edition | weights | what is different |
|---|---|---|
| `normal` (default) | Qwen3.8-Flash-Next NVFP4 | — |
| `hyperpro` | ThinkingCap (Qwen3.8-Flash-Next NVFP4) | OEA on (an approximate MoE path; outputs change slightly, quality checked below) |

### What changed

- **Up to 4 concurrent requests, 262,144-token context each**, with 7 GB of VRAM left for the display (`DISPLAY_RESERVE_GB=7`).
- **Hierarchical prefix cache in host RAM (32 GB at the default `CONC=4`)**, so conversations that are not running keep their prefix and come back without a full re-prefill.
- **MTP speculative decoding with an adaptive draft length**. The default table is calibrated for 4 concurrent requests.
- Single-request optimizations including PLE embedding offload, fused verify and optimized MoE/draft kernels.
- FlashInfer 0.6.17 changes are distributed as overlays, assembled locally into normal and OEA variants rather than shipping full dependency trees.
- The launcher now requires an admin key for admin/management routes by default, including tensor-based LoRA loading, with diagnostic readback redaction. Inference routes keep upstream authentication; the security limits are below.
- Removed for licensing reasons: the MUSA `fused_add_rms_norm_kernel.mu` and the Hunyuan3D `rasterizer.cpp`, `rasterizer_gpu.cu`, `rasterizer.h` and `mesh_processor.cpp`. The MUSA fused-add-RMSNorm kernel and Hunyuan3D rasterizer/mesh extensions are therefore unavailable in this release.

### Requirements and how to run

- An RTX PRO 6000 Blackwell (96 GB). Other sm120 cards are untested.
- The base environment from the upstream [`BUILD.md`](https://github.com/jpezzulli/sglang-rtxpro6000/blob/16e5682aad6e3335f38ec8d5711176278f12a086/BUILD.md): Python 3.12, PyTorch 2.13.0+cu130, FlashInfer 0.6.17 and a CUDA 13 toolkit with `nvcc`. The first start compiles kernels, so `nvcc` and a host C++ compiler are required.
- Measured on: Ubuntu 24.04 (Linux 7.0), NVIDIA driver 595.91.07, RTX PRO 6000 Blackwell Max-Q Workstation Edition, nvcc 13.3.73, g++ 13.3.0.
- **The first start compiles kernels with nvcc.** On the machine used by aiueo52 and contributors (12 CPU threads, MAX_JOBS=2) the first start compiles 14 modules: about 47 min, peak RAM about 20 GiB (the largest, one MoE module, about 27 min and 16 GiB). Part of this build shared its CPUs with other work, so treat the time as an upper bound. Later starts reuse the cache.
  - Hyper pro compiles its own MoE module in place of that one. We have not timed it: our 90-minute test budget ran out about 9 minutes into this module, at 11 of 97 steps.
  - SGLang's own JIT kernels (three small ones) are built on the first start, before the server reports ready: 13–17 s in total on the machine used by aiueo52 and contributors. They need `ninja` on `PATH` (it is installed in the venv; run with the venv active). Without it the server stops at its first request.
  - Not included: kernels that compile on first use at run time (CuTe-DSL NVFP4 quantize, Triton) and CUDA graph capture. These need the GPU and were not timed separately.
- Host RAM: about 100 GiB for the server process with the defaults (measured peak 100.5 GiB: about 67 GiB for the model and its offloads, plus the 32 GB HiCache host pool). Set `FLASH_NEXT_HICACHE_GB` lower to reduce the pool at `CONC=4`. Measured on a 256 GB machine.
- Weights are not included (see Assets below). Normal uses RadixArk/Qwen3.8-Flash-Next-NVFP4; hyper pro uses a local ThinkingCap overlay on that model, built with `tools/thinkingcap_overlay/overlay.py` (16 GB of new files; the rest are links to the original).

Clone the fork's branch, then build and activate a venv following the upstream `BUILD.md` linked above. Only that venv plus `serve-fast.sh` is tested; **installing this branch with pip (for example `pip install ./python`) is not tested or supported**.

```bash
git clone --branch flash-next-fast https://github.com/aiueo52/sglang-rtxpro6000.git
cd sglang-rtxpro6000
# Build and activate the venv from the upstream BUILD.md before continuing.
python3 -m pip install flashinfer-python==0.6.17 ninja
python3 tools/assemble_flashinfer.py
export TARGET_MODEL=/path/to/Qwen3.8-Flash-Next-NVFP4   # hyper pro: the model-overlay-orig directory
CONC=4 FLASH_NEXT_EDITION=normal bash serve-fast.sh wa  # FLASH_NEXT_EDITION=hyperpro for hyper pro
```

The measured runs used `serve-fast.sh` inside that venv, not a pip install of this tree. With `REPO` denoting the checkout root, the launcher sets `PYTHONPATH` to `$REPO/flashinfer-vt:$REPO/python` for normal or `$REPO/flashinfer-vtoea:$REPO/python` for hyperpro (OEA on). The assembled FlashInfer tree and this checkout's Python source therefore take precedence over installed packages.

Only the changed or added FlashInfer files ship in `flashinfer-overlays/`. The rest comes from the public FlashInfer 0.6.17 package. `tools/assemble_flashinfer.py` builds `flashinfer-vt/` for normal and `flashinfer-vtoea/` for hyper pro beside `serve-fast.sh`:

- The tool uses only the Python standard library and needs no GPU or network.
- It requires version 0.6.17 and checks every file it keeps against the package's RECORD hashes. It also checks both final trees against `flashinfer-overlays/MANIFEST.json`. The manifest identifies the release's assembled trees, including modification notices, comment edits and the optional policy-loader label/text update; these hashes are not a new performance measurement.
- It never writes into site-packages, and it refuses to overwrite a non-empty target unless that target is already identical.
- If the trees are missing, `serve-fast.sh` prints the command to build them.

Full instructions and settings are in the [fork README](https://github.com/aiueo52/sglang-rtxpro6000/blob/d21289fb923ac65b7b913b73c6e32a2fa2601e7b/README.md). This repository also records the [squashed SGLang diff](patches/sglang/perf-1027-core.diff) and the [FlashInfer overlay diffs](patches/flashinfer/2026-10-27/). The earlier launchers and numbered patch series below still describe the 2026-10-02 build.

#### Server access and redaction

- The server binds to **127.0.0.1** on `PORT` (default `8001`) and serves the OpenAI-compatible API under `/v1` with the model name `flash-next`.
- The launcher configures the inherited admin and management routes, including weight updates and tensor-based LoRA loading (`/load_lora_adapter_from_tensors`), to require an admin key by default. It generates a random per-start key unless you set `FLASH_NEXT_ADMIN_API_KEY`. Tools calling admin routes (for example `/flush_cache`) need `FLASH_NEXT_ADMIN_API_KEY` set before launch and must send that value in an `Authorization: Bearer <key>` header. The shipped `serve-*` helpers and `oea` do not call admin routes; `oea` uses the control file.
- **The launcher does not enable LoRA.** If you enable it, inference requests can name loaded adapters. Only path-loaded adapters, supplied through `--lora-paths` or admin routes, can reload from their recorded path after eviction or an admin unload; unloading them is not access revocation. Tensor-loaded adapters record the placeholder path `__tensor__` and cannot reload this way. Restart the server without that adapter in `--lora-paths` to retire it. This is upstream behavior, not per-adapter access control added by the fork.
- **Several tokenizer workers are refused while a key is set.** With `--tokenizer-worker-num` greater than 1, upstream skips the authentication middleware, so this release stops at startup with `authentication with multiple tokenizer workers is not supported in this release` when `--api-key` or `--admin-api-key` is set. The launcher always sets an admin key, so launches through it must keep one tokenizer worker (the default).
- **Readback redaction is not a complete security boundary.** `/server_info`, deprecated `/get_server_info` and the startup log always redact `admin_api_key`, `api_key` and `ssl_keyfile_password` when set; unset values remain `None`. Request and crash dumps are off by default. When enabled, the crash-dump `resolved_config` snapshot, `server_args` and recorded launch command redact secrets. Request data, including prompts and outputs, remains sensitive and is not redacted; treat dump files as sensitive. The admin key remains visible in the live process argv to local processes, as described below.
- SGLang's `server_args.py` has no admin-key environment-variable support, so the launcher passes `--admin-api-key=<key>` in argv. Other local processes can read it with `ps`: on Linux, command lines are visible to all local users by default, not just the server's user.
- Normal inference endpoints (`/v1/*`, `/generate`) and `/health` and `/metrics` remain unauthenticated by default on 127.0.0.1. Upstream CORS allows any origin, so any web page you visit can send inference requests to them. **Do not expose the server to a network without an authenticating proxy.** To require a key for inference and server-info endpoints, append upstream arguments after the preset: `bash serve-fast.sh wa --api-key=<inference-key>`. For keys beginning with `-`, use `--api-key=<key>`; argparse may reject a separate value that starts with `-`. This does not replace the admin key; `/health` and `/metrics` stay unauthenticated.
- `bash serve-fast.sh wa --print` prints the resolved command and a filtered environment without starting anything. Argument names are lowercased, stripped of leading dashes, and normalized from `_` to `-`; values are shown as `<redacted>` when the name ends in `key`, `token`, `secret`, `password`, `passwd`, `auth`, `cred`, `creds`, `credential` or `credentials`. Both `--flag value` and `--flag=value` are handled, including sensitive long-option abbreviations such as `--admin-api-k` and `--api-ke`, plus `--hf-token` abbreviations. Known nonsensitive options such as `--ssl-keyfile`, `--speculative-num-draft-tokens` and `--speculative-token-map` stay visible, as do the exact harmless token options `--bucket-time-to-first-token` and `--kt-max-deferred-experts-per-token`. After a credential flag without `=`, the next token is masked whatever it starts with, unless it is itself a credential option such as `--admin-api-key` (then that option's value is masked); when the masked token is a bare `--` or looks like a credential flag without `=`, the token after it is masked too. Environment redaction is unchanged: values whose names contain `TOKEN`, `KEY`, `SECRET`, `PASS`, `AUTH` or `CRED` (case-insensitive) are hidden. Review the output before sharing it; diagnostic redaction does not hide runtime argv from local processes.

#### Concurrency and adaptive settings

**`CONC=1` is not a one-request admission limit.** It leaves `serve-local.sh`'s `--max-running-requests 4` and `--chunked-prefill-size 8192` in place, defaults to 10 Mamba cache slots, and does not enable HiCache, even if `FLASH_NEXT_HICACHE_GB` is set. Send only one request at a time to reproduce a single-stream workload. This documents the measured launcher behavior; it has not been changed to enforce a limit of one.

`ADAPTIVE_CONFIG` defaults to `adaptive/w16_c4.json`, for both editions and every `CONC` unless explicitly overridden. The other retained adaptive tables are explicit alternatives or test fixtures; the launcher does not automatically select them by edition or `CONC`. `CONC=6` is unvalidated and its long-context ladder does not fit in KV.

#### Runtime files and disk use

By default, `serve-fast.sh` writes under `~/.cache/flash-next-fast/<edition>/`: `hf/` for Hugging Face caches, `workspace/` for FlashInfer builds, `jit/`, `triton/`, `nvfp4/` and other compiler caches, `tmp/`, `runtime/` and `oea/`. If unset, `TMPDIR` points to `tmp/`, `XDG_RUNTIME_DIR` to `runtime/`, and `XDG_CACHE_HOME` to `xdg/` there. A pre-existing `XDG_CACHE_HOME` supplies the base instead of `~/.cache`; explicit cache-variable overrides are preserved. Model weights remain in `TARGET_MODEL`, not copied into these caches.

Budget several GB per edition for compiled-kernel/build caches (a planning estimate, not a measured disk total); use varies with kernels and toolchain, and temporary build files need extra space. A packed 49,152 × 2,560 NVFP4 head is about 71 MB per cache entry; changing model or map can create another entry. The 32 GB HiCache pool is host RAM, not a disk allocation.

#### Live OEA control (hyperpro)

After startup completes, use `./oea status`, `./oea on`, `./oea off` or `./oea rows <2..16>`. **Switch only between generations/requests, never during a generation.** `on` uses the server's launch value of `OEA_K0` (default `6`); `off` restores standard computation. `rows` changes the minimum number of live verify rows needed for OEA without changing on/off (default `16`).

The default control file is `~/.cache/flash-next-fast/hyperpro/oea/flash-next.ctl`; acknowledgements are in the adjacent `.status.jsonl` file. If you override `XDG_CACHE_HOME` or `SGLANG_OEA_CTL_FILE`, pass the same settings to the launcher and `oea`. The helper controls an already-running hyperpro server; it does not start one.

### Measured results

The machine was one RTX PRO 6000 Blackwell (Max-Q), which also drives a 4K desktop; the screen was off during measurement. Each comparison was made against our previous best build of the same edition (the builds served before this release: 2026-10-22 for normal, 2026-10-23 for hyper pro; not the 2026-10-02 build documented further down, and not published separately), with alternating server starts (A B B A blocks) and 95 % intervals over the start pairs. The ratio, quality and BetterBench measurements predate the release's licensing, packaging and launcher cleanups; the per-genre table below was measured on the final release.

Since measurement, the Python changes are security hardening (the admin key on tensor-based LoRA loading, the refusal of authentication with several tokenizer workers, and redaction of readbacks, logs, dumps and argv) and docstring, log and error text. The inference path is unchanged, and the launcher command only adds the admin key. In the FlashInfer overlay, the optional tactic-rows policy loader changed only its accepted approval label and its error text; tactic rows stay disabled and no policy is shipped.

The per-genre runs are published in `results/runs-1027/genre/`. The other figures in this update are author-reported; their run files are not published in this repository. Besides the per-genre runs, `results/` contains the earlier September and 2026-10-02 runs, not the run files for the October 27 comparisons, quality checks or BetterBench results.

#### Single request (c = 1): measured configurations only

| edition | server ms/step (new/old) | decode tok/s (new/old) | verdict |
|---|---|---|---|
| normal | 1.001 [0.993, 1.008] | 1.001 [0.987, 1.015] | no significant difference (12 starts) |
| hyperpro | 0.978 [0.966, 0.989] | 1.029 [1.003, 1.055] | faster (12 starts) |

These are averaged over 11 workload shapes: English and Japanese prose, agent loop, reasoning, code edit, greedy and sampling, and one 64k-token prompt.

Conditions: both arms of every comparison used a private MTP head and token map (the fork uses the stock checkpoint head and ships public maps instead, see Assets). Do not assume these ratios apply to the public assets; their absolute speed is in the BetterBench table. The normal row used tactic rows, which are not available in this release. With tactic rows off (the release setting), an earlier 8-start check found every shape within ±4 %, but its intervals were too wide for a conclusion.

#### Quality

| check | normal | hyperpro |
|---|---|---|
| GSM8K 1,319 (thinking closed), c=1 → c=4 | 624 → 622 | 648 → 654 |
| 200 problems with thinking, c=1 → c=4 | 192 → 191 | 193 → 193 |
| 4 × 64k-token synthetic, cross-talk / wrong / mismatch vs c=1 | 0 / 0 / 0 | 0 / 0 / 0 |

For ThinkingCap against the stock weights on the same fast path: +0.6 points [−1.2, +2.7] over 308 items (GSM, MATH, CE), thinking 14 % shorter, time to answer 12 % shorter.

#### BetterBench (BetterBench 0.6.0, the machine used by aiueo52 and contributors, screen off, one start per build)

| build | c=1 decode tok/s | 4 concurrent, total tok/s | TTFT p50 c=1 | prefill tok/s @ 32k |
|---|---|---|---|---|
| flash-next-fast normal | 369.9 | 710.2 | 101 ms | 12,796 |
| flash-next-fast hyperpro | 342.3 | 693.7 | 103 ms | 12,266 |
| previous best build (2026-10-22), normal | 371.6 | 331.5 | 99 ms | 12,918 |
| previous best build (2026-10-23), hyperpro | 353.3 | 323.6 | 101 ms | 12,401 |

Conditions:
- All four builds ran with the public token maps shipped by the fork and the stock MTP head in the model checkpoint. Hyper pro used the locally built ThinkingCap overlay.
- c=1 decode is BetterBench's weighted median over its single-stream categories.
- The previous builds serve one request at a time, so their 4-concurrent requests queue; flash-next-fast serves 4 at once (`CONC=4`).
- One start per build: treat differences of a few percent as run-to-run noise.
- Reference runs with a private token map and fine-tuned MTP head (not shipped) reached 420.9 (normal) and 423.7 (hyperpro) tok/s at c=1; the public assets were about 12 % and 19 % slower in those runs.

#### Per genre, one request at a time, public components (final release, 2026-10-27)

| build | code edit | English prose | Japanese prose | agent |
|---|---|---|---|---|
| flash-next-fast normal | 611 (529–680) / 8.75 | 279 (272–285) / 2.43 | 288 (265–371) / 2.48 | 382 (357–410) / 3.75 |
| flash-next-fast hyperpro | 657 (575–731) / 9.36 | 270 (259–294) / 2.39 | 288 (257–339) / 2.55 | 367 (350–389) / 3.62 |
| previous best build (2026-10-22), normal | 617 (582–679) / 9.35 | 277 (257–299) / 2.42 | 292 (268–326) / 2.52 | 379 (361–395) / 3.74 |

For one request at a time, the final release (normal) was within 1.5 % of the previous best build in every genre; hyper pro was about 8 % faster than normal on code edits, 3–4 % slower on English prose and agent turns, and equal on Japanese prose.

Conditions:
- The final release as committed, with the fork's public token maps and the stock MTP head; hyper pro used the locally built ThinkingCap overlay. The launcher's default settings (`CONC=4 bash serve-fast.sh wa`) applied, except port 30000 and FlashInfer modules loaded from an earlier build of the same sources (`PREBUILT_CACHE=1`; nine smaller modules of that build target `sm_120a`, where a fresh build targets `sm_120f`; see `results/runs-1027/README.md`). The previous best build serves one request at a time and used the same public assets.
- The 32 held-out prompts of `results/TABLES.md` section 8 (8 per genre; output caps 12,288 tokens for code edit, 6,144 for prose, 8,192 for agent), thinking on, greedy, one request at a time, one server start per build, screen off. Each server first got one unmeasured 8-token streaming request.
- Cells: mean client-side decode t/s (min–max) / mean acceptance (tokens per verify step, bonus included).
- One start per build: treat differences of a few percent as run-to-run noise.
- Run files: `results/runs-1027/genre/G*.jsonl` (timings, token counts and acceptance; no generated text); the table is also `results/TABLES.md` section 9.
- Reference only: the 2026-10-06 public-components run of the 2026-10-02 build (`wa`, greedy, `results/TABLES.md` section 8) gave 468 / 232 / 248 / 307. That run had the screen on at 160 Hz and was on another day, so it is not a like-for-like comparison.

#### Per genre, one request at a time, private components (not shipped)

| build | code edit | English prose | Japanese prose | agent |
|---|---|---|---|---|
| flash-next-fast normal | 764 (722–825) / 11.68 | 297 (282–314) / 2.50 | 326 (312–362) / 2.74 | 438 (394–493) / 4.24 |
| flash-next-fast hyperpro | 768 (665–836) / 11.41 | 300 (279–373) / 2.61 | 322 (298–357) / 2.74 | 434 (418–457) / 4.41 |
| previous best build (2026-10-22), normal | 724 (684–755) / 11.35 | 302 (286–319) / 2.57 | 365 (305–421) / 3.17 | 434 (412–467) / 4.32 |

With the private head and maps, hyper pro was within about 1 % of normal in every genre; normal was 6 % faster than the previous best build on code edits, within 2 % on English prose and agent turns, and 11 % slower on Japanese prose. That gap is in acceptance (2.74 vs 3.17 tokens per verify step), not in step time. With one start per build, these runs cannot tell a build difference from start-to-start variation: three starts of one earlier build (`results/bn1-study-v2/`, `wa`) gave 2.56–3.35 on Japanese prose, and the public-components pair above shows no such gap (2.48 vs 2.52). Compared with these runs, the public components in the table above were 20 % slower on code edits, 6 % on English prose, 12 % on Japanese prose and 13 % on agent turns for normal, and 10–15 % slower in every genre for hyper pro.

Conditions:
- The same builds, settings, prompts and warm-up as the public table above; only the assets differ: the fine-tuned MTP head (for hyper pro, on the ThinkingCap overlay) and the private draft maps (49,152 rows, and 32,768 rows for the recurrent head), the pair of the 12-start comparisons above. The previous best build used the same files, as its production entry does.
- **These assets are not published, so this table cannot be reproduced from this repository.**
- One start per build: treat differences of a few percent as run-to-run noise. A first hyper pro attempt was stopped when the screen turned on and was repeated from a fresh server start; its numbers are not used.
- Run files: `results/runs-1027/genre/P*.jsonl` (timings, token counts and acceptance; no generated text); the table is also `results/TABLES.md` section 10.
- Reference only: the 2026-10-06 private-components run of the 2026-10-02 build (`wa`, greedy, `results/TABLES.md` section 8) gave 584 / 253 / 290 / 359. That run had the screen on at 160 Hz and was on another day, so it is not a like-for-like comparison.

#### Reference: tactic rows (not available in this release)

Tactic rows select MoE GEMM tactics by row count. **Tactic rows are not available in this release (no policy is shipped).** Every `SGLANG_TACTIC_ROWS` value other than `0` stops at start. Policy-building and approval tools are not included.

These normal-edition prefill numbers come from an earlier tree on the machine used by aiueo52 and contributors, before a later SGLang MoE fix. They are reference values, not measurements or features of this release:

| | without | with | change |
|---|---|---|---|
| c = 1 | 13,361 tok/s | 14,021 tok/s | +4.9 % |
| c = 4 | 10,335 tok/s | 10,968 tok/s | +6.1 % |

Earlier-tree quality checks found logit drift within A/A noise and retrieval 24/24 with tactic rows versus off for normal; hyperpro kept them off.

#### Not in this version

- 6 concurrent requests as a default: at 6 the long ladder does not fit in KV.
- Per-request verify/draft-length control, an experimental prefill/decode interleaving scheduler and experimental additional kernels were measured but are not included. `FLASH_NEXT_PD_INTERVAL` only sets upstream's existing `--prefill-decode-interval`; it does not enable that experimental scheduler.

#### Assets

The defaults and paths below are in the **fork checkout**, not this companion repository. This repository keeps the public maps and source records under `tokenmaps/public/` and provides their [build instructions](tokenmaps/README.md).

- **Draft token map:** `tokenmaps/public_49152.pt` is the default in the fork. It was built from public data only; the sources and their licenses are in `tokenmaps/public_49152.sources.json`. The two public maps and their two `.sources.json` files are offered under **CC BY-SA 4.0, not Apache-2.0**; see the attributions and source-use caveats in the fork's `THIRD_PARTY_NOTICES.md`. To use your own map, set `TOKEN_MAP=/path/to/map.pt` (a sorted list of token ids saved with `torch.save`). The build and preparation scripts named in the provenance JSON are not in the fork; rebuild the maps with this repository's `tokenmaps/build_hot_vocab.py` and `tokenmaps/public/prep_public.py` (see [`tokenmaps/README.md`](tokenmaps/README.md) for the recipe and byte-reproducibility limits).
- **Recurrent MTP head map:** `tokenmaps/public_32768.pt` (a subset of the above, same sources) is the default. Set `FLASH_NEXT_RECUR_HEAD=` (empty) to turn it off, or point it at your own map. At launch, `serve-fast.sh` aborts if a configured map file is missing. During server initialization, if loading or installing an existing recurrent map fails, the server logs a warning and keeps the original MTP head and token map. This fallback does not apply to the draft token map.
- **MTP head:** the stock head in the model checkpoint is used. A head you trained yourself can be used by putting it in your own checkpoint directory and pointing `TARGET_MODEL` at that directory. Our own fine-tuned head is not shipped.
- **ThinkingCap (hyper pro):** the weights are not redistributed here. Download the ThinkingCap NVFP4 checkpoint linked below and build the overlay locally (see [`tools/thinkingcap_overlay/overlay.py`](https://github.com/aiueo52/sglang-rtxpro6000/blob/d21289fb923ac65b7b913b73c6e32a2fa2601e7b/tools/thinkingcap_overlay/overlay.py); `plan`, `build`, then `verify`).

#### Changes from upstream

These counts describe the 2026-10-27 fork, not the historical 2026-10-02 patch series:

- Compared with the public base `16e5682`, `python/` has **61 modified, 31 added and 5 deleted files**, all under `python/sglang/`. `test/` has **3 modified, 28 added and 1 deleted file**. Counts include all file types, exclude generated Python caches, and compare symlinks by type and target; these are file counts, not test-case counts.
- Of the **2,065 upstream test files**, 2,061 are unchanged, 3 modified and 1 removed, for **2,092 files under `test/`** in the release. The removed file is `test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py`: its fake worker no longer matches the fork's worker, so it failed without testing the production path (see the fork's `MODIFICATIONS.md`). Its removal is not a licensing removal. The additions include CPU-only redaction and route-authentication tests.
- In FlashInfer 0.6.17, 7 files are modified in each variant and 1 is added (normal only); only the overlays are shipped in the fork.

#### Model cards and licenses

Model weights have separate terms from this code. Read the model cards and linked licenses before use:

- [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) (revision `de4b8e4`): the source model. Its LICENSE file is the Qwen Community License 1.0, which sets conditions on some commercial uses.
- [bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-NVFP4/tree/df3922b21fc28a8604cc9b31451495690d155194) (revision `df3922b`): its LICENSE file is the Qwen Community License 1.0, which sets conditions on some commercial uses.
- [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/tree/7b719225242aacd3dbd3f9407468c2ee9a9d2594) (revision `7b71922`): its card lists the license as `other` and refers to the source model for terms.

## Results

The tables in this section are the 2026-09 state. The 2026-10-02 production build and what changed are in
[*2026-10 update*](#2026-10-update) below.

> **Private components in these numbers.** The `w4`, `w16` and `wa` rows of both tables below were
> measured with a **fine-tuned MTP draft head** (v3 in the headline table, v5 in the final table) and a
> **reduced draft vocabulary (token map, 49,152 tokens)**. Both were trained / built by the author on
> private data (their own chat and agent-session logs and MTP-training prompts) and are **not
> published**, so these rows cannot be reproduced exactly with the public checkpoint alone. With the
> checkpoint's original MTP head and a different token map, acceptance and t/s need not match these
> historical runs (see *Limitations*). Since 2026-10 a token map built only from public data is published
> (`tokenmaps/public/`); it is not the map these rows were measured with.

### Headline (2026-09-06, all three launch profiles in one session)

Tokens per second as seen by a streaming client, one request at a time, greedy decoding, thinking
mode on (reasoning tokens count). Mean of 2 measured repeats after a warm-up; range in parentheses.

| launch profile | code-edit | prose-en | agent-loop |
|---|---:|---:|---:|
| unmodified base fork (2026-09-02, recommended sampling, n=3) | 211 (210–213) | 129 (125–131) | 174 (166–181) |
| `w4` — draft width 4 | 399 (398–400) | 280 (275–285) | **349** (347–351) |
| `w16` — draft width 16 | **671** (655–687) | 187 (175–198) | 314 (312–316) |
| `wa` — adaptive width | 595 (546–644) | **282** (282–283) | 326 (318–335) |
| mean tokens accepted per verify step (w4 / w16 / wa) | 3.89 / 11.85 / 10.24 | 2.61 / 3.05 / 2.60 | 3.30 / 5.20 / 4.48 |

Best profile vs the base-fork row, as a **reference comparison only**: **3.2×** code-edit, **2.2×** prose-en,
**2.0×** agent-loop. The base row was measured on a different day (2026-09-02 vs 09-06), with the harness's
recommended sampling instead of greedy, and at a different power limit (300 W vs 325 W), so the ratios are
not a controlled A/B; the 8-prompt × 3-restart table below is the more robust measurement of the final state.
Source files: `results/runs/p2m2332-{w4,w16,wa}.jsonl`, `results/runs/baseline-nextn.jsonl`
(tables regenerated by `python3 results/make_tables.py` → `results/TABLES.md`).

### Final configuration, 8 held-out prompts per domain × 3 server restarts (2026-09-08)

A more robust measurement of the final state (`w4` and the final three-width `wa` policy, fine-tuned
MTP head v5): 32 synthetic prompts not used for any tuning, a fresh server process for each of three
restarts, longer outputs (up to 12,288 / 6,144 / 8,192 tokens).

| profile | code-edit | prose-en | prose-ja | agent-loop |
|---|---:|---:|---:|---:|
| `w4` | 380 (364–390) | 259 (245–325) | 272 (249–301) | 334 (314–353) |
| `wa` | 579 (517–662) | 255 (237–304) | 294 (253–629) | 358 (331–388) |

Mean over 24 requests per cell (min–max). Measured with the private fine-tuned MTP head v5 and the
private 49,152-token draft vocabulary (neither is published; see the note above). Source:
`results/bn1-study-v2/*.jsonl`. A post-adoption smoke
test of the final `wa` on the headline prompts gave 664 / 273 / 396 t/s (code-edit / prose-en /
agent-loop; `results/runs/wa5prod0221-wa.jsonl`).

### Exact conditions

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell **Max-Q** Workstation Edition, 96 GB GDDR7 (~1.79 TB/s), SM120. Power limit 325 W (the card's maximum; default 300 W) for the headline and final tables, set from 2026-09-03 on; their 2026-09-02 base row, like every other run that day (see `docs/timeline.md`), used the default 300 W. No power-limit throttling was seen in 39 nvidia-smi samples taken at 100 ms intervals during decode (2026-09-07); throttling shorter than the sampling interval cannot be ruled out. The same GPU also drove the desktop (KDE/X11, 4K at 160 Hz) |
| host | AMD Ryzen Threadripper 9960X, 256 GB RAM, Ubuntu 24.04.4, kernel 7.0 |
| software | NVIDIA driver 595.84 (open), CUDA 13.3 (nvcc 13.3.73), Python 3.12.13, PyTorch 2.13.0+cu130, Triton 3.7.1, flashinfer-python 0.6.17 + `patches/flashinfer`, sgl-kernel 0.4.6.post1, SGLang = jpezzulli/sglang-rtxpro6000 @ `16e5682aad` (SGLang 0.5.19.dev485 base) + `patches/sglang` |
| model | [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) (NVFP4 routed experts; dense weights served as weight-only FP8 by our patches), with a **privately fine-tuned MTP draft head** (v3 for the headline table, v5 for the final table) and a reduced draft vocabulary of 49,152 tokens — neither is published (see *Limitations*) |
| serving | SGLang, TP 1, `--kv-cache-dtype fp8_e4m3`, page size 64, context length 262,144, NEXTN (MTP) linear-chain speculation with top-k 1, profiles from `launch/serve-fast.sh`; the final table used memory fraction 0.920 and `MAX_TOTAL_TOKENS=131072` (`docs/reproduce.md` §4) |
| requests | one at a time (batch size 1), greedy (`temperature=0`), chat template with `enable_thinking=true`, `reasoning_effort=medium` |
| workloads | synthetic prompts in `bench/workloads/`: **code-edit** = mechanical edit of a 6,076-token Python module, 2,400 output tokens; **prose-en** = original short story, 104-token prompt, 1,200 output tokens; **agent-loop** = continue a tool-calling transcript, 388-token prompt, 1,200 output tokens; **prose-ja** = Japanese short story. In the headline runs most generated tokens are reasoning tokens |
| metric | client decode rate = (completion tokens − 1) / (time of last token chunk − time of first), i.e. excluding time to first token; acceptance = completion tokens / target verify steps (bonus token included) from the server's Prometheus counters |

### Comparison with published numbers

For context only: the single-GPU, single-stream numbers for this model that we could find published
(as of 2026-09-28) are 171 tok/s and, in a later configuration, 207 tok/s at short context — reported
by the author of the base fork on the
[NVIDIA developer forum](https://forums.developer.nvidia.com/t/optimized-qwen3-8-flash-next-on-1x-rtx-pro-6000-171-tok-s-524k-and-hicache-nixl-persistence/381722),
and 120–173 tok/s with MTP in a
[Level1Techs thread](https://forum.level1techs.com/t/qwen-3-8-flash-next-nvfp4-on-a-single-rtx-pro-6000-and-system-memory/254579).
Our numbers are higher than those, but prompts, output lengths, sampling, thinking settings and GPU
variants differ, so this is not a like-for-like comparison — and it would not exist without the base
fork. We do not claim to be the fastest.

## 2026-10 update

This section records the historical **2026-10-02 build**, not the 2026-10-27 fork release. Its measurements, flags and limits apply to that earlier build.

On 2026-10-01/02 we cut fixed per-step costs, fixed two correctness problems (one in the sampling verify, one in the draft), and
added rejection sampling. The production build changed on 2026-10-02 (SGLang patches 0106–0121,
FlashInfer patches 08–10). Every SGLang-side piece has its own flag in `launch/serve-fast.sh`, on by default; the FlashInfer RQ2
prologue change (patches 08 and 10) has none. Details
and evidence: [`docs/optimizations.md`](docs/optimizations.md) §H; what did not ship:
[`docs/rejected.md`](docs/rejected.md) §7; how it was measured: [`docs/measurement.md`](docs/measurement.md) §9.

What changed:

- **Fixed-cost cuts** (RT1 router top-k, SV1/SV2 sparse sampling verify, FG1 GDN stream overlap, DG1
  draft MoE GEMV, RQ2 MoE routing prologue).
- **Sampling fidelity.** The speculative verify ignored `min_p`, so with LM Studio's `min_p 0.05` it could
  emit tokens the normal sampler excludes. Fixed (no measurable speed cost). A second fix (ST1) lets the
  draft's attention see its newest positions. It brought no measurable speed-up and ships with XA1, a
  Triton decode-attention kernel (12.2 vs 16.3 µs per call).
- **Rejection sampling** for sampling requests: sparse, with a sharpened 16-token draft support and block
  verification.
- **524,288-token context** on `wa` (factor-2 YaRN, opt-in): needles up to 480k tokens pass.
- **A public-data draft token map**, `tokenmaps/public/public_49152.pt` (CC BY-SA 4.0, sources in
  `tokenmaps/README.md`).

Historical measurements: A/Bs against the then-previous production build on `wa`, with the same private MTP head and
token map as the tables above. These are 8 server starts each, in both greedy and LM Studio's default
sampling (temperature 0.8, top_p 0.95, top_k 40, min_p 0.05). 95 % CIs treat each server start as the
unit:

| change | LM Studio sampling | greedy |
|---|---|---|
| fixed-cost cuts (all five + RQ2), ms per output token | **−9.6 %** [−13.6, −5.3] | **−4.6 %** [−8.9, −0.1] |
| XA1 on top of ST1, t/s | +2.8 % (arm-level CI [−0.9, +6.7]) | +1.2 %, unresolved |
| rejection-sampling package, t/s | **+6.5 %** [+1.4, +12.0]; code-edit +1.6 %, unresolved | step time −2.2 %, unresolved |

The rejection-sampling package did **not** pass the adoption rule we set before measuring it: code-edit
stayed unresolved. We shipped it anyway for the prose and agent gains (+6 to +10 %). Turn it off with
`SGLANG_OPT_SPEC_SPARSE_RS=0` (with `launch/as-measured/serve-fast.sh` also set `SGLANG_RS_BLOCK_VERIFY=0`,
or the server refuses to start). `w16` has run the new build only in single-prompt smokes (`w4` is in the table below).

Absolute throughput of the 2026-10-02 build, measured on 2026-10-06 with the 32 held-out prompts of the
September study (8 per workload, thinking on, one request at a time, one server start per row, launched the
way LM Studio launches it, except that the desktop stayed at 160 Hz: LM Studio's entry switches it to 60 Hz
while serving, which `docs/measurement.md` puts at 5–7 % shorter steps). Full table and conditions: [`results/TABLES.md`](results/TABLES.md) section 8;
raw (stripped) files: `results/runs-1002/pub1006/`.

| build | sampling | code-edit | prose-en | prose-ja | agent-loop |
|---|---|---:|---:|---:|---:|
| `wa`, private MTP head + private map | greedy | 584 | 253 | 290 | 359 |
| `wa`, private MTP head + private map | LM Studio default | 568 | 240 | 310¹ | 340 |
| `w4`, private MTP head + private map | greedy | 382 | 254 | 275 | 339 |
| **`wa`, public components only**² | greedy | **468** | **232** | **248** | **307** |
| **`w4`, public components only**² | greedy | **356** | **241** | **254** | **301** |
| unmodified base fork, W4 (2026-09-02, section 1 prompts)³ | recommended | 211 | 129 | 117 | 174 |

Mean decode t/s over 8 prompts per cell (client side, thinking tokens included).
¹ One prompt reached 744 t/s; without it the mean is about 248.
² The checkpoint's original MTP head (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, all files hash-checked against
the Hub) + `tokenmaps/public/public_49152.pt`. Same code, flags and launcher as the rows above; only the
model directory and the token map differ. This is the historical public-asset result for the 2026-10-02
build, not a measurement of the 2026-10-27 fork.
³ Different, shorter prompts, so indicative only; it is the same baseline row as in the first table.

Greedy numbers of the 2026-10-02 build with the private head and map were within a few percent of the
September study (`wa` 579 / 255 / 294 / 358; that study averaged three server starts and used the September
launcher settings, so this is a rough check). The early-October work mostly helped sampled requests.
In these historical runs, using only public components made `wa` about 20 % slower on code-edit and
8–15 % slower elsewhere, mainly through lower acceptance (code-edit 8.1 vs 10.5 tokens per step).
These differences are not predictions for the 2026-10-27 fork or other workloads.

## How it works (short version)

This explanation and its speed figures describe the historical September and 2026-10-02 configurations.

At batch size 1 every decode step is a chain of ~1,500–2,500 small GPU kernels that mostly stream
weights from DRAM. Speed comes from two multipliers: **tokens accepted per step** (speculative
decoding) and **step time**. Details, numbers and evidence for each item: [`docs/optimizations.md`](docs/optimizations.md).

1. **Wider, cheaper speculation.** A reduced draft vocabulary (FR-Spec-style token map) so the draft
   head is small; the QSA pending-index ring generalised so draft widths above 4 work at all; W4 / W8 /
   W16 profiles; an adaptive policy (`wa`) that switches between 3, 7 and 15 draft steps while it
   generates, using the draft's own top-1 confidence and a measured step-time model (after fixing several
   bugs in SGLang's stock adaptive mode); a fine-tuned MTP head.
2. **Dense weights as FP8 through a custom Triton W8A16 GEMV.** CUTLASS W8A8 FP8 GEMMs made the step
   35 % *slower* at M ≤ 16 rows; a weight-only FP8 GEMV tuned per shape (with in-launch split-K) made
   it faster — for the dense layers, the lm_heads and the MTP layer.
3. **Fewer, cheaper kernels.** Fused KV-FP8 store, three-kernel hyper-connection mix with FP8 weights,
   strided GDN reads, fused norms into GEMVs, folded HC boundaries, page-parallel QSA metadata,
   programmatic dependent launch for Triton kernels, and more — kernels per step went from 3,175 to
   ~2,100 at W16.
4. **MoE glue inside FlashInfer.** Six csrc patches to the CUTLASS fused-MoE path: one-kernel routing
   prologue for top-10/512 experts, strides and row expansion folded into it, memset folded, active
   expert group packing.
5. **Singleton-route pruning (changes the model's computation).** In a verify batch, a route to an
   expert used by only one row with routing weight < 0.08 is dropped, evaluated inside the routing
   prologue: step −9.5 % (W16) / −10 % (W4). A quality audit (≈4,000 questions per arm from GSM8K, MMLU,
   HumanEval and JCommonsenseQA, plus a synthetic 18.5k-token needle retrieval test) found no statistically significant difference from
   the unpruned model; that is not a proof of equivalence (non-inferiority at a 0.5 pp margin was not
   established for GSM8K or HumanEval, see Limitations). **Test conditions differ from the speed runs:** the quality
   audit ran with thinking disabled and up to 2 concurrent requests, whereas the historical speed runs here are
   single-request with thinking enabled. Batching weakens the pruning (a route is only pruned when a
   single row uses the expert), so the audit is **not a conservative test** of the single-request setting.
   Because the decision depends on every token in the verify batch, the draft (head, token map, width)
   also influences the target's output; it is not output-neutral.

The bandwidth view: by our own byte estimate, the weights that must be read per step put a floor of
roughly 4–5 ms on a W4 step and 8.5–9.5 ms on a W16 step. We measured ≈25 ms (W16) at the start and
≈9 ms (W4) / ≈16 ms (W16) at the end; the rest is the latency of a long serial chain of 1,500–2,100
small kernels. 1,000 tokens/s on one GPU is not reachable this way. See
[`docs/roofline.md`](docs/roofline.md).

## What did not work

These historical September and 2026-10-02 rejected experiments are as useful as the adopted ones;
each has numbers in [`docs/rejected.md`](docs/rejected.md). A few examples:

| idea | result |
|---|---|
| dense FP8 via CUTLASS W8A8 GEMM | −29 to −36 % t/s (≈40 µs fixed cost per small GEMM) |
| n-gram (suffix-match) drafting, width 16 | hit rate 37–49 %, but 314–347 t/s on code-edit vs 429–433 for plain MTP-15 |
| NVFP4 (4-bit) dense GEMVs / lm_heads | kernels 1.5–1.7× faster, but in-server acceptance loss on the benchmark workloads outweighed it |
| persistent "megakernel" with grid barriers | a grid barrier costs more than the kernel launch it replaces |
| L2-cache prefetch / persisting pins for the next layer's weights | slower: the desktop's GPU context evicts persisting lines; HC kernels lose bandwidth |
| raising the pruning threshold to 0.10 | quality gate inconclusive and unexplained behaviour changes: not shipped |
| dense rejection sampling over the draft vocabulary (2026-10) | acceptance +5–8 % under sampling, but code-edit t/s −3.6 % to −7.7 %: the per-step cost ate the gain |
| fixing the draft's attention prefix as a speed-up (2026-10) | no measurable change in acceptance over 8 server starts; ships only as a correctness fix |
| draft-head candidate shortlists, sparse FP4 experts, lossless FP8 weight coding, narrower experts / HC bottleneck without retraining | all no-go (quality loss or no net speed-up) |
| retraining the MTP head on more data (several rounds) | in-server acceptance on the benchmark workloads mostly did not improve |

## Reproduce

For the 2026-10-27 fork, use the update above. The following recipe rebuilds the historical 2026-10-02
configuration; see [`docs/reproduce.md`](docs/reproduce.md). In short: clone the base fork, `patches/sglang/apply.sh`,
build its venv, `patches/flashinfer/apply.sh`, download the NVFP4 checkpoint, use the public token map
`tokenmaps/public/public_49152.pt` or build your own (`tokenmaps/`), then `SGLANG_DIR=… TOKEN_MAP=… launch/serve-fast.sh wa` and measure with `bench/`
(`fnbench`).

## Limitations and measurement noise

This section concerns the historical September and 2026-10-02 configurations, not the four-request fork release above.

- **Not published, so not exactly reproducible:** the fine-tuned MTP draft heads and the private token maps
  were derived from the author's private conversations and agent transcripts. With the public
  checkpoint's original MTP head, the historical prose/agent runs had lower acceptance (the fine-tuned
  heads added roughly 5–10 % there); this is not a general speed guarantee. The published public-data token map shares 72.0 % of its ids with the private
  one. Measured together (original head + public map, 2026-10-06), `wa` was about 20 % slower on code-edit
  and 8–15 % slower on prose and agent-loop than with the private pair (table in the 2026-10 update). Or build your own token map from text you may use (`tokenmaps/README.md`),
  and train your own head on your own data with [`docs/train-your-own-mtp-head.md`](docs/train-your-own-mtp-head.md)
  (code in [`mtp-train/`](mtp-train/)).
- **Workloads are narrow.** The headline table uses one prompt per workload; code-edit is an extremely
  repetitive mechanical edit where wide speculation shines (11.9 tokens accepted per step). The
  8-prompt table is the fairer summary. Thinking is on, so most counted tokens are reasoning text.
- **Noise.** The GPU also drives a 4K 160 Hz desktop; the compositor preempts the GPU and adds up to
  7.5 % to step time (measured with compositing on vs off), and client-side t/s of identical configurations varied by ±10–15 % across sessions.
  Comparisons were therefore made within one session, with CUPTI kernel time, with acceptance measured
  separately, and later with an 8-prompt ABBA protocol ([`docs/measurement.md`](docs/measurement.md)).
- **Greedy is not bit-reproducible:** the MoE finalize epilogue and the HC split-K kernels use atomic
  adds, so outputs and acceptance vary run to run (typically ±4–6 % acceptance, more on long W16 chains).
- **The pruning changes the model.** The quality audit found no statistically significant difference
  from the unpruned build (McNemar, smallest p = 0.362), with rough detection limits of ±1.5 pp
  (GSM8K/MMLU) and ±3 pp (HumanEval). "No significant difference" is not non-inferiority: at a 0.5 pp
  margin, non-inferiority was **not established** for GSM8K or HumanEval (INCONCLUSIVE; one-sided 95 %
  Tango score lower bounds −1.80 pp and −1.02 pp vs the unpruned build, where HumanEval's 164 problems
  are underpowered for 0.5 pp when the difference is small; `results/quality-q1/q2_quality_analysis.md`), while MMLU and JCommonsenseQA
  passed against it. The audit ran with thinking disabled and up to 2 concurrent requests,
  unlike the single-request, thinking-enabled speed runs; because batching weakens the pruning, it is not a
  conservative test of the single-request setting that the speed numbers use, and quality with thinking
  enabled was not measured. Set `PRUNE_TAU=0` to turn it off.
- Single-user oriented: memory fractions and state-slot counts are sized for one request at a time on
  a GPU that also runs a desktop. Throughput with concurrent requests was not optimised. Three known
  issues are listed in [`docs/optimizations.md`](docs/optimizations.md) §H7: two only matter with several
  running requests or other callers; the third (rare exact ties at the sparse-verify cut) can also
  touch a single sampled request.
- The baseline row was measured with the harness's recommended sampling (temperature 0.2–0.7), the
  other rows greedy.
- Historical test issue: two unit tests of the rejected `NGRAM_CHAIN` drafter failed after a later
  logging change (`test_ngram_chain_worker.py`). This test file was removed in the 2026-10-27 update;
  the drafter is not used by the measured or default profiles, but the experimental `ngram` profile
  in `launch/serve-fast.sh` selects it.

## Repository layout

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
| `mtp-train/` | MTP draft-head fine-tuning pipeline: dump client, corpus readers, self-generation, trainer, renewal evaluator, write-back (no data or weights) |
| `docs/` | optimisations, rejected experiments, roofline, measurement method, timeline, reproduction; `lab-notes/` = the original engineering logs |

## Licence and credits

Apache License 2.0 (`LICENSE`), matching SGLang and FlashInfer; see `NOTICE` for the upstream
projects whose code the patches modify. A few patch hunks touch files derived from
flash-linear-attention (MIT) and causal-conv1d (BSD-3-Clause); those keep their own licences
(texts in `licenses/`, file list in `NOTICE`). The two public token maps (`tokenmaps/public/public_49152.pt`,
`tokenmaps/public/public_32768.pt` and their two source records) are derived from public datasets including
Wikipedia and are offered under CC BY-SA 4.0
with the attributions in `NOTICE`. No model weights are included. Qwen3.8-Flash-Next is under the Qwen Community License 1.0 (https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE), not Apache-2.0; this repo contains no model files. Much of the implementation
and measurement work was done with AI coding assistants under the author's direction; all numbers
come from the author's machine.
