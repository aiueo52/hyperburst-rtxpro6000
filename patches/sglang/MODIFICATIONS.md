# Modified upstream files

Apache License 2.0 §4(b) notice: the SGLang series, FlashInfer patches and overlays, and
the 2026-10-27 update by aiueo52 and contributors modify the upstream files listed below.
The SGLang base is
[jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) at
`16e5682aad6e3335f38ec8d5711176278f12a086` (itself derived from SGLang and, for some files,
vLLM, flash-linear-attention, causal-conv1d and others). See the published fork's
`THIRD_PARTY_NOTICES.md` and `third_party_licenses/flash-next-fast/` for retained notices
and license texts.

The current-release inventory mirrors the published fork's `MODIFICATIONS.md`. Unless stated
otherwise, release paths below are relative to that fork; patch paths are in this repository.

The first table includes existing base-fork files that still differ in the published fork; files added
by the series or update are listed separately where relevant to release maintenance.
The update column includes release maintenance after the 2026-10-27 performance update,
including security, notice, docstring/log/error-text and tooling changes. “yes” compares the
current file with the previous public build, `5105985`; “no” means that the old-series
version is unchanged by the update. Notice-only changes count as modifications.

Compared with the public base `16e5682`, `python/` has **61 modified, 31 added and
5 deleted files**, all under `python/sglang/`. `test/` has **3 modified, 28 added and 1 removed
file**. Of the **2,065 upstream test files**, 2,061 are unchanged, 3 modified and 1 removed,
for **2,092 files under `test/`**. These are file counts, not test-case counts; all file types
are included, generated Python caches are excluded, and symlinks are compared by type and target.

Old-series patch mappings are retained from the previous table; `–` means no mapping there.
Patch numbers are the 121 series commits in order after `16e5682aad` (0001 = first), excluding
the two header commits, and match `patches/sglang/series/` in this repository. Each patch records
its historical change; the update column records subsequent differences, not new patch numbers.

| file | old series patches | changed by the 2026-10-27 update |
|---|---|---|
| `.gitignore` | – | yes |
| `README.md` | – | yes |
| `python/sglang/kernels/jit/csrc/elementwise/hc_combine.cuh` | 0054, 0058 | yes |
| `python/sglang/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py` | 0001 | yes |
| `python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py` | 0048, 0070 | yes |
| `python/sglang/kernels/ops/elementwise/elementwise.py` | 0055, 0071 | yes |
| `python/sglang/kernels/ops/elementwise/hc_combine.py` | 0054, 0058 | yes |
| `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py` | 0023, 0045 | yes |
| `python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py` | – | yes |
| `python/sglang/kernels/ops/speculative/__init__.py` | 0114, 0115 | yes |
| `python/sglang/kernels/ops/speculative/eagle.py` | – | yes |
| `python/sglang/kernels/ops/speculative/reject_sampling.py` | – | yes |
| `python/sglang/kernels/ops/speculative/topk1.py` | 0016, 0082, 0114, 0115 | yes |
| `python/sglang/srt/arg_groups/overrides.py` | – | yes |
| `python/sglang/srt/arg_groups/speculative_hook.py` | 0002 | yes |
| `python/sglang/srt/environ.py` | 0019, 0042, 0054, 0056, 0059, 0075, 0082, 0107, 0108, 0109, 0112, 0113, 0114, 0115, 0116, 0117, 0118, 0119, 0121 | yes |
| `python/sglang/srt/entrypoints/http_server.py` | – | yes |
| `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py` | 0012 | yes |
| `python/sglang/srt/layers/attention/linear/gdn_backend.py` | 0048 | yes |
| `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py` | 0018 | yes |
| `python/sglang/srt/layers/attention/qsa/graph_metadata.py` | 0005, 0043 | yes |
| `python/sglang/srt/layers/attention/qsa/metadata.py` | 0005 | yes |
| `python/sglang/srt/layers/attention/qsa/qsa_indexer.py` | 0005 | yes |
| `python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py` | 0005, 0015, 0016, 0121 | yes |
| `python/sglang/srt/layers/hc_mix_triton.py` | 0010 | yes |
| `python/sglang/srt/layers/hyperconnection.py` | 0003, 0010, 0017, 0025, 0031, 0034, 0040, 0054, 0056, 0059, 0075, 0109 | yes |
| `python/sglang/srt/layers/linear.py` | 0003, 0099 | yes |
| `python/sglang/srt/layers/logits_processor.py` | 0032 | yes |
| `python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py` | 0100, 0103 | yes |
| `python/sglang/srt/layers/moe/topk.py` | 0019, 0110, 0111 | yes |
| `python/sglang/srt/layers/quantization/fp8.py` | 0004, 0032, 0040, 0049, 0079, 0098 | yes |
| `python/sglang/srt/layers/quantization/modelopt_quant.py` | 0003, 0022, 0112 | yes |
| `python/sglang/srt/layers/quantization/nvfp4_online.py` | 0022 | yes |
| `python/sglang/srt/managers/overlap_utils.py` | 0116 | yes |
| `python/sglang/srt/managers/scheduler.py` | – | yes |
| `python/sglang/srt/managers/tokenizer_manager.py` | – | yes |
| `python/sglang/srt/managers/utils.py` | – | yes |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | 0005 | yes |
| `python/sglang/srt/mem_cache/memory_pool.py` | 0019, 0020 | yes |
| `python/sglang/srt/mem_cache/qsa_kv_pool.py` | 0005 | yes |
| `python/sglang/srt/model_executor/forward_context.py` | – | yes |
| `python/sglang/srt/model_executor/input_buffers.py` | 0014 | yes |
| `python/sglang/srt/model_executor/model_runner.py` | 0004, 0012, 0013 | yes |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | 0012 | yes |
| `python/sglang/srt/model_executor/runner/eager_runner.py` | 0114 | yes |
| `python/sglang/srt/models/qwen2_moe.py` | 0003, 0019, 0024, 0040, 0041, 0055, 0057, 0060, 0112 | yes |
| `python/sglang/srt/models/qwen3_5.py` | 0006, 0007, 0048, 0049, 0050, 0051, 0079, 0109 | yes |
| `python/sglang/srt/models/qwen3_5_mtp.py` | 0011, 0022 | yes |
| `python/sglang/srt/models/qwen3_next.py` | 0004 | yes |
| `python/sglang/srt/models/qwen4_exp.py` | 0003, 0017, 0055, 0056, 0075, 0109 | yes |
| `python/sglang/srt/models/qwen4_exp_mtp.py` | 0019, 0024, 0042, 0044 | yes |
| `python/sglang/srt/server_args.py` | 0002, 0003 | yes |
| `python/sglang/srt/speculative/adaptive_runtime_state.py` | 0012, 0061, 0082, 0104 | yes |
| `python/sglang/srt/speculative/adaptive_spec_params.py` | 0012, 0061, 0062, 0063, 0066, 0082 | yes |
| `python/sglang/srt/speculative/base_spec_worker.py` | 0082 | yes |
| `python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py` | 0106, 0107, 0114, 0116 | yes |
| `python/sglang/srt/speculative/eagle_draft_extend_cuda_graph_runner.py` | 0061 | yes |
| `python/sglang/srt/speculative/eagle_info.py` | 0116 | yes |
| `python/sglang/srt/speculative/eagle_utils.py` | 0107, 0108, 0113, 0115, 0116, 0117, 0119 | yes |
| `python/sglang/srt/speculative/eagle_worker_common.py` | 0115, 0116 | yes |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | 0004, 0006, 0012, 0014, 0016, 0065, 0082, 0095, 0096, 0104, 0106, 0107, 0114, 0115, 0116, 0117, 0118, 0119 | yes |
| `python/sglang/srt/speculative/spec_info.py` | 0002, 0116, 0117, 0119 | yes |
| `python/sglang/srt/speculative/spec_utils.py` | 0106, 0107, 0108, 0113, 0116, 0117, 0118, 0119 | yes |
| `test/registered/kernels/ops/speculative/test_spec_topk1.py` | 0114, 0115 | no |
| `test/registered/unit/models/test_qwen4_exp_mtp.py` | 0044 | no |
| `test/registered/unit/spec/test_adaptive_spec_params.py` | 0063 | no |

## Patch series (pre-2026-10-27)

The first table retains the historical mappings for files that still differ in the published
fork. The original series also temporarily modified this file; it is net unchanged and is not
a current modified base-fork file:

| modified upstream file | patches |
|---|---|
| `python/sglang/kernels/ops/attention/fla/layernorm_gated.py` | 0071 (reverted by 0072; net unchanged) |

The separate training-data dump hook in `patches/sglang/mtp-dump/` (nine patches, applied on
top of `flash-next-series` by its own `apply.sh`, not part of the 121-patch series or the
2026-10-27 update) also modifies `python/sglang/srt/models/qwen4_exp.py` and
`python/sglang/srt/models/qwen4_exp_mtp.py` and adds `python/sglang/srt/models/mtp_dump.py`.

### Historical FlashInfer patch mapping

Carried over from the 2026-10-02 version of this file. Modified by `patches/flashinfer/` (historical
production chain 01–10) and its copies under `bench/bench/moe_smallm/patches/`:

- `flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` (01–06, 08, 10)
- `flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h` (01–06)
- `flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py` (07; also
  `kernels/microbench/glue_e/gdn_wy_T16_strided.patch` and the series' `bench/glue_e/gdn_wy_T16_strided.patch`)
- `flashinfer/jit/core.py` (09)

Additionally, only in `patches/flashinfer/experimental/` (rejected variants):
`flashinfer/data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu`,
`flashinfer/fused_moe/core.py`.

The current vt/vtoea overlays are listed separately below and are published as alternative,
self-contained diffs against the pristine wheel in `patches/flashinfer/2026-10-27/`; apply them
instead of the historical 01–10 chain, not on top of it.

## Release security, text, tests and tooling

These entries describe release-maintenance changes and additions, not new numbered
performance patches. Redaction affects diagnostic copies only, not live authentication or TLS
credentials. Text-only changes do not constitute a new GPU performance measurement.

| file | release change |
|---|---|
| `README.md` | Corrects source/test inventory, credits, asset terms and serving/security guidance; keeps the base fork's README below the release introduction unchanged. |
| `THIRD_PARTY_NOTICES.md` | Corrects inventories and retained third-party attributions; records CC BY-SA 4.0 for the four public token-map data files and their source-use caveats. |
| `MODIFICATIONS.md` | Updates this modification inventory, including security, text, tests, tools and FlashInfer overlays. |
| `serve-fast.sh` | Configures the source overlay and cache paths, then delegates launch and redacted command/environment diagnostics to `serve-local.sh`. |
| `serve-local.sh` | Generates a random per-start admin key with `secrets.token_hex(32)` in isolated Python (`-I -B`), or uses a nonempty `FLASH_NEXT_ADMIN_API_KEY`, and passes it as `--admin-api-key=<key>` to accept dash-leading keys. Redacts sensitive long-option abbreviations in diagnostic argv. After a credential flag without `=`, the next token is masked whatever it starts with, unless it is itself a credential option; after a masked bare `--` or credential-looking token, the token after it is masked too. Known nonsensitive options, including `--bucket-time-to-first-token` and `--kt-max-deferred-experts-per-token`, stay visible when no value mask is pending. Launch presets and user-argument forwarding are unchanged. |
| `python/sglang/srt/arg_groups/overrides.py` | Redacts credential fields in resolved readbacks and diagnostic server-argument/argv copies, including sensitive parser-recognized long-option abbreviations, without mutating live values. Applies the same argv rule as `serve-local.sh`. |
| `python/sglang/srt/entrypoints/http_server.py` | Requires the admin key, when one is set, for `/load_lora_adapter_from_tensors`, which deserializes pickled tensors, like the other tensor routes. Refuses to start with `--api-key` or `--admin-api-key` when `--tokenizer-worker-num` is greater than 1, because upstream skips the authentication middleware in that mode. Inference routes are unchanged. |
| `python/sglang/srt/server_args.py` | Keeps credential redaction in configuration readbacks and startup diagnostics. |
| `python/sglang/srt/managers/tokenizer_manager.py` | Redacts request/crash dump configuration snapshots, server arguments and recorded launch argv without changing request payloads or live credentials. |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | Updates the recurrent-head fallback warning to name the public `FLASH_NEXT_RECUR_HEAD` setting alongside the internal variable; fallback behavior is unchanged. |
| `test/registered/unit/test_readback_redaction.py` | Added regression coverage for set, unset and empty credentials in readbacks/startup logs while preserving live values. |
| `test/registered/unit/test_dump_redaction.py` | Added request/crash dump regression coverage while preserving live values and argv. |
| `test/registered/unit/test_argv_redaction.py` | Added CPU-only argv redaction tests, including option abbreviations; checks the `serve-local.sh --print` lists against the real parser. |
| `test/registered/unit/entrypoints/test_http_server_auth.py` | Added CPU-only tests: admin routes need the admin key when one is set, inference routes keep upstream authentication, and authenticated multi-tokenizer startup is refused before workers start. |
| `tools/assemble_flashinfer.py` | Adds standard-library-only reconstruction of both backend trees with wheel/overlay hashes, validated paths and non-overwriting targets. |
| `tools/thinkingcap_overlay/overlay.py` | Documents and builds a user-supplied ThinkingCap/original-head overlay; does not bundle model weights or grant new model rights. |
| `tools/thinkingcap_overlay/convert.py` | Supplies checkpoint/header validation and streaming tensor-copy support for the overlay tool. |
| `flashinfer-overlays/MANIFEST.json` | Records the shipped overlay and assembled-tree hashes, including notice/comment edits and the optional policy-loader label/text update; per-file source changes are listed below. |

Other added test files (the three modified upstream tests remain in the first table):

| added file | coverage |
|---|---|
| `test/registered/kernels/ops/mamba/test_causal_conv1d_chain_parallel.py` | Parallel causal-conv1d chains. |
| `test/registered/unit/layers/quantization/test_qwen4_exp_dense_fp8.py` | Qwen4 dense FP8 path. |
| `test/registered/unit/mem_cache/test_mamba_none_mode_sizing.py` | Mamba none-mode sizing. |
| `test/registered/unit/mem_cache/test_qsa_pending_ring.py` | QSA pending ring. |
| `test/registered/unit/spec/test_adaptive_confidence.py` | Adaptive confidence. |
| `test/registered/unit/spec/test_adaptive_runtime_state_sizing.py` | Adaptive runtime-state sizing. |
| `test/registered/unit/spec/test_adaptive_switch_point.py` | Adaptive switch points. |
| `test/registered/unit/spec/test_adaptive_target_autotune.py` | Adaptive target autotuning. |
| `test/registered/unit/spec/test_stage2_bucket_widths.py` | Stage-two bucket widths. |
| `test/srt/layers/test_draft_logits_out.py` | Draft-logit output. |
| `test/srt/layers/test_gdn_wy_t16_strided.py` | Strided GDN WY verify. |
| `test/srt/layers/test_glue_d.py` | Fused kernel integration. |
| `test/srt/layers/test_hc_combine_fused.py` | Fused hyperconnection combine. |
| `test/srt/layers/test_hc_fused.py` | Fused hyperconnections. |
| `test/srt/layers/test_hc_mix2.py` | Hyperconnection mix. |
| `test/srt/layers/test_oea_prologue.py` | OEA prologue. |
| `test/srt/layers/test_qsa_graph_metadata_pages.py` | QSA graph metadata pages. |
| `test/srt/layers/test_w8a16_gemv_silu.py` | W8A16 GEMV/SiLU. |
| `test/srt/layers/test_w8a16_stage3_golden.py` | W8A16 golden results. |
| `test/srt/layers/test_w8a16_stage3_gpu.py` | W8A16 GPU execution; removes internal orchestration wording from the docstring without changing tests. |
| `test/srt/layers/test_w8a16_stage3_interpreter.py` | W8A16 interpreter execution. |
| `test/srt/layers/test_w8a16_stage3_planner.py` | W8A16 planning. |
| `test/srt/layers/test_w8a16_stage3_utils.py` | W8A16 utilities. |
| `test/srt/speculative/test_draft_fuse.py` | Draft fusion. |

## Removed base-fork files

### Removed test

`test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py` is an upstream test that the
old series changed (patch 0012). It is removed. Its fake worker lacks attributes that the fork's
worker reads, so both the upstream version and the series version fail on the fork's tree without testing the
production worker path. This is not a licensing removal.

### Licensing removals

The update removes these five inherited files for licensing reasons. The Moore Threads kernel
carried a proprietary/confidential notice; the four Hunyuan3D extension files carried Tencent
Hunyuan non-commercial-license notices. These paths identify removed files, not shipped code.

- `python/sglang/kernels/aot/csrc/elementwise/fused_add_rms_norm_kernel.mu`
- `python/sglang/kernels/ops/diffusion/ext/hunyuan3d_rasterizer/rasterizer.cpp`
- `python/sglang/kernels/ops/diffusion/ext/hunyuan3d_rasterizer/rasterizer_gpu.cu`
- `python/sglang/kernels/ops/diffusion/ext/hunyuan3d_rasterizer/rasterizer.h`
- `python/sglang/kernels/ops/diffusion/ext/mesh_processor/mesh_processor.cpp`

These five paths are excluded entirely from `patches/sglang/perf-1027-core.diff`, including
its deletion hunks, so no file content from them is redistributed in the diff. After applying
the diff to the pinned base, complete the licensing removals with this single path-only command:

```bash
git rm -- \
  python/sglang/kernels/aot/csrc/elementwise/fused_add_rms_norm_kernel.mu \
  python/sglang/kernels/ops/diffusion/ext/hunyuan3d_rasterizer/rasterizer.cpp \
  python/sglang/kernels/ops/diffusion/ext/hunyuan3d_rasterizer/rasterizer_gpu.cu \
  python/sglang/kernels/ops/diffusion/ext/hunyuan3d_rasterizer/rasterizer.h \
  python/sglang/kernels/ops/diffusion/ext/mesh_processor/mesh_processor.cpp
```

The removed upstream test is not one of these five paths; its deletion is included in the diff.

## FlashInfer (flashinfer-python 0.6.17 wheel)

The published fork ships 15 source overlays: seven wheel-modified files in each of `vt` and `vtoea`,
plus one vt-only addition. The fork's `tools/assemble_flashinfer.py` assembles the dependencies
from the user-installed wheel. The changes below are taken from the fork's `THIRD_PARTY_NOTICES.md`,
“Modified FlashInfer files”; the original copyright and license notices remain applicable.
Wheel license copies are at `flashinfer-overlays/{vt,vtoea}/LICENSE`, with a retained FlashInfer
notice at `third_party_licenses/flash-next-fast/NOTICE.flashinfer`.

| overlay file | changes from the wheel |
|---|---|
| `flashinfer-overlays/vt/flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` | Fuses the MoE routing prologue into grouped-GEMM setup and uses a cheaper expert sort for small batches. |
| `flashinfer-overlays/vt/flashinfer/data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu` | Adds optional per-request lengths (`vt_lens`) to the CUTLASS fused-MoE entry point. |
| `flashinfer-overlays/vt/flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h` | Fuses the MoE routing prologue into grouped-GEMM setup and uses a cheaper expert sort for small batches. |
| `flashinfer-overlays/vt/flashinfer/fused_moe/core.py` | Passes `vt_lens` through and adds the optional tactic-rows hook. |
| `flashinfer-overlays/vt/flashinfer/fused_moe/tactic_rows_policy.py` | Added: loads and checks the optional tactic-rows policy; vt only. The update changes its accepted approval label to `trial` and its error text from private to local cache and workspace; no policy is shipped. |
| `flashinfer-overlays/vt/flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py` | Reads strided q/k/v for the 16-token verify shape; identical in both variants. |
| `flashinfer-overlays/vt/flashinfer/jit/core.py` | With `FLASHINFER_P2_NO_NINJA=1`, loads an existing compiled module as-is and raises an error instead of compiling if it is missing; identical in both variants. |
| `flashinfer-overlays/vt/flashinfer/jit/fused_moe.py` | Builds the MoE module under its own name, `fused_moe_<arch>_vt_prologue_v1`. |
| `flashinfer-overlays/vtoea/flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh` | Fuses the MoE routing prologue into grouped-GEMM setup and uses a cheaper expert sort for small batches. |
| `flashinfer-overlays/vtoea/flashinfer/data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu` | Adds optional per-request lengths (`vt_lens`) to the CUTLASS fused-MoE entry point. |
| `flashinfer-overlays/vtoea/flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h` | Fuses the MoE routing prologue into grouped-GEMM setup and uses a cheaper expert sort for small batches. |
| `flashinfer-overlays/vtoea/flashinfer/fused_moe/core.py` | Passes `vt_lens` through on the OEA MoE path, without tactic rows. |
| `flashinfer-overlays/vtoea/flashinfer/gdn_kernels/gdn_decode_bf16_wy_output_only.py` | Reads strided q/k/v for the 16-token verify shape; identical in both variants. |
| `flashinfer-overlays/vtoea/flashinfer/jit/core.py` | With `FLASHINFER_P2_NO_NINJA=1`, loads an existing compiled module as-is and raises an error instead of compiling if it is missing; identical in both variants. |
| `flashinfer-overlays/vtoea/flashinfer/jit/fused_moe.py` | Builds the MoE module under its own name, `fused_moe_<arch>_vtoea_prologue_v1`. |

The vtoea files differ from vt where its OEA MoE path differs, without tactic rows. No tactic-rows
policy JSON is shipped; `tactic_rows_policy.py` is loader code. Complete wheel dependency trees
are user-assembled, not redistributed here; see `flashinfer-overlays/MANIFEST.json` for hashes.

## Header commits and update

In the published fork branch `flash-next-fast`, a commit after patch 0105 adds the one-line “Modified by aiueo52 for flash-next-fast (2026);
see MODIFICATIONS.md.” header comment to every file the first 105 patches modify, and adds
this list. A second commit after patch 0121 adds the header to files first modified by
0106–0121. Neither header commit is counted in the patch numbers above.
`patches/sglang/apply.sh` alone does not add them.

The 2026-10-27 update commit refreshes the release tree, modification notices and this inventory
on top of `5105985`; it is separate from the numbered 121-patch series.
