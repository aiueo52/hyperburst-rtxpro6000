#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 aiueo52 and contributors
set -euo pipefail
if [ "${SGLANG_TACTIC_ROWS-0}" != 0 ]; then
  echo "[serve-fast] tactic rows are not available in this release (no policy is shipped)" >&2
  exit 2
fi
REPO="$(cd "$(dirname "$0")" && pwd)"
PRINT_ONLY=0
ARGS=()
for arg in "$@"; do
  if [ "$arg" = --print ]; then PRINT_ONLY=1; else ARGS+=("$arg"); fi
done
set -- "${ARGS[@]}"
if [ ! -d "$REPO/flashinfer-vt/flashinfer" ] || [ ! -d "$REPO/flashinfer-vtoea/flashinfer" ]; then
  echo "[serve-fast] FlashInfer trees missing; run: python3 $REPO/tools/assemble_flashinfer.py" >&2
  [ "$PRINT_ONLY" = 1 ] || exit 2
fi
PROFILE="${1:-wa}"; shift || true
export FLASH_NEXT_PRINT_ONLY="$PRINT_ONLY"
: "${TARGET_MODEL:?Set TARGET_MODEL to the model directory for the selected edition}"
TOKEN_MAP="${TOKEN_MAP:-tokenmaps/public_49152.pt}"
# Use the 32768-row public map for the recurrent head when present.
if [ "${FLASH_NEXT_RECUR_HEAD+x}" != x ] && [ -f "$REPO/tokenmaps/public_32768.pt" ]; then
  FLASH_NEXT_RECUR_HEAD=tokenmaps/public_32768.pt
fi
if [ -n "${FLASH_NEXT_RECUR_HEAD:-}" ]; then
  export SGLANG_RND_MTP_RECUR_HEAD="$FLASH_NEXT_RECUR_HEAD"
else
  unset SGLANG_RND_MTP_RECUR_HEAD
fi
export -n TOKEN_MAP
export PORT="${PORT:-8001}"
PYTHON="${PYTHON:-python3}"
if [ "$PRINT_ONLY" = 0 ]; then
  [ -d "$TARGET_MODEL" ] || { echo "[serve-fast] TARGET_MODEL must be a model directory" >&2; exit 2; }
  ASSETS=()
  [ "$TOKEN_MAP" = none ] || ASSETS+=("$TOKEN_MAP")
  [ -z "${FLASH_NEXT_RECUR_HEAD:-}" ] || ASSETS+=("$FLASH_NEXT_RECUR_HEAD")
  for asset in "${ASSETS[@]}"; do
    case "$asset" in /*) file="$asset" ;; *) file="$REPO/$asset" ;; esac
    [ -f "$file" ] || { echo "[serve-fast] required asset not found: $asset" >&2; exit 2; }
  done
fi
export FLASH_NEXT_EDITION="${FLASH_NEXT_EDITION:-normal}"
case "$FLASH_NEXT_EDITION" in
  normal) EDITION_OEA=0 ;;
  hyperpro) EDITION_OEA=1 ;;
  *) echo "[serve-fast] FLASH_NEXT_EDITION must be normal or hyperpro" >&2; exit 2 ;;
esac
export CONC="${CONC:-4}"
export DISPLAY_RESERVE_GB="${DISPLAY_RESERVE_GB:-7}" GPU_MEMORY_GB="${GPU_MEMORY_GB:-95.6}"
export WA_MEM_FRACTION="${WA_MEM_FRACTION:-0.96}" FLASH_NEXT_HICACHE_GB="${FLASH_NEXT_HICACHE_GB:-32}"
export FLASH_NEXT_PD_INTERVAL="${FLASH_NEXT_PD_INTERVAL:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export LANG="${LANG:-C.UTF-8}" LC_ALL="${LC_ALL:-C.UTF-8}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FLASHINFER_NO_DOWNLOAD=1 SGLANG_LOG_MS=1
# Capture the user's XDG base before assigning any per-application defaults.
CACHE_BASE="${XDG_CACHE_HOME:-$HOME/.cache}/flash-next-fast/$FLASH_NEXT_EDITION"
export HF_HOME="${HF_HOME:-$CACHE_BASE/hf}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_BASE/xdg}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}" TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_BASE/triton}" SGLANG_JIT_CACHE_DIR="${SGLANG_JIT_CACHE_DIR:-$CACHE_BASE/jit}"
export SGLANG_NVFP4_CACHE_DIR="${SGLANG_NVFP4_CACHE_DIR:-$CACHE_BASE/nvfp4}" SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$CACHE_BASE/sglang}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_BASE/inductor}" CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CACHE_BASE/cuda}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_BASE/extensions}" TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$CACHE_BASE/numba}" TRITON_HOME="${TRITON_HOME:-$CACHE_BASE/triton-home}"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
export TMP="${TMP:-$TMPDIR}" TEMP="${TEMP:-$TMPDIR}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-$CACHE_BASE/runtime}" SGLANG_OEA_CTL_FILE="${SGLANG_OEA_CTL_FILE:-$CACHE_BASE/oea/flash-next.ctl}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_BASE/workspace}"
export ADAPTIVE_CONFIG="${ADAPTIVE_CONFIG:-$REPO/adaptive/w16_c4.json}"
export SGLANG_TACTIC_ROWS="${SGLANG_TACTIC_ROWS:-0}"
PREBUILT_CACHE="${PREBUILT_CACHE:-0}"
export -n PREBUILT_CACHE
case "$PREBUILT_CACHE" in
  0|1) ;;
  *) echo "[serve-fast] PREBUILT_CACHE must be 0 or 1" >&2; exit 2 ;;
esac
unset FLASHINFER_DISABLE_JIT FLASHINFER_P2_NO_NINJA
unset SGLANG_TACTIC_ROWS_APPROVED_SHA256
if [ "$PREBUILT_CACHE" = 1 ]; then export FLASHINFER_P2_NO_NINJA=1; fi

case "$CONC" in
  1|4|6) ;;
  *) echo "[serve-fast] CONC must be 1, 4 or 6" >&2; exit 2 ;;
esac
if [ "$CONC" != 1 ] && [ "$PROFILE" != wa ]; then
  echo "[serve-fast] CONC=4/6 requires profile wa" >&2; exit 2
fi
OEA="${OEA:-$EDITION_OEA}"
EDITION_DL_TAU_BY_S='3:0.1,7:0.2,15:0.4,31:0.5'
EDITION_DL_STEPUP=1
EDITION_VT_TAU=''
WA_DEFAULT_CONFIG="$REPO/adaptive/w16_c4.json"
WA_SLOTS=10
WA_CHUNK=4096
WA_GRAPH_BS=(1)
if [ "$PROFILE" = wa ] && [ "$CONC" != 1 ]; then
  WA_DEFAULT_CONFIG="$REPO/adaptive/w16_c$CONC.json"
  WA_SLOTS=$((5 * CONC))
  WA_CHUNK=8192
  if [ "$FLASH_NEXT_EDITION" = normal ]; then
    case "${FLASH_NEXT_CHUNK-8192}" in
      4096|8192|16384|32768) WA_CHUNK="${FLASH_NEXT_CHUNK-8192}" ;;
      *) echo "[serve-fast] FLASH_NEXT_CHUNK must be 4096, 8192, 16384 or 32768" >&2; exit 2 ;;
    esac
    WA_PD_INTERVAL="${FLASH_NEXT_PD_INTERVAL-0}"
    if [[ ! "$WA_PD_INTERVAL" =~ ^[0-9]+$ ]]; then
      echo "[serve-fast] FLASH_NEXT_PD_INTERVAL must be a nonnegative integer" >&2; exit 2
    fi
  fi
  WA_HICACHE_GB="${FLASH_NEXT_HICACHE_GB-0}"
  if [[ ! "$WA_HICACHE_GB" =~ ^[0-9]+$ ]]; then
    echo "[serve-fast] FLASH_NEXT_HICACHE_GB must be a nonnegative integer" >&2; exit 2
  fi
  WA_GRAPH_BS=()
  for ((bs=1; bs<=CONC; bs++)); do WA_GRAPH_BS+=("$bs"); done
  WA_MEM_FRACTION="${WA_MEM_FRACTION:-0.96}"
  export SGLANG_FP8_W8A16_GEMV_MAX_M="${SGLANG_FP8_W8A16_GEMV_MAX_M:-32}"
  export SGLANG_NVFP4_GEMV_MAX_M="${SGLANG_NVFP4_GEMV_MAX_M:-32}"
  export SGLANG_OPT_MAMBA_NONE_MODE_SIZING="${SGLANG_OPT_MAMBA_NONE_MODE_SIZING:-1}"
fi
if [ "$CONC" != 1 ]; then
  WA_LAUNCH_SLOTS="${MAMBA_SLOTS:-$WA_SLOTS}"
  if [[ ! "$WA_LAUNCH_SLOTS" =~ ^[1-9][0-9]*$ ]] ||
     [[ ${#WA_LAUNCH_SLOTS} -lt ${#WA_SLOTS} ]] ||
     { [[ ${#WA_LAUNCH_SLOTS} -eq ${#WA_SLOTS} ]] && [[ "$WA_LAUNCH_SLOTS" < "$WA_SLOTS" ]]; }; then
    echo "[serve-fast] MAMBA_SLOTS must be an integer >= $WA_SLOTS for CONC=$CONC" >&2; exit 2
  fi
  TRAILING=("$@")
  for ((i=0; i<${#TRAILING[@]}; i++)); do
    flag="${TRAILING[i]%%=*}"
    case "$flag" in
      --max-running-requests) expected="$CONC" ;;
      --max-mamba-cache-size) expected="$WA_LAUNCH_SLOTS" ;;
      --chunked-prefill-size) expected="$WA_CHUNK" ;;
      --prefill-decode-interval)
        [ "$FLASH_NEXT_EDITION" = normal ] || continue
        expected="$WA_PD_INTERVAL" ;;
      --enable-hierarchical-cache)
        [ "$FLASH_NEXT_EDITION" = normal ] || [ "${FLASH_NEXT_HICACHE_GB+x}" = x ] || continue
        if [[ "$WA_HICACHE_GB" =~ ^0+$ ]] || [[ "${TRAILING[i]}" == *=* ]] ||
           { ((i+1<${#TRAILING[@]})) && [[ "${TRAILING[i+1]}" != --* ]]; }; then
          echo "[serve-fast] $flag conflicts with FLASH_NEXT_HICACHE_GB=$WA_HICACHE_GB" >&2; exit 2
        fi
        continue ;;
      --hicache-size|--hicache-host-memory-mode|--hicache-write-policy|--hicache-io-backend|--hicache-mem-layout)
        [ "$FLASH_NEXT_EDITION" = normal ] || [ "${FLASH_NEXT_HICACHE_GB+x}" = x ] || continue
        if [[ "$WA_HICACHE_GB" =~ ^0+$ ]]; then
          echo "[serve-fast] $flag conflicts with FLASH_NEXT_HICACHE_GB=$WA_HICACHE_GB" >&2; exit 2
        fi
        case "$flag" in
          --hicache-size) expected="$WA_HICACHE_GB" ;;
          --hicache-host-memory-mode) expected=cache ;;
          --hicache-write-policy) expected=write_through ;;
          --hicache-io-backend) expected=kernel ;;
          --hicache-mem-layout) expected=page_first ;;
        esac ;;
      --hicache-storage-*|--hicache-prefetch-*)
        [ "$FLASH_NEXT_EDITION" = normal ] || [ "${FLASH_NEXT_HICACHE_GB+x}" = x ] || continue
        echo "[serve-fast] $flag conflicts with RAM-only HiCache" >&2; exit 2 ;;
      --cuda-graph-bs|--cuda-graph-bs-decode) expected="${WA_GRAPH_BS[*]}" ;;
      *) continue ;;
    esac
    values=()
    if [[ "${TRAILING[i]}" == *=* ]]; then
      values+=("${TRAILING[i]#*=}")
    else
      while ((i+1<${#TRAILING[@]})) && [[ "${TRAILING[i+1]}" != --* ]]; do
        i=$((i+1)); values+=("${TRAILING[i]}")
        [[ "$flag" == --cuda-graph-bs* ]] || break
      done
    fi
    if [ "${values[*]}" != "$expected" ] ||
       { [[ "$flag" == --cuda-graph-bs* ]] && [ "${#values[@]}" -ne "${#WA_GRAPH_BS[@]}" ]; }; then
      echo "[serve-fast] $flag conflicts with CONC=$CONC (expected $expected)" >&2; exit 2
    fi
  done
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}"
export SGLANG_FP8_W8A16_GEMV=1 SGLANG_GDN_BA_TRITON_GEMV=1
export SGLANG_KV_FP8_FUSED_STORE=${SGLANG_KV_FP8_FUSED_STORE:-1}
export SGLANG_HC_MIX2=${SGLANG_HC_MIX2:-1}
export FLASHINFER_GDN_WY_STRIDED_QKV=${FLASHINFER_GDN_WY_STRIDED_QKV:-1}
export SGLANG_DRAFT_LOGITS_OUT=${SGLANG_DRAFT_LOGITS_OUT:-1}
export SGLANG_ROUTER_GEMV=${SGLANG_ROUTER_GEMV:-1}
export SGLANG_HC_MIX2_FP8=${SGLANG_HC_MIX2_FP8:-1}
export SGLANG_SHARED_GATEUP_FUSED=${SGLANG_SHARED_GATEUP_FUSED:-1}
export SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS=${SGLANG_DRAFT_SKIP_VOCAB_WEIGHTS:-1}
export SGLANG_QSA_META_PAGE_PARALLEL=${SGLANG_QSA_META_PAGE_PARALLEL:-1}
export SGLANG_MTP_EMBED_TABLE=${SGLANG_MTP_EMBED_TABLE:-1}
export SGLANG_GDN_CONV_CHAIN_PARALLEL=${SGLANG_GDN_CONV_CHAIN_PARALLEL:-1}
export SGLANG_GDN_PROJ_DIRECT_LAYOUT=${SGLANG_GDN_PROJ_DIRECT_LAYOUT:-1}
export SGLANG_GDN_AB_STASH_DIRECT=${SGLANG_GDN_AB_STASH_DIRECT:-1}
export SGLANG_HC_GATE_EARLY=${SGLANG_HC_GATE_EARLY:-2}
export SGLANG_SHARED_GATE_EARLY=${SGLANG_SHARED_GATE_EARLY:-1}
export SGLANG_HC_APPLY_MIX_FUSED=${SGLANG_HC_APPLY_MIX_FUSED:-1}
export SGLANG_HC_LAYER_APPLY_FUSED=${SGLANG_HC_LAYER_APPLY_FUSED:-1}
export SGLANG_NORM_INTO_GEMV=${SGLANG_NORM_INTO_GEMV:-1}
if [ "${E_DRAFT:-1}" = 1 ] && [ "$PROFILE" != ngram ]; then
  export SGLANG_MTP_LMHEAD_NVFP4=${SGLANG_MTP_LMHEAD_NVFP4:-1} SGLANG_RND_DRAFT_CTX_VOCAB=${SGLANG_RND_DRAFT_CTX_VOCAB:-256}
  export SGLANG_RND_DRAFT_CTX_TOPK=${SGLANG_RND_DRAFT_CTX_TOPK:-0}
fi
export SGLANG_MTP_LMHEAD_NVFP4=${SGLANG_MTP_LMHEAD_NVFP4:-0}
export SGLANG_MOE_PRUNE_SINGLETON_TAU=${SGLANG_MOE_PRUNE_SINGLETON_TAU:-${PRUNE_TAU:-0.08}}
export SGLANG_MOE_PRUNE_IN_PROLOGUE=${SGLANG_MOE_PRUNE_IN_PROLOGUE:-1}
export SGLANG_LMHEAD_NVFP4=${SGLANG_LMHEAD_NVFP4:-0}
export FLASHINFER_MOE_PACK_GROUPS=${FLASHINFER_MOE_PACK_GROUPS:-1}
export SGLANG_TRITON_PDL=${SGLANG_TRITON_PDL:-1}
export PYTHONPATH="$REPO/flashinfer-vt:$REPO/python"
export FLASHINFER_WORKSPACE_BASE
case "${PLE_SIDE:-1}:${STACK_D:-1}" in
  [01]:[01]) ;;
  *) echo "[serve-fast] PLE_SIDE and STACK_D must be 0 or 1" >&2; exit 2 ;;
esac
case "${VT_GRAMMAR:-1}" in
  [01]) ;;
  *) echo "[serve-fast] VT_GRAMMAR must be 0 or 1" >&2; exit 2 ;;
esac
export SGLANG_RND_VT_GRAMMAR="${VT_GRAMMAR:-1}"
export SGLANG_RND_PLE_SIDE="${PLE_SIDE:-1}" SGLANG_RND_PLE_SIDE_PARTS=ab
unset SGLANG_RND_PLE_TEST SGLANG_RND_VT_TAU SGLANG_RND_DL_TAU_BY_S
export SGLANG_RND_DL_TAU=0 SGLANG_RND_DL_STEPUP=0 SGLANG_RND_DL_HOSTW=0 SGLANG_RND_DL_STEPUP31=0
if [ "$PROFILE" = wa ] && [ "${STACK_D:-1}" = 1 ] && [ "${E_DRAFT:-1}" = 1 ]; then
  export SGLANG_RND_DL_TAU="${DL_TAU:-0}"
  if [ -n "${DL_TAU_BY_S-$EDITION_DL_TAU_BY_S}" ]; then
    export SGLANG_RND_DL_TAU_BY_S="${DL_TAU_BY_S-$EDITION_DL_TAU_BY_S}"
  fi
  export SGLANG_RND_DL_STEPUP="${STEPUP:-$EDITION_DL_STEPUP}"
  if [ -n "${VT_TAU-$EDITION_VT_TAU}" ]; then
    export SGLANG_RND_VT_TAU="${VT_TAU-$EDITION_VT_TAU}"
  fi
  export PYTHONPATH="$REPO/flashinfer-vt:$REPO/python"
  export FLASHINFER_WORKSPACE_BASE
else
  unset SGLANG_RND_VT_TAU SGLANG_RND_DL_TAU_BY_S
  export SGLANG_RND_DL_TAU=0 SGLANG_RND_DL_STEPUP=0
fi
case "$OEA:${OEA_STATS:-0}" in
  [01]:[01]) ;;
  *) echo "[serve-fast] OEA and OEA_STATS must be 0 or 1" >&2; exit 2 ;;
esac
case "${OEA_K0:-6}" in [1-9]) ;; *) echo "[serve-fast] OEA_K0 must be 1..9" >&2; exit 2 ;; esac
case "${OEA_MIN_ROWS:-16}" in
  [2-9]|1[0-6]) ;;
  *) echo "[serve-fast] OEA_MIN_ROWS must be 2..16" >&2; exit 2 ;;
esac
unset SGLANG_OEA_WIDTH SGLANG_OEA_MAX_BS SGLANG_OEA_STATS SGLANG_OEA_MODE SGLANG_OEA_MIN_ROWS
if [ "$OEA" = 1 ] && [ "$PROFILE" != ngram ]; then
  export PYTHONPATH="$REPO/flashinfer-vtoea:$REPO/python"
  export FLASHINFER_WORKSPACE_BASE
  export SGLANG_OEA_K0="${OEA_K0:-6}" SGLANG_OEA_MODE=oea SGLANG_OEA_MIN_ROWS="${OEA_MIN_ROWS:-16}"
  export SGLANG_OEA_STATS="${OEA_STATS:-0}"
  export SGLANG_OEA_CTL_FILE="${SGLANG_OEA_CTL_FILE-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/oea/flash-next.ctl}"

else
  export SGLANG_OEA_K0=0
fi
export SGLANG_ROUTER_FAST_TOPK=${SGLANG_ROUTER_FAST_TOPK:-1} SGLANG_OPT_SPEC_SPARSE_VERIFY=${SGLANG_OPT_SPEC_SPARSE_VERIFY:-1}
export SGLANG_OPT_SPEC_SPARSE_TOPK=${SGLANG_OPT_SPEC_SPARSE_TOPK:-1} SGLANG_OPT_GDN_FRONT_OVERLAP=${SGLANG_OPT_GDN_FRONT_OVERLAP:-1}
export SGLANG_OPT_DRAFT_MOE_GEMV=${SGLANG_OPT_DRAFT_MOE_GEMV:-1}
export SGLANG_SPEC_MIN_P=${SGLANG_SPEC_MIN_P:-1}
export SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX=${SGLANG_ENABLE_QSA_SHARED_TAIL_PREFIX:-1} SGLANG_OPT_TRITON_DECODE_ATTN=${SGLANG_OPT_TRITON_DECODE_ATTN:-1}
export SGLANG_OPT_SPEC_SPARSE_RS=${SGLANG_OPT_SPEC_SPARSE_RS:-1} SGLANG_RS_DRAFT_TOPK=${SGLANG_RS_DRAFT_TOPK:-16}
export SGLANG_RS_DRAFT_TEMP_SCALE=${SGLANG_RS_DRAFT_TEMP_SCALE:-0.7} SGLANG_RS_DRAFT_ONEHOT_ABOVE=${SGLANG_RS_DRAFT_ONEHOT_ABOVE:-0.9}
export SGLANG_RS_GREEDY_FAST=${SGLANG_RS_GREEDY_FAST:-1} SGLANG_RS_BLOCK_VERIFY=${SGLANG_RS_BLOCK_VERIFY:-1}
if [ "${PROFILE:-}" != ngram ]; then
  export SGLANG_OPT_RS_EXTEND_TAIL=${SGLANG_OPT_RS_EXTEND_TAIL:-1} SGLANG_OPT_RS_EXTEND_FUSED_CONF=${SGLANG_OPT_RS_EXTEND_FUSED_CONF:-1}
  export SGLANG_OPT_QSA_DRAFT_META_BATCH=${SGLANG_OPT_QSA_DRAFT_META_BATCH:-1} SGLANG_OPT_DRAFT_TAIL=${SGLANG_OPT_DRAFT_TAIL:-1}
  export SGLANG_OPT_COMMIT_GLUE_FUSED=${SGLANG_OPT_COMMIT_GLUE_FUSED:-1} SGLANG_OPT_DRAFT_EXTEND_LENS=${SGLANG_OPT_DRAFT_EXTEND_LENS:-1}
fi
export TARGET_MODEL


[ "$PROFILE" = wa ] && [ "${CONTEXT_LENGTH:-0}" -gt 262144 ] && WA_MEM_FRACTION="${WA_LONG_MEM_FRACTION:-0.96}"
# Reserve is GiB; use the last effective fraction, just as the measured launcher did.
# No hardware probing or ML imports are needed to resolve a launch.
WA_EFFECTIVE_MEM_FRACTION="$("$PYTHON" -I -B -c '
import math, os, sys
try:
    base, reserve, total = (float(os.environ[k]) for k in
                            ("WA_MEM_FRACTION", "DISPLAY_RESERVE_GB", "GPU_MEMORY_GB"))
    if not all(map(math.isfinite, (base, reserve, total))) or not (0 < base <= 1 and reserve >= 0 and total > 0):
        raise ValueError()
    fraction = base - reserve / total
    if fraction <= 0.10:
        raise ValueError()
except (ValueError, OverflowError):
    sys.exit("[serve-fast] invalid WA_MEM_FRACTION / DISPLAY_RESERVE_GB / GPU_MEMORY_GB (remaining fraction must exceed 0.10)")
print(format(fraction, ".12g"))
')"
if [ "$PRINT_ONLY" = 0 ]; then
  for directory in "$HF_HOME" "$HF_HUB_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
      "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$SGLANG_JIT_CACHE_DIR" "$SGLANG_NVFP4_CACHE_DIR" \
      "$SGLANG_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$TORCH_EXTENSIONS_DIR" \
      "$TORCH_HOME" "$NUMBA_CACHE_DIR" "$TRITON_HOME" "$TMPDIR" "$TMP" "$TEMP" \
      "$XDG_RUNTIME_DIR" "$FLASHINFER_WORKSPACE_BASE"; do mkdir -p "$directory"; done
fi
if [ "$PRINT_ONLY" = 0 ] && [ "$SGLANG_OEA_K0" != 0 ] && [ -n "$SGLANG_OEA_CTL_FILE" ]; then
  mkdir -p "$(dirname "$SGLANG_OEA_CTL_FILE")"; : > "$SGLANG_OEA_CTL_FILE"
  [ -f "$SGLANG_OEA_CTL_FILE.status.jsonl" ] && mv -f "$SGLANG_OEA_CTL_FILE.status.jsonl" "$SGLANG_OEA_CTL_FILE.status.jsonl.prev"
  printf '{"launch": true, "pid": %d, "t": %d, "cfg": {"mode": "oea", "k0": %d, "min_rows": %d}}\n' \
    "$$" "$(date +%s)" "$SGLANG_OEA_K0" "$SGLANG_OEA_MIN_ROWS" > "$SGLANG_OEA_CTL_FILE.status.jsonl"
fi
COMMON=()
[ "$TOKEN_MAP" != none ] && COMMON+=(--speculative-token-map "$TOKEN_MAP")
COMMON+=(--qwen4-exp-dense-fp8 shared_expert,attn,linear_attn,lm_head,mtp_dense
        --speculative-draft-model-quantization modelopt_fp4 --enable-metrics)
if [ "$PROFILE" = wa ] && [ "$CONC" != 1 ]; then
  COMMON+=(--max-running-requests "$CONC" --cuda-graph-bs-decode "${WA_GRAPH_BS[@]}"
           --chunked-prefill-size "$WA_CHUNK")
  if [ "$FLASH_NEXT_EDITION" = normal ]; then
    if [[ ! "$WA_PD_INTERVAL" =~ ^0+$ ]]; then
      COMMON+=(--prefill-decode-interval "$WA_PD_INTERVAL")
    fi
  fi
  if [[ ! "$WA_HICACHE_GB" =~ ^0+$ ]]; then
    COMMON+=(--enable-hierarchical-cache --hicache-size "$WA_HICACHE_GB"
             --hicache-host-memory-mode cache --hicache-write-policy write_through
             --hicache-io-backend kernel --hicache-mem-layout page_first)
  fi
fi
if [ "$PROFILE" = ngram ]; then export SGLANG_OPT_SPEC_SPARSE_RS=0
elif [ "$SGLANG_OPT_SPEC_SPARSE_RS" = 1 ]; then COMMON+=(--speculative-use-rejection-sampling); fi
case "$PROFILE" in
  w4)    MEM_FRACTION="${MEM_FRACTION:-${W4_MEM_FRACTION:-0.935}}" exec "$REPO/serve-local.sh" "${COMMON[@]}" "$@" ;;
  w8)    MAMBA_SLOTS="${MAMBA_SLOTS:-10}" exec "$REPO/serve-local.sh" "${COMMON[@]}" --speculative-num-steps 7 --speculative-num-draft-tokens 8 "$@" ;;
  w16)   MAMBA_SLOTS="${MAMBA_SLOTS:-10}" SGLANG_SHARED_GATEUP_FUSED="${W16_GATEUP:-0}" SGLANG_ROUTER_GEMV="${W16_ROUTER:-1}" exec "$REPO/serve-local.sh" "${COMMON[@]}" --speculative-num-steps 15 --speculative-num-draft-tokens 16 --mem-fraction-static "${W16_MEM_FRACTION:-0.93}" "$@" ;;
  wa)    SGLANG_ADAPTIVE_STEP_A="${SGLANG_ADAPTIVE_STEP_A:-7.943}" SGLANG_ADAPTIVE_STEP_B="${SGLANG_ADAPTIVE_STEP_B:-0.5554}" SGLANG_ADAPTIVE_TARGET_AUTOTUNE="${SGLANG_ADAPTIVE_TARGET_AUTOTUNE:-1}" SGLANG_ADAPTIVE_POLICY="${SGLANG_ADAPTIVE_POLICY:-confidence}" MAMBA_SLOTS="${MAMBA_SLOTS:-$WA_SLOTS}" SGLANG_SHARED_GATEUP_FUSED="${W16_GATEUP:-0}" SGLANG_ROUTER_GEMV="${W16_ROUTER:-1}" exec "$REPO/serve-local.sh" "${COMMON[@]}" --speculative-num-steps 15 --speculative-num-draft-tokens 16 --mem-fraction-static "$WA_EFFECTIVE_MEM_FRACTION" --speculative-adaptive --speculative-adaptive-config "${ADAPTIVE_CONFIG:-$WA_DEFAULT_CONFIG}" --enable-request-time-stats-logging --decode-log-interval 10 "$@" ;;
  ngram) MAMBA_SLOTS="${MAMBA_SLOTS:-10}" exec "$REPO/serve-local.sh" "${COMMON[@]}" --speculative-algorithm NGRAM_CHAIN --speculative-num-steps 15 --speculative-num-draft-tokens 16 "$@" ;;
  *) echo "usage: $0 {w4|w8|w16|wa|ngram} [extra sglang args]" >&2; exit 2 ;;
esac
