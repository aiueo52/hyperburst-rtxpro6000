# Public-data MTP head recipe

This recipe uses only the pinned public sources in `tokenmaps/public/`. It ships
no data, generated continuations, dumps, or weights. Keep all runtime outputs
outside this checkout. Review the source licenses and the model terms before
using or redistributing any result; source licenses do not automatically license
model-generated continuations.

Apart from `dump_server.py`, which runs the dump server in the foreground, these
scripts do not start or stop servers; you manage the servers and their lifetime.
The included tests use synthetic CPU-only inputs and fake servers, not a real
model. Against the reported run's records, `build_prompts.py` rebuilt the
prompt file byte for byte, `dump-hook.patch` gives exactly the dump server's
source files, and `analyze_ab.py` reproduced the A/B numbers; the scripts have
not been run against a live server in this form. The result of
that run, its resources and its deviations from the preregistration are in
[`docs/public-data-mtp-head.md`](../../../docs/public-data-mtp-head.md).

## Setup and inputs

All servers run from the fork's `flash-next-fast` checkout at `d21289f`, the
release in the top-level README, with its venv. Extraction uses a patched copy
of that checkout (step 4). Python needs the `mtp-train` dependencies; prompt
building also needs `pyarrow`, `tokenizers`, and `jinja2`. The companion
benchmark imports require Python 3.12 or newer. The reported run used the
serving venv's Python for every stage. Nothing in this recipe downloads
dependencies or data. Run each server in its own terminal with the same
variables. Fetch the raw files with
`tokenmaps/public/fetch_public.sh DEST` (it writes them to `DEST/raw`);
`tokenmaps/public/public_32768.sources.json` pins their revisions and hashes.

```bash
export REPO=/path/to/companion                  # this repository
export FORK=/path/to/flash-next-fast            # the fork checkout at d21289f
export PY=/path/to/venv/bin/python
export WORK=/path/to/work
export RAW=/path/to/public-tokenmap/raw         # fetch_public.sh DEST/raw
export MODEL=/path/to/Qwen3.8-Flash-Next-NVFP4  # the source checkpoint
export TOKEN_MAP="$FORK/tokenmaps/public_49152.pt"
export SECONDARY_MAP="$FORK/tokenmaps/public_32768.pt"
export ENDPOINT=http://127.0.0.1:8001           # serve-fast.sh default PORT
export R="$REPO/mtp-train/recipes/public-data"
mkdir -p "$WORK"
cd "$REPO/mtp-train"
```

All paths are explicit arguments (some scripts also accept documented environment
variables). Use the same source model, tokenizer, template, and primary token map
through extraction, offline evaluation, and A/B serving. The secondary map is
recorded as **unevaluated**, not silently treated as a second successful gate.
Outputs must not replace inputs. Dump references remain confined to their
selected root, including intermediate symlink targets.

## 1. Check the source checkpoint and build prompts

```bash
"$PY" "$R/model_check.py" --model-dir "$MODEL"
"$PY" "$R/build_prompts.py" --raw "$RAW" \
  --sources "$REPO/tokenmaps/public/public_32768.sources.json" \
  --model-dir "$MODEL" --output-dir "$WORK/prompts"
```

`model_check.py` checks the index and tensor headers without loading the weights.
`build_prompts.py` checks pinned raw-file hashes, uses the model's local tokenizer
and default chat template, and writes `prompts.jsonl` plus `manifest.json`.
Selection preserves the reference seed, source ordering, length cap, six genre
counts, and per-genre held-out split: 1,700 prompts in total. The five SWE tool
schemas are embedded in the script. The manifest records source/license and
asset hashes, never local filesystem paths. Identical raw bytes and model assets
are required for identical prompt JSONL bytes. The reported run's
`prompts.jsonl` is 6,018,320 bytes with sha256
`b44bd0c8c26164bb58decd58641bdc43a437592df2a5f540293625543a9fdffb`.

## 2. Generate continuations

Serve the release's normal edition with the stock head, four requests at once
and HiCache off, as in the reported run. It uses the fork's public token maps
by default. Do not set the dump environment variables:

```bash
cd "$FORK" && TARGET_MODEL="$MODEL" FLASH_NEXT_EDITION=normal CONC=4 \
  FLASH_NEXT_HICACHE_GB=0 bash serve-fast.sh wa
```

Its `/generate` accepts `input_ids` and returns `output_ids`. Wait for
`/health`, then pass the base URL, not `/v1` or `/generate`:

```bash
"$PY" "$R/generate.py" --prompts "$WORK/prompts/prompts.jsonl" \
  --endpoint "$ENDPOINT" --output-dir "$WORK/generated"
```

The client sends temperature 0.6, top-p 0.95, top-k 20, and a 3,072-token
continuation cap, with four concurrent requests. It writes successful ID-only
rows to `outputs.jsonl` and a hash-bound `generation.json` receipt. Rerun the same
command to resume: completed identities are validated and skipped, an interrupted
final JSONL fragment is repaired, and failed requests are retried rather than
published as valid outputs. A changed prompt file or sampling policy is refused.
`--timeout` controls each HTTP request timeout.

## 3. Exclude the A/B workloads and assemble the corpus

```bash
"$PY" "$R/workload_hashes.py" --bench "$REPO/bench" \
  --model-dir "$MODEL" --output "$WORK/ab-hashes.json"
"$PY" "$R/build_corpus.py" --prompts "$WORK/prompts/prompts.jsonl" \
  --outputs "$WORK/generated/outputs.jsonl" --manifest "$WORK/prompts/manifest.json" \
  --model-dir "$MODEL" --ab-hashes "$WORK/ab-hashes.json" \
  --output-dir "$WORK/corpus"
```

The hash builder uses the companion benchmark's public prompt sets (override
with `--sets`), captures requests without making network calls, and persists only
exact-sequence and all-position 64-token shingle hashes. Workload text and token
IDs are neither printed nor saved by this stage.

The corpus builder concatenates already-rendered prompt IDs and continuation
IDs without decoding or adding tokens. It preserves upstream build/held-out
membership, checks provenance and complete generation coverage, and writes
`build.jsonl`, `heldout.jsonl`, `quarantine.jsonl`, and `report.json`. Exact and
shingle overlaps with A/B requests/message bodies are quarantined from build;
held-out duplicate document hashes are also excluded from build. Missing A/B
hash evidence quarantines all build documents and leaves the corpus not ready.
Review exclusions, especially the report's over-ten-percent flag, before moving
on; exclusions are never relaxed automatically.

## 4. Extract hidden states and greedy renewal documents

Stop the sampling server yourself. The dump hook for the fork is
`dump-hook.patch` in this directory. It adds `python/sglang/srt/models/mtp_dump.py`
(the hook from `patches/sglang/mtp-dump/`) and its call sites in `qwen4_exp.py`
and `qwen4_exp_mtp.py`. Applied to the fork at `d21289f`, it gives exactly the
files the reported run's dump server used. Apply it to a copy of the checkout,
including its built FlashInfer trees, so the serving checkout stays unchanged:

```bash
cp -a "$FORK" "$WORK/fork-dump"
patch -d "$WORK/fork-dump" -p1 < "$R/dump-hook.patch"
"$PY" "$R/dump_server.py" --checkout "$WORK/fork-dump" --model-dir "$MODEL" \
  --dump-dir "$WORK/dump"
```

`dump_server.py` resolves the release's `w4` profile with `serve-fast.sh --print`
and runs it in the foreground as in the reported run: one request at a time,
radix cache and CUDA graphs off, a 4,096-token context, 2,048-token chunked
prefill, the public draft map, and top-8 dumps (`SGLANG_MTP_DUMP_TOPK=8`) with a
100 GB cap. It drops the launcher's `--chat-template`, which fails together with
`--skip-tokenizer-init`, and unsets the tap and draft-side dump variables.
It refuses, with the reason, a checkout without the hook, a launcher that does
not run the checkout's own `server.py` and Python trees, and a dump directory
that is not empty. `--print` shows the command and the added environment
without starting anything; the dump directory (mode 0700) is created only when
the server starts. The reported run also set `PREBUILT_CACHE=1`, which only
forbids JIT rebuilds. Its dump took 90.5 GB.

Wait for `/health` before running the client. Do not send unrelated requests while
extracting. The client and server must see the same dump directory. Use a fresh
stage directory; existing stage manifests or receipts are refused. Complete
hook shards produced during server warmup are preserved under
`startup/*.safetensors.quarantined`, with their original names and content hashes
bound into receipts. Their changed suffix keeps recursive index/evaluation scans
from treating warmups as training documents. Pending hook writes must finish
before this quarantine step; incomplete shards are never silently accepted.

```bash
"$PY" "$R/extract.py" --documents "$WORK/corpus" --dump-dir "$WORK/dump" --plan-only
"$PY" "$R/extract.py" --documents "$WORK/corpus" --dump-dir "$WORK/dump" \
  --endpoint "$ENDPOINT"
```

Extraction preserves document boundaries and held-out membership, times the
first 200,000 tokens, and stops with code 74 if the remaining estimate exceeds
its stage deadline (`--max-seconds`). It additionally generates temperature-zero
held-out continuations and extracts their complete prompt-plus-continuation
states for renewal evaluation. A failed or interrupted extraction is not resumed
as a complete dataset: retain it and use a fresh dump directory/server for a new
attempt.

The default finalization verifies every manifest-bound shard and its IDs/top-8
arrays, then runs `extract/verify_dump.py`, `extract/build_index.py`, and
`extract/build_eval_set.py`. Outputs include `manifest.jsonl`, pilot timing,
`dump-result.json`, `extraction-result.json`, `index.json`, and `eval_fixed.json`.
The extraction receipt also records measured greedy/argmax agreement; a
metadata-only temperature claim is insufficient for an exact-renewal gate.

For separate extraction and CPU finalization, use `--extract-only` in the live
command, stop the server yourself, then run:

```bash
"$PY" "$R/extract.py" --documents "$WORK/corpus" --dump-dir "$WORK/dump" \
  --postprocess-only
```

Stop the dump server yourself after extraction and unset the dump variables
before serving any benchmark requests.

## 5. Train and evaluate the head

```bash
"$PY" "$R/train_head.py" --model-dir "$MODEL" --dump-dir "$WORK/dump" \
  --eval-set "$WORK/dump/eval_fixed.json" --output-dir "$WORK/train" \
  --token-map "$TOKEN_MAP" --secondary-token-map "$SECONDARY_MAP" \
  --max-seconds 3600
```

This uses section 10 of the companion training guide: BF16 computation,
8,192 tokens per microbatch with accumulation 3, soft-target loss, accepted-prefix
rollout k=3, and held-out renewal at k=3 and k=15. The stage has a 60-minute cap
and keeps the last 300 s of it for the final evaluation and writes; in the
reported run the trainer stopped on its time limit after 544 steps, before its
1,000-step limit. It keeps the best weights and one rolling snapshot, without
`latest.pt` or optimizer state, plus history and original/best evaluation
reports. `--dry-run` prints the trainer arguments;
`--tiny --device cpu` is a synthetic rehearsal and can never authorize writeback.
Use a new output directory; existing training results are not overwritten.

`gate.json` remains false until both reports and checkpoint/model/map bindings
are validated. Source identity covers the complete checkpoint content, including
shards, configuration, and tokenizer assets; a changed source cannot authorize
writeback merely because its index stayed unchanged. Older gates without this
content binding are refused. The first identity calculation reads all checkpoint
files; subsequent unchanged identities use the core digest cache. Exact greedy
renewal must improve at least one chain length by
1% without degrading either by more than 1%. Missing or incomparable evidence
fails closed. A training process completing successfully does not mean its gate
passed. The secondary map's binding is provenance only.

If exact greedy renewal is unavailable specifically because the dumped
continuation differs slightly from the target's greedy argmax, an explicit
post-hoc label-replay decision may be requested:

```bash
"$PY" "$R/regate.py" --train-from "$WORK/train" --output-dir "$WORK/replay-gate" \
  --dump-dir "$WORK/dump" --model-dir "$MODEL" \
  --token-map "$TOKEN_MAP" --secondary-token-map "$SECONDARY_MAP"
```

This only accepts a finished `missing_greedy_renewal` decision, rebinds the
extraction evidence and selected temperature-zero documents, and requires a
measured greedy/argmax agreement of at least 0.90 (`LABEL_REPLAY_MIN_AGREEMENT`).
That line was chosen after the reported run's exact gate had been denied. It reads the selected tensors on CPU to bind
the existing renewal-data fingerprint and verify measured agreement; matching
document counts alone do not establish identity. Receipt agreement counts must
exactly match the selected, possibly truncated evaluation data; otherwise the
replay is refused. It creates a separate result with a hard
link to the same checkpoint (same filesystem required), retains the original
as `gate-exact-greedy.json`, and marks the deviation explicitly. It never
silently replaces the original decision or makes sampled text into greedy data.

In the reported run the exact gate was denied with `missing_greedy_renewal`. The
measured agreement was 0.917, not exact, and the replay gate passed. Both the
replay gate and its 0.90 line are post-hoc deviations; the companion document
lists them with the others.

## 6. Write back into a new model directory

Choose the exact-greedy result, or explicitly choose the separate replay result:

```bash
export APPROVED="$WORK/train"
"$PY" "$R/writeback.py" --src "$MODEL" --dst "$WORK/model-with-head" \
  --ckpt "$APPROVED/best-mtp.safetensors" --gate "$APPROVED/gate.json" --verify \
  --status-file "$WORK/writeback.json"
"$PY" "$R/model_check.py" --model-dir "$WORK/model-with-head"
```

The wrapper re-evaluates the measured reports and their hashes rather than
trusting a saved `passed` boolean. A denied gate skips writeback. It never
replaces an existing model directory or modifies source weights. Only MTP
weights are changed; unchanged shards are linked to the source. Keep the source
model available. `--strategy new-shard` is an alternative to the default
rewritten-shard strategy. Verification checks that non-MTP tensors are unchanged.

## 7. Run the public serving A/B and analyze start means

A is the unchanged source head and H is the written-back head. Serve both from
the unpatched fork checkout with the release's `wa` profile, changing only the
model directory. A:

```bash
cd "$FORK" && TARGET_MODEL="$MODEL" FLASH_NEXT_EDITION=normal CONC=4 bash serve-fast.sh wa
```

H:

```bash
cd "$FORK" && TARGET_MODEL="$WORK/model-with-head" FLASH_NEXT_EDITION=normal CONC=4 \
  bash serve-fast.sh wa
```

Everything else is the launcher default, including the 32 GiB hierarchical cache
and the public token maps. The reported starts also set `PREBUILT_CACHE=1` and ran
with no other GPU work and the screen off. Keep the dump variables unset and do
not add `--skip-tokenizer-init`. Keep every other serving setting, token map,
workload, and sampling policy identical. Restart for each of the eight starts in
order **A,H,H,A,A,H,H,A**. Each start needs its own new output file.

After `/health` answers and before measuring, send one unmeasured streaming
request. The release labels its generated-token counter by streaming mode and
its start-up warm-up does not stream, so otherwise the first measured prompt
meets a new counter series and `--require-acceptance` refuses it:

```bash
curl -sN "$ENDPOINT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"flash-next","messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"temperature":0,"stream":true}' \
  > /dev/null
```

Do not count it among the 32 measured public-suite requests.

For each ready server, use the companion benchmark directly:

```bash
PYTHONPATH="$REPO/bench" "$PY" -m fnbench run \
  --endpoint "$ENDPOINT/v1" --engine sglang \
  --workloads code-edit,prose-en,prose-ja,agent-loop --repeats 1 \
  --prompt-sets "$REPO/bench/workloads/sets" --require-acceptance \
  --sampling greedy --model flash-next --label A --out "$WORK/start1.jsonl" \
  --allow-proc python --allow-proc sglang::
```

Change the label and file for each start, not the benchmark suite. Native
fnbench records contain metrics and hashes, not prompt or completion bodies.
The analyzer also accepts the numeric indexed genre-row format. Each file must
cover eight prompts in each of the four genres, with matching identities across
starts; native rows must identify greedy sampling and repeat 1.

```bash
"$PY" "$R/analyze_ab.py" --order A,H,H,A,A,H,H,A \
  "$WORK/start1.jsonl" "$WORK/start2.jsonl" "$WORK/start3.jsonl" "$WORK/start4.jsonl" \
  "$WORK/start5.jsonl" "$WORK/start6.jsonl" "$WORK/start7.jsonl" "$WORK/start8.jsonl"
```

Acceptance length is primary; decode throughput is secondary. The exact
one-sided permutation test uses the eight start means (70 balanced labelings),
not individual prompts as independent samples. The preregistered good condition
requires a 5% gain in code-edit or agent-loop with p<=0.05 and both prose ratios
at least 0.98. A significant 3% loss in any genre is bad, taking precedence if
both independent conditions hold. Otherwise report the numbers. Do not infer
serving gains from the offline training gate alone.

## CPU tests

From `mtp-train/`, using an environment with the test dependencies:

```bash
CUDA_VISIBLE_DEVICES='' "$PY" -B -m pytest -q -p no:cacheprovider recipes/public-data/tests
```

Tests use only small synthetic inputs and mocked server responses. They do not
establish live-server correctness or equality against any previously built real
prompt bank.
