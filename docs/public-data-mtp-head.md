# Retrain the MTP head on public data only

The fine-tuned MTP draft head used in most of this repository's measurements was trained on the
author's private data and is not published (see [`train-your-own-mtp-head.md`](train-your-own-mtp-head.md)).
The public-components results use the checkpoint's stock MTP head instead. This page describes a
third option: a head retrained **only on public data**, that is, public prompts and the model's own
continuations of them. It gives the method, the scripts and the result of one run.

- **No weights are published.** A retrained head is a derivative of the model weights. Build your
  own with the scripts in [`mtp-train/recipes/public-data/`](../mtp-train/recipes/public-data/).
- **No data is published.** The scripts fetch the public sources at pinned revisions and rebuild
  the prompt set. The continuations come from your own server.
- **One run.** All numbers below come from one training run and one server A/B on one machine
  (2026-11-03 and 2026-11-04). Four steps did not go as preregistered. Read
  [Deviations](#deviations-from-the-preregistration) before relying on the numbers.
- **Script status.** The scripts are the public form of the scripts that produced this run. For
  publication, machine-specific guards were removed and paths became arguments. Three were checked
  against this run's records:
  - `build_prompts.py` rebuilt the run's prompt file byte for byte from the fetched sources;
  - `dump-hook.patch`, applied to the release, gives exactly the source files of the run's dump
    server;
  - `analyze_ab.py` reproduced the A/B tables below from the run's per-start records.

  The others have passed CPU tests against fakes only. In this form, none has been run against a
  live server. Run them on a small budget first.

Contents:

1. [Result](#result)
2. [What you need](#what-you-need)
3. [Data](#data)
4. [Steps](#steps)
5. [Offline gate](#offline-gate)
6. [Server A/B and its preregistered lines](#server-ab-and-its-preregistered-lines)
7. [Deviations from the preregistration](#deviations-from-the-preregistration)
8. [Limits](#limits)
9. [Licences and terms](#licences-and-terms)

---

## Result

The server A/B ran on the 2026-10-27 release: the fork's `flash-next-fast` branch at `d21289f`,
normal edition, launcher defaults (`CONC=4`). Both arms used the fork's public token maps, and the
screen was off.

- **A:** the checkpoint's stock MTP head. This is the public-components configuration.
- **H:** the head retrained on public data.

There were 8 server starts in the order A H H A A H H A. Each start ran the 32 held-out prompts of
`results/TABLES.md` section 8 once each (8 per genre). Requests went one at a time, greedy, with
thinking on. The output caps were the README's: 12,288 tokens for code edit, 6,144 for prose and
8,192 for agent.

| genre | A acceptance | H acceptance | H/A | p(H>A) | A decode tok/s | H decode tok/s | H/A tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| code edit | 8.95 | 9.57 | 1.069 | 0.029 | 597.3 | 674.7 | 1.130 |
| English prose | 2.42 | 2.50 | 1.031 | 0.014 | 275.0 | 289.7 | 1.053 |
| Japanese prose | 2.55 | 2.97 | 1.166 | 0.043 | 291.7 | 336.2 | 1.152 |
| agent | 3.68 | 4.13 | 1.121 | 0.014 | 374.1 | 424.5 | 1.135 |

- **Acceptance** is the server's mean number of tokens per verify step, bonus included. It is the
  preregistered primary metric. Decode tok/s is measured on the client and is secondary.
- **Each value** is the mean of the arm's 4 start means.
- **p** is an exact one-sided permutation p over all 70 ways to split the 8 start means 4:4, with
  ties counted. With 4 starts per arm, the smallest possible p is 1/70 ≈ 0.014.
- **Preregistered verdict: "good".**
  - Code edit and agent met the gain line (H/A ≥ 1.05 with p ≤ 0.05).
  - Both prose genres stayed at or above 0.98.
  - No genre met the "bad" line.
  - The lines are [below](#server-ab-and-its-preregistered-lines).
- **Cross-check:** arm A is within 3 % of the README's public-components table for the same release
  (normal row, one start on 2026-10-27).

Per-start acceptance, in run order (A1 A2 A3 A4 / H1 H2 H3 H4):

| genre | A | H |
|---|---|---|
| code edit | 9.37, 9.13, 8.57, 8.74 | 9.55, 9.93, 9.56, 9.24 |
| English prose | 2.43, 2.43, 2.41, 2.43 | 2.45, 2.47, 2.56, 2.51 |
| Japanese prose | 2.34, 2.48, 2.71, 2.66 | 2.76, 2.83, 2.62, 3.67 |
| agent | 3.67, 3.69, 3.62, 3.75 | 4.18, 4.12, 4.08, 4.14 |

### One row inflates the Japanese-prose gain

In start H4, one Japanese-prose prompt produced no answer text: its thinking ran to the
6,144-token cap.

- That row's acceptance was 8.90. The same prompt gave 2.31–2.84 in the other seven starts.
- It was the only Japanese-prose row without an answer among all 64.

What this row tells us:

- Greedy outputs still vary between starts because of numerical nondeterminism. A long, highly
  predictable stretch of thinking would give this pattern. We did not read the output text, so the
  cause is not confirmed.
- The head only proposes tokens; the target model's argmax decides the output. This row therefore
  says little about the head itself.

A post-hoc check, not preregistered: dropping that prompt from all eight starts gives
Japanese-prose H/A 1.091 (p 0.057) for acceptance and 1.102 for decode tok/s.

The verdict does not change. The prose genres only need H/A ≥ 0.98, and the gain line was met by
code edit and agent.

---

## What you need

The figures come from one machine:

- one RTX PRO 6000 Blackwell Max-Q (96 GB), which also drives a 4K desktop that holds about
  3.5–4.1 GiB of GPU memory;
- 251 GiB of RAM;
- 48 CPU threads. In the A/B, the server and the benchmark client shared 12 of them.

| stage | time | GPU memory peak, whole device (MiB) | host memory |
|---|---|---:|---|
| extraction (dump server, 1,700 documents) | 17 min 25 s, including the server boot | 93,054 | MemAvailable fell by about 74 GB |
| verify, index, held-out eval set | 31 s | — | small |
| training | 544 steps of about 6.1 s (time cap) | 91,319 | trainer about 29 GB, plus the 90.5 GB dump read through mmap, plus snapshot writes |
| server A/B (8 starts) | 2 h 13 min, 921–1,010 s per start | 93,222–93,336 | about 100 GiB per server, plus 126 GiB of model files read into the page cache at every boot |

- **Training time cap.** The stage has a 60-minute cap and reserves 300 s of it for the final
  evaluation and writes. The trainer itself got 3,295 s and stopped on that limit, before its
  1,000-step limit.
- **Disk.** Our estimate before the run was 135 GiB of new files:
  - 90.5 GB of dump (3,081 files);
  - the training outputs;
  - 13.7 GiB for the written-back model directory. Only the index and the 3 shards that hold
    `mtp.*` tensors are new files; every other file is a symlink to the original.
- **Memory limits.** If you run a stage inside a memory cgroup, size the cgroup for the whole
  working set, including page cache.
  - A 128 GiB limit was too small for training. The trainer, the mmap'd dump and the snapshot
    writes stalled in reclaim during the second epoch. A 170 GiB limit worked.
  - At every server boot, the model's page cache is charged to the server's cgroup. Under a
    128 GiB limit, the cgroup reclaimed its own page cache during boot while the host still had
    free memory. We then dropped our own model files from the page cache during boots; see the
    fourth deviation.

---

## Data

The prompt set reuses the public sources of this repository's public token maps (see
[`tokenmaps/README.md`](../tokenmaps/README.md)). Fetch them with `tokenmaps/public/fetch_public.sh`,
which downloads them at pinned revisions. The script also downloads the Wikipedia shards used by the
maps; the prompt set does not use them.

| genre | source | prompt text | prompts (build / held-out) | licence |
|---|---|---|---:|---|
| `en_chat` | `HuggingFaceH4/ultrachat_200k` (test_sft) | first user turn | 300 (270 / 30) | MIT |
| `ja_chat` | `llm-jp/oasst2-33k-ja` | first user turn | 300 (270 / 30) | Apache-2.0 |
| `reasoning` | `open-thoughts/OpenThoughts-114k` (one shard) | first user turn | 250 (225 / 25) | Apache-2.0 |
| `tools` | `NousResearch/hermes-function-calling-v1` | first user turn, with that conversation's tool definitions | 300 (270 / 30) | Apache-2.0 |
| `swe` | `nebius/SWE-agent-trajectories` (one shard) | the issue text of the first user turn, with five generic file tools | 200 (180 / 20) | CC BY 4.0 |
| `code` | CPython 3.13.0, Go 1.23.0, React 18.3.1, Vue core 3.5.12, Tokio 1.40.0, curl 8.10.1 | a 40–250-line chunk of one source file, with one of six code-edit instructions | 350 (315 / 35) | PSF-2.0, BSD-3-Clause, MIT, MIT, MIT, curl |

- **Selection** is deterministic: a fixed seed and fixed counts. In each genre, a fixed tenth is
  held out. Our prompt file is 6,018,320 bytes with sha256
  `b44bd0c8c26164bb58decd58641bdc43a437592df2a5f540293625543a9fdffb`.
- **Rendering.** Prompts are rendered with the model's own chat template and its default thinking
  settings, and capped at 6,000 input tokens.
- **Continuations** come from the release server (normal edition) at temperature 0.6, top-p 0.95
  and top-k 20, with up to 3,072 new tokens and 4 requests at a time.
- **Corpus.** Each document is the rendered prompt plus the generated token IDs.
  - Build: 1,530 documents, 3,909,706 tokens.
  - Held-out: 170 documents, 435,556 tokens. Held-out documents never enter training.
- **Overlap with the A/B prompts.**
  - The 32 A/B prompts are rendered the way the benchmark sends them.
  - Every build document is compared with them by full-sequence hashes and by hashes of all
    64-token windows.
  - Matching build documents would be dropped. In our run none matched: 0 documents, 0 tokens.
  - The scripts compare hashes only. They never print or save the text or token IDs of the A/B
    prompts.

---

## Steps

The commands, inputs and outputs are in
[`mtp-train/recipes/public-data/README.md`](../mtp-train/recipes/public-data/README.md). In order:

1. **Fetch** the public sources: `tokenmaps/public/fetch_public.sh`.
2. **Build the prompts.** `build_prompts.py` builds the 1,700 prompts and their manifest. The
   manifest records each source's revision, sha256 and licence.
3. **Generate.** Serve the release in its normal edition with `CONC=4` and the hierarchical cache
   off. `generate.py` then generates the continuations.
4. **Build the corpus.** `workload_hashes.py` hashes the A/B prompts. `build_corpus.py` then builds
   the token-ID corpus and drops overlapping build documents.
5. **Extract.** Apply `dump-hook.patch` to a copy of the release checkout and start the dump
   server from the copy with `dump_server.py`. It uses the release's `w4` profile, one request at a
   time. Then run `extract.py` to extract the teacher-forced dump.
   - The patch adds the dump hook of
     [guide §4](train-your-own-mtp-head.md#4-build-the-server-with-the-dump-hook) to the release
     itself, so no separate patch series is needed.
   - Each document is cut into chunks of at most 2,048 tokens. Radix cache is off, so every chunk
     is prefilled in full.
   - The first 200,000 tokens are timed, and the time for the rest is estimated from them.
   - For 48 held-out documents, the server also generates up to 128 tokens at temperature 0. These
     are for the offline renewal gate.
   - `extract.py` then verifies the dump and builds the index and the held-out eval set.
6. **Train and gate.** `train_head.py` trains from the checkpoint's stock head and applies the
   offline gate.
   - It uses the first-stage settings of
     [guide §10](train-your-own-mtp-head.md#10-train), with the 60-minute cap.
   - It keeps the best head and one rolling snapshot, without optimizer state.
7. **Write back.** Only if the gate passes, `writeback.py` writes the best head into a new model
   directory and `model_check.py` checks it. The source model is never modified.
8. **A/B.** Run the server A/B, then analyse it with `analyze_ab.py`.

---

## Offline gate

The gate scores renewal acceptance at k=3 and k=15. Renewal acceptance is simulated draft acceptance
with renewal at each verify step, as in
[guide §11](train-your-own-mtp-head.md#11-evaluate-offline-with-the-renewal-evaluator). It uses the
48 held-out documents generated at temperature 0.

- **Pass** if one score improves by at least 1 % over the stock head and the other does not worsen
  by more than 1 %.
- **A missing score** fails the gate.
- The gate selects a candidate only. The server A/B decides.

Our run, scored by argmax-label replay (see the first deviation below):

| | stock head | retrained head | change |
|---|---:|---:|---:|
| renewal acceptance, k=3 | 2.7258 | 2.8484 | +4.50 % |
| renewal acceptance, k=15 | 3.5129 | 3.9896 | +13.57 % |

The teacher-forced held-out eval ran during training:

- step 0: agreement@1 0.7125, renewal 3.119;
- best step (200): agreement@1 0.8011, renewal 3.411.

---

## Server A/B and its preregistered lines

These lines were written down and hash-frozen before the first start.

- **Arms and order:** A H H A A H H A, 4 starts per arm. Every start is required, and there are no
  optional arms.
- **Workload:** the 32 held-out prompts, once each per start, in a fixed order. No reruns or
  exclusions are used to pick numbers.
- **Primary metric:** server acceptance.
  - Per start and genre, average the 8 prompts.
  - An arm's value is the mean of its 4 start means.
  - Decode tok/s is reported the same way, for reference.
- **Test:** an exact one-sided permutation over the 70 labelings of the 8 start means, with ties
  counted.
- **Decision lines:**
  - **"Good":** code edit or agent H/A ≥ 1.05 with p(H>A) ≤ 0.05, and both prose genres H/A ≥ 0.98.
  - **"Bad":** any genre H/A ≤ 0.97 with p(H<A) ≤ 0.05.
  - **"Report the numbers":** neither line holds.
  - If both lines hold, both are reported and the headline is "bad".
- **"Not decidable":** if any start fails the runner's checks, the run stops. It is never judged on
  fewer starts.

`analyze_ab.py` applies the same lines to one fnbench genre file per start.

---

## Deviations from the preregistration

Four things differ from the plan.

### 1. The gate's scoring was changed after the result was seen

- **The preregistered gate** used exact greedy renewal. Every generated token of the 48
  temperature-0 documents had to equal the argmax in the dump.
- **What happened:** the dump agreed on only 5,633 of 6,144 tokens (0.917), so the gate could not
  pass by construction. It denied the head (`missing_greedy_renewal`). That decision is kept as
  `gate-exact-greedy.json`.
- **The change:** `regate.py` re-scored the same documents with the dump's argmax as the label
  (argmax-label replay).
- **What did not change:** the gain lines, the evaluation reports and the frozen A/B.

### 2. The 0.90 agreement line for replay was also chosen after the result was seen

Replay is allowed only when the dump's greedy agreement is at least 0.90.

### 3. Seed caches for the A/B were extended between attempts

- The first A/B attempt stopped during its first start's boot. The server compiled a kernel module,
  and the harness stops on any compiler process.
- Before the next attempt, we copied four kernel modules into every arm's seed cache. The same
  release tree had built them at runtime in an earlier run. Nothing was compiled.

### 4. Page-cache drops differed between arms

- The second attempt stopped during its first start's boot on the memory-pressure line. Neither
  the first nor the second attempt measured anything; both are recorded as "not decidable".
- In the third attempt, the one reported here, we dropped our own model files from the page cache
  (`posix_fadvise(DONTNEED)`). Drops ran during server boots and between starts, never while
  measuring.
- The trigger was widened during start H2's measurement. From then on, a drop also ran when the
  server's memory cgroup came within 24 GiB of its limit.
- Drops per start:

  | start | A1 | H1 | H2 | A2 | A3 | H3 | H4 | A4 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | drops | 1 | 0 | 1 (by hand, during boot) | 5 | 5 | 5 | 5 | 5 |

- While measuring:
  - Memory pressure stayed at or below 0.54 in every start.
  - The lowest free memory was about 74 GiB in A1 and H1, one start per arm, and 97–101 GiB in the
    other six starts.

---

## Limits

- **One run, one machine.** This is one training run and one 8-start A/B. The gain may differ on
  another run, machine or build.
- **Short prompts.** The A/B prompts are short. The corpus prompts are at most 6,000 tokens, with up
  to 3,072 new tokens. Long-context agent sessions are not represented in either.
- **Offline gate.** The gate only selects a candidate. Only the server A/B tells whether serving
  speed improves.
- **Untested scripts.** Apart from the three checks against this run's records, the scripts in their
  public form were tested on the CPU only (see the script status at the top).

---

## Licences and terms

- **Prompt sources:** the licences are in the table above. If you redistribute the prompts or
  anything built from them, keep the required notices. That includes the attribution for
  SWE-agent-trajectories (CC BY 4.0, by Nebius). The prompt manifest records each source file's
  revision and sha256.
- **Generated continuations:** they come from the model at temperature 0.6. Check the model's terms
  before you redistribute generated data; the source licences alone do not settle it. This
  repository does not distribute any.
- **Token maps:** the public maps used in both A/B arms are CC BY-SA 4.0 (see `NOTICE`).
- **Code:** the scripts are Apache-2.0, like the rest of `mtp-train/`.
- **Weights:** a written-back checkpoint is a derivative of Qwen3.8-Flash-Next. Qwen3.8-Flash-Next
  is under the Qwen Community License 1.0, not Apache-2.0. See
  [guide §12](train-your-own-mtp-head.md#12-write-back-into-a-checkpoint-copy) and the README's
  model cards before you share one. No weights are published here.
