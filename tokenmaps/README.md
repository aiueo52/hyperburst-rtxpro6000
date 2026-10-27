# Reduced draft vocabulary (hot token map)

The MTP draft head normally scores all 248,320 vocabulary rows at every draft step. With
`--speculative-token-map <file.pt>` SGLang restricts the draft head to a list of "hot" token ids
(FR-Spec style); the draft proposes only tokens from that list, the target still verifies with the
full vocabulary, so the map never decides on its own what is emitted — a token outside the map
simply cannot be drafted and costs acceptance. The output is not guaranteed identical, though: with
the default singleton expert pruning, which experts the target drops depends on every token in the
verify batch, so a different map (like any draft change) can change the target's output.

In the historical single-request build, the map also let the draft head stay in FP8 and run
through the Triton W8A16 GEMV. In that private-asset configuration, full-vocabulary drafting
measured W16 code-edit 520 → 357 t/s (see `docs/rejected.md`); this is not a measurement of the
public maps or the 2026-10-27 fork.

## Historical September / 2026-10-02 private maps (not published)

| map | built from | rows | used |
|---|---|---:|---|
| `hot_32768` | token frequencies of the *generated* side of the author's own agent sessions and chat logs (private) | 32,768 | 2026-09-02 … 09-04 |
| `hot2_49152` | 50 % the same private corpus + 50 % the target model's own greedy outputs (argmax) on the MTP-training prompts | 49,152 | from 2026-09-04, including the September headline and 2026-10-02 private-asset runs |

The private `.pt` files, the token counts and any statistics of the source corpus are **not** in this
repository because they were derived from private conversations. The blended map
replaced the frequency-only 32k map on 2026-09-04; its offline acceptance evaluation used private
data and is not published. The measured kernel step time was essentially the same
(`docs/optimizations.md` A1). The scripts that built them are in `as-used/` for reference; they read
private inputs (paths from environment variables) and are not runnable as is.

## The public maps (`public/`, 2026-10)

`public/public_49152.pt` is a 49,152-row map built **only from public data**, so that a "public
components only" configuration (the checkpoint's original MTP head + this map) can be run by anyone.
It is not the map behind the historical private-asset speed numbers: `hot2_49152` stays private,
and the two maps share 35,404 of their 49,152 ids (72.0 %). Its historical in-server speed, together
with the original MTP head, is in `results/TABLES.md` section 8 and the README's 2026-10 update
(`wa` greedy on the 2026-10-02 build, measured 2026-10-06: about 20 % slower than the private pair
on code-edit, 8–15 % on the other workloads). Those differences are specific to those runs.

The 2026-10-27 **fork** instead defaults to the public 49,152-row draft map, the public 32,768-row
recurrent-head map below and the stock MTP head from the checkpoint. The README's October 27
BetterBench table and its public-components per-genre table use those public assets; its private-components
per-genre table uses the author's private head and maps instead. Of that update's run files, only the per-genre
runs are published (`results/runs-1027/genre/`); the other files in `results/` are earlier runs. In the
fork the maps live directly under `tokenmaps/`; in this
companion repository they live under `tokenmaps/public/`.

| source | text used | weight | licence |
|---|---|---:|---|
| `code` | source files of CPython 3.13.0, Go 1.23.0, React 18.3.1, Vue core 3.5.12, Tokio 1.40.0, curl 8.10.1 | 0.25 | PSF-2.0, BSD-3-Clause, MIT, MIT, MIT, curl |
| `enwiki` | English Wikipedia, one shard of the 20231101 dump (`wikimedia/wikipedia`) | 0.10 | CC BY-SA 3.0 / GFDL |
| `jawiki` | Japanese Wikipedia, one shard of the 20231101 dump | 0.10 | CC BY-SA 3.0 / GFDL |
| `en_chat` | assistant turns of `HuggingFaceH4/ultrachat_200k` (test_sft) | 0.10 | MIT |
| `ja_chat` | assistant turns of `llm-jp/oasst2-33k-ja` | 0.10 | Apache-2.0 |
| `reasoning` | assistant turns of `open-thoughts/OpenThoughts-114k` (one shard) | 0.20 | Apache-2.0 |
| `agent_swe` | "ai" turns of `nebius/SWE-agent-trajectories` (one shard) | 0.10 | CC BY 4.0 |
| `agent_tools` | "gpt" turns of `NousResearch/hermes-function-calling-v1` | 0.05 | Apache-2.0 |

Every file is pinned to a dataset revision or release tag; `public/public_49152.sources.json` lists
them with their sha256, the per-source character and token totals and the held-out coverage
(95.3–99.8 % per source; 10 % of documents held out). Conversations contribute only the generated
side (assistant turns), as for the private map, but there is **no self-generated target output** in
this map, which is the main difference from `hot2_49152` (50 % self-generated).

Licence and attribution: both public maps and both source records are offered under **CC BY-SA 4.0**,
not Apache-2.0, because they are derived from Wikipedia text (CC BY-SA 3.0) and SWE-agent-trajectories
(CC BY 4.0); see `NOTICE` ("Data-derived files") for the attributions. The maps contain token ids
only, no text. The raw downloads are **not** in this repository; fetch them yourself. Run the following
from this companion repository's root to rebuild both sizes with its preparation and build scripts:

```bash
bash tokenmaps/public/fetch_public.sh ~/datasets/public-tokenmap          # ~1.1 GB, writes SHA256SUMS
python tokenmaps/public/prep_public.py --raw ~/datasets/public-tokenmap/raw --out ~/datasets/public-tokenmap/text
T=~/datasets/public-tokenmap/text
python tokenmaps/build_hot_vocab.py --tokenizer $HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
    --jsonl code=$T/code.jsonl:0.25 --jsonl enwiki=$T/enwiki.jsonl:0.1 --jsonl en_chat=$T/en_chat.jsonl:0.1 \
    --jsonl jawiki=$T/jawiki.jsonl:0.1 --jsonl ja_chat=$T/ja_chat.jsonl:0.1 --jsonl reasoning=$T/reasoning.jsonl:0.2 \
    --jsonl agent_swe=$T/agent_swe.jsonl:0.1 --jsonl agent_tools=$T/agent_tools.jsonl:0.05 \
    --sizes 32768 49152 --seed 7 --prefix public --out-dir ~/tokenmap-work/
```

From the prepared text, the 49,152-row build was checked to reproduce the published file byte for byte
(sha256 `48164e62…62dae5d`, checked on 2026-10-06 on the CPU). The 32,768-row output is a rebuild using
the recorded ranking recipe, not a verified byte-for-byte reproduction (see below). Preparation is
deterministic, but it was run only once, so a different pyarrow version could in principle change the
prepared text. The scripts do not regenerate the published provenance JSON.

For the historical 2026-10-02 launcher, serve with
`TOKEN_MAP=$PWD/tokenmaps/public/public_49152.pt ./launch/serve-fast.sh wa`.
For the 2026-10-27 fork, both public maps are already the defaults; to use rebuilt maps, set
`TOKEN_MAP` and `FLASH_NEXT_RECUR_HEAD` to their paths when launching the fork's `serve-fast.sh`.

### 2026-10-27: 32,768-row public map

`public/public_32768.pt` contains 32,768 sorted token ids, including all 33 added tokens. Its
sha256 is `373292b6b074c0e34d51eb77ad31b0ddd893f036f27e750234378b3ca6402487` (118,283 bytes).
`public/public_32768.sources.json` records it as a subset of `public/public_49152.pt` (sha256
`48164e62467c0be4ff4ea3ef2c18685d3bff5af1fecc4809a8690f5d362dae5d`). The new map and its source
record are also **CC BY-SA 4.0**; see `NOTICE` ("Data-derived files").

The source revisions, hashes, licences and build weights match the 49152 record above; no private
transcripts or self-generated target outputs are included. Held-out coverage is 90.69–99.42 % per
source. The release provenance names `build_public_map.py` (seed 7, 10 % of documents held out,
weighted normalised training-token frequencies, added tokens first, stable first-occurrence ties)
and `prep_public.py`, with **`included: false` for both**. Those flags describe the fork release
and are preserved. This repository's `tokenmaps/build_hot_vocab.py` implements the recorded ranking
recipe, and `tokenmaps/public/prep_public.py` prepares the public inputs; use the command above
with `--sizes 32768 49152` to rebuild the maps without the unavailable `build_public_map.py`.
The 32,768-row rebuild has not been checked byte for byte against the published map.

The prep rule and per-source document/character totals match the 49152 record, but the `prep`
blocks are not identical: 32768 records `script: prep_public.py` and `included: false`, while
49152 records `script: tokenmaps/public/prep_public.py` and no `included` field. We therefore do
not claim they came from the same prep run.

## Building your own

1. Collect text that looks like what you will generate. The best single source is the target's own
   greedy output on prompts representative of your use (this is exactly what the verifier compares
   against):

   ```bash
   ./launch/serve-fast.sh w4 &          # any profile; TOKEN_MAP unset is fine for this step
   python tokenmaps/collect_selfgen.py --endpoint http://127.0.0.1:8001/v1 \
       --prompts my_prompts.jsonl --out ~/tokenmap-work/gen.jsonl --max-tokens 2048
   ```

2. Build maps (CPU only; needs `tokenizers`, and `torch` to write `.pt`):

   ```bash
   python tokenmaps/build_hot_vocab.py --tokenizer $HOME/models/RadixArk/Qwen3.8-Flash-Next-NVFP4 \
       --jsonl selfgen=$HOME/tokenmap-work/gen.jsonl:1 --text corpus=some_corpus_you_may_use.txt:1 \
       --sizes 32768 49152 --out-dir ~/tokenmap-work/
   ```

3. Serve with it: `TOKEN_MAP=$HOME/tokenmap-work/hot_49152.pt ./launch/serve-fast.sh wa`.

`--out` / `--out-dir` are required. The generated text and the maps are derived from your prompts and
corpora, so both scripts refuse an output path inside the repository unless you pass
`--allow-in-repo` (inside `tokenmaps/`, `.pt`, `.json`, `.jsonl` and `.txt` files are git-ignored;
elsewhere in the repository they may not be, so do not commit them).

Historical draft-map tuning notes (not measurements of the 2026-10-27 recurrent-head map):

- The historical draft W8A16 GEMV tile tables were tuned for 49,152 rows. An earlier undersized
  private frequency-only map lost ~10 % t/s at W16 on the public synthetic workloads; the suspected
  cause was an untuned GEMV shape. A sweep over smaller sizes was run on private data only; no
  result of it is published. This does not rule out the fork's 32,768-row recurrent-head map,
  which has a different role.
- Aim for held-out coverage of the target's argmax tokens as close to 100 % as you can get; the
  script prints it per source (10 % of documents are held out).
- All added/special tokens are always included.
- Prefer self-generated output over a public corpus alone (web text, benchmark questions): a public
  corpus does not contain the model's reasoning style or the domain you actually decode.

Tests (CPU, no server): `cd tokenmaps && python -m pytest -q tests` checks the output-path guard.
