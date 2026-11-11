# Limitations and measurement noise

> Moved from the top-level [README](../README.md) on 2026-11-12. The text is unchanged; only heading levels and relative link paths were adjusted.

This section concerns the historical September and 2026-10-02 configurations, not the four-request fork release above.

- **Not published, so not exactly reproducible:** the fine-tuned MTP draft heads and the private token maps
  were derived from the author's private conversations and agent transcripts. With the public
  checkpoint's original MTP head, the historical prose/agent runs had lower acceptance (the fine-tuned
  heads added roughly 5–10 % there); this is not a general speed guarantee. The published public-data token map shares 72.0 % of its ids with the private
  one. Measured together (original head + public map, 2026-10-06), `wa` was about 20 % slower on code-edit
  and 8–15 % slower on prose and agent-loop than with the private pair (table in the 2026-10 update). Or build your own token map from text you may use (`tokenmaps/README.md`),
  and train your own head on your own data with [`docs/train-your-own-mtp-head.md`](train-your-own-mtp-head.md)
  (code in [`mtp-train/`](../mtp-train/)). A recipe that retrains the head on public data only, with the
  result of one run and no weights, is in [`docs/public-data-mtp-head.md`](public-data-mtp-head.md).
- **Workloads are narrow.** The headline table uses one prompt per workload; code-edit is an extremely
  repetitive mechanical edit where wide speculation shines (11.9 tokens accepted per step). The
  8-prompt table is the fairer summary. Thinking is on, so most counted tokens are reasoning text.
- **Noise.** The GPU also drives a 4K 160 Hz desktop; the compositor preempts the GPU and adds up to
  7.5 % to step time (measured with compositing on vs off), and client-side t/s of identical configurations varied by ±10–15 % across sessions.
  Comparisons were therefore made within one session, with CUPTI kernel time, with acceptance measured
  separately, and later with an 8-prompt ABBA protocol ([`docs/measurement.md`](measurement.md)).
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
  issues are listed in [`docs/optimizations.md`](optimizations.md) §H7: two only matter with several
  running requests or other callers; the third (rare exact ties at the sparse-verify cut) can also
  touch a single sampled request.
- The baseline row was measured with the harness's recommended sampling (temperature 0.2–0.7), the
  other rows greedy.
- Historical test issue: two unit tests of the rejected `NGRAM_CHAIN` drafter failed after a later
  logging change (`test_ngram_chain_worker.py`). This test file was removed in the 2026-10-27 update;
  the drafter is not used by the measured or default profiles, but the experimental `ngram` profile
  in `launch/serve-fast.sh` selects it.
