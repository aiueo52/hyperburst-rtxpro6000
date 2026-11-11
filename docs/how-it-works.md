# How it works (short version)

> Moved from the top-level [README](../README.md) on 2026-11-12. The text is unchanged; only heading levels and relative link paths were adjusted.

This explanation and its speed figures describe the historical September and 2026-10-02 configurations.

At batch size 1 every decode step is a chain of ~1,500–2,500 small GPU kernels that mostly stream
weights from DRAM. Speed comes from two multipliers: **tokens accepted per step** (speculative
decoding) and **step time**. Details, numbers and evidence for each item: [`docs/optimizations.md`](optimizations.md).

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
[`docs/roofline.md`](roofline.md).
