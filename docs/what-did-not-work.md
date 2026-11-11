# What did not work

> Moved from the top-level [README](../README.md) on 2026-11-12. The text is unchanged; only heading levels and relative link paths were adjusted.

These historical September and 2026-10-02 rejected experiments are as useful as the adopted ones;
each has numbers in [`docs/rejected.md`](rejected.md). A few examples:

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
