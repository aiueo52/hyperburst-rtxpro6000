# Bundle 2: single-stream measurements

Bundle 2 combines H35a (fused draft-step confidence kernels), H38 (MTP FC GEMV),
and H40 (small-M MoE prologue optimization). H35a operates only at batch size 1;
its public fork commit is `af368fcdd`. H38 flag support is already in that fork,
and fork commit `899fc4e16` adds the launcher defaults described below.
The measured tree also differs from the public fork beyond H35a, as established
by a measured-tree versus fork diff. These results do not establish performance
of the exact public checkout.

BetterBench concurrency-1 weighted decode throughput increased from **431.0 to
443.6 tok/s (+2.91%)**. This was one measurement window (four ABBA launches),
with both paired comparisons favoring B and a preregistered early GO. Report it
as **about +3% (one measurement window)**, not a precise general speedup; estimated
window-to-window sigma is about 1.2%. The preregistered answer gate passed:
B versus A agreement 0.3235, above the A versus A minimum 0.2922.

The measured configuration was wa with sparse RS on, STACK_D=1, E_DRAFT=1 and
OEA=0, with the H35a/H38 flags enabled. The fork's `serve-fast.sh` exposes these
knobs. With `HB_BUNDLE2=1` (default) it supplies the two flags for exactly this
configuration, which the normal edition's defaults match (the hyper pro edition
sets OEA=1); other profiles and editions are unchanged. `HB_BUNDLE2=0` supplies
neither flag and keeps the caller's environment; explicit caller values are
respected using shell `:-` defaults. H40 is a build-time patch: once the VT
module is built with it, H40 is always on, and `HB_BUNDLE2=0` does not switch it
off. To run without H40, rebuild the VT module without
`patches/flashinfer/2026-11-08/h40.patch`.

Component microbenchmarks: H40 saved 2.46 microseconds per layer (79 microseconds
per update), with byte-identical outputs; H38 saved about 5.6 microseconds per
update. H35a was not measured separately. These component timings are not
independent throughput measurements and should not be added into a speedup.

Measurements used one RTX PRO 6000 Blackwell and a private production MTP head.
Public heads and stock MTP have different acceptance, so absolute throughput
will differ. A similar relative gain is a hypothesis based on shorter draft
steps, not a public-head measurement. H35a/H38 change draft numerics only;
verification preserves the target distribution, while identical seeds need not
produce identical strings. H40 outputs were byte-identical in the reported tests.
H40 requires rebuilding one FlashInfer translation unit, extending the first
build. Deployment on the author's machine is pending a smoke test.

This preparation verified patch application and the launcher's `--print` output
only: across 12 launcher configurations, the change adds the two flags for the
measured configuration and nothing else.
No build, GPU correctness test or public-checkout throughput measurement was run.
