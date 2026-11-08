# H40: small-M VT MoE prologue

This patch optimizes `fusedBuildExpertMapsSortFirstTokenAndStrides` for small M,
including warp ranking, row arithmetic and coalesced metadata publication.
It targets the public VT source, preserving its first-line modification notice.

## Application order

Use an independent pristine `flashinfer-python==0.6.17` copy containing
`flashinfer/`. Apply `2026-10-27/vt.patch` first, then this patch:

```bash
# From the independent backend root; PATCH_ROOT is this repository's patches directory.
git apply -p1 "$PATCH_ROOT/flashinfer/2026-10-27/vt.patch"
git apply --check -p1 "$PATCH_ROOT/flashinfer/2026-11-08/h40.patch"
git apply -p1 "$PATCH_ROOT/flashinfer/2026-11-08/h40.patch"
```

VT is self-contained against pristine 0.6.17. Do not append it or H40 to
`01`–`10`; the historical `../apply.sh` is unchanged. This patch targets VT,
not the alternative VTOEA backend. The first build takes longer because one
CUTLASS translation unit must be rebuilt.

## Always on once built

H40 is a build-time patch: once the VT module is built with it, H40 is always on.
The fork launcher's `HB_BUNDLE2=0` switches off only the H35a/H38 defaults, not H40.
To run without H40, rebuild the VT module from pristine 0.6.17 plus `vt.patch`
only. After applying or removing this patch, build in a fresh
`FLASHINFER_WORKSPACE_BASE`, so that a module compiled from the other source is
not reused (with `FLASHINFER_P2_NO_NINJA=1`, set by the fork launcher's
`PREBUILT_CACHE=1`, an existing compiled module is loaded as-is).

## Verification

`git apply --check -p1` on a fresh copy of the public VT file exited 0.
Applying the patch yielded SHA-256 `f00f44857e7624d738808d03d1edefc302d23e319048b33651b241d2ddd87bd4`.
The first notice line was preserved. Apart from that line and one comment line
(line 1668, which now points to this README), all bytes match the prototype
modified file (SHA-256 `bbd3b51336e40e43162ea9096e505ed15eecd6b4c3db2d30baed4043bc474e7f`).
The prototype diff applied with fuzz 0 to stock and with offset +1 to public VT.
No build or GPU execution was performed during this preparation.

The reported H40 microbenchmark saved 2.46 microseconds per layer (79 per update)
and produced byte-identical outputs. Bundle throughput was 431.0 → 443.6 tok/s,
+2.91%, or about +3% in one measurement window; see
[measurement limitations](../../../docs/bundle2-measurements.md).
