# FlashInfer 0.6.17 — 2026-10-27

`vt.patch` and `vtoea.patch` are alternative, self-contained patches against
pristine `flashinfer-python==0.6.17`, not additions to `01`–`10`. Each modifies
seven wheel files; VT also adds the optional `fused_moe/tactic_rows_policy.py`
loader (no policy JSON).

## Provenance

Targets are the public release overlays at
`flashinfer-overlays/{vt,vtoea}/`. The release assembler,
`tools/assemble_flashinfer.py`, checks every overlay file hash and the final
assembled-tree hash against `flashinfer-overlays/MANIFEST.json`.

The baseline is the pristine `flashinfer-python==0.6.17` wheel: every baseline file was
checked by SHA-256 and size against the wheel's
`flashinfer_python-0.6.17.dist-info/RECORD` before the diffs were made.

## Relationship to older patches

- `01`–`06` and `08` supply the earlier MoE changes. These backends further add
  `vt_lens`/live-row plumbing, variant JIT names, VT's policy loader and VTOEA's OEA path.
- `07` supplies the GDN changes and `09` the JIT changes, retained apart from
  publication notices and comment wording.
- **`10` is excluded:** both backends retain `G2_EXPAND_SF_HOIST=0`, not its
  default 1; `G2_EXPAND_QUANT_UNROLL=2`. Do not append `10`. The historical
  `../apply.sh` does not install these alternative backends.

## Application and verification

Prefer the release assembler. For manual application, use an independent,
RECORD-verified pristine copy; never patch an installed venv or shared cache.
Apply exactly one patch per clean backend root containing `flashinfer/`:

```bash
patch -p1 < /absolute/path/to/vt.patch
# Or use vtoea.patch in a separate pristine copy.
```

Both patches were verified with `patch -p1` on pristine copies: they reproduce
the corresponding public release overlay files byte for byte.
