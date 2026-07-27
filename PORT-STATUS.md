# mlx-restormer-swift — port status

**Work order:** P10 in `mlxengine-todo/PORT-QUEUE.md` — Restormer, "one port, several products."

Upstream: [`swz30/Restormer`](https://github.com/swz30/Restormer) — **plain MIT**, Copyright (c) 2022
Syed Waqas Zamir and contributors. Zamir et al., *Restormer: Efficient Transformer for
High-Resolution Image Restoration*, CVPR 2022.

> ⚠️ **Do not confuse the licence.** The *same author's* MIRNet / MIRNetv2 / MPRNet / CycleISP carry
> an **Academic Public License**. Restormer does not — it is unqualified MIT with full commercial
> rights, verified by raw-file fetch. This is the queue's correction and it holds.

---

## Stage 0 ✅ PASSED (2026-07-27)

| Fact | Verified value |
|---|---|
| Licence | **MIT**, zero NC / Academic-Public text anywhere |
| Weights | **First-party GitHub release v1.0**, all 14 checkpoints |
| Parameters | **26,126,644** — 104.51 MB fp32 / 52.25 MB fp16 |
| Load | `strict=True` **clean** (0 missing / 0 unexpected), 494 tensors |
| Constructor | `dim=48`, `num_blocks=[4,6,6,8]`, `num_refinement_blocks=4`, `heads=[1,2,4,8]`, `ffn=2.66`, `bias=False` |

Weight availability was probed **before** planning this row (the P1 lesson). Downloaded
`motion_deblurring.pth` (13,179 downloads) and `single_image_defocus_deblurring.pth` (5,571).

**Motion-deblur and single-image-defocus configs are byte-identical** — one architecture, two
products. `real_denoising.pth` is the third.

### Things that differ from the sibling FFTformer port

1. **The state dict nests under `raw['params']`** (BasicSR convention); FFTformer's was flat.
   Getting this wrong yields zero matching keys.
2. **Both LayerNorm variants are live.** Deblur checkpoints are `WithBias`; the Gaussian-denoising
   ones are `BiasFree`. In FFTformer `BiasFree` was dead code.
   ⚠️ `BiasFree` does **not** centre its output — it is `x / sqrt(var + eps) · weight`, variance
   computed about the mean but no mean subtraction. It is not "LayerNorm minus the bias".
3. **Resampling is PixelUnshuffle / PixelShuffle**, not bilinear. MLX has neither natively.
4. **Level 1's decoder runs at 2·dim, not dim** — there is deliberately no 1×1 reduction after
   `up2_1`, so the concatenated skip stays full width through level 1 and refinement.

Skipping the `dual_pixel_defocus` checkpoint per the queue: it needs dual-pixel sensor data that
user-imported photos never carry, its numbers are ~0.7 dB higher, and they are the ones most often
quoted.

---

## Stage 1 — parity ✅ **ALL GATES GREEN**

`swift run restormer-gate --all <goldens> <weights>`, CPU stream, judged on **relative** error.

| Gate | Result | Worst |
|---|---|---|
| **S0** key contract | ✅ 494 tensors, 26,126,644 params, strict `verify: .all` clean — for **both** checkpoints | exact |
| **S1** primitives | ✅ 4/4 — both LayerNorm variants + the pixel-shuffle pair | 2.6e-06 |
| **S2** blocks | ✅ 3/3 — MDTA, GDFN, TransformerBlock | 2.6e-07 |
| **S3** full model | ✅ 4/4 at cosine **1.00000000** | 1.0e-06 |

### The pixel-shuffle tolerance, justified rather than assumed

`upsample_shuffle` carries a 1e-5 tolerance where `downsample_unshuffle` uses 1e-6. Structural
reason: `up4_3`'s conv accumulates over **384** input channels where `down1_2`'s accumulates over
**48** — eight times the accumulation, ~3× the rounding.

A loosened tolerance is only honest if it still catches what it guards. The failure mode here is the
pixel-shuffle **channel ordering**: splitting the trailing axis as `(r, r, C)` instead of `(C, r, r)`
compiles, runs, and silently scrambles the image. The gate runs that exact mistake as a probe:

```
↳ ordering probe: the (r,r,C) mistake gives rel=1.30e+00 — 493451x the observed
  rounding, so a 1e-5 gate still catches it by ~5 orders of magnitude
```

---

## Remaining

- [ ] Multi-variant `imageRestore` package (motion / defocus / realDenoise) — same request shape as
      NAFNet and FFTformer, so **no contract change**; variant-selected.
- [ ] Conformance gates, footprint measurement, publish to `mlx-community`, registry row.
- [ ] ⚠️ Validation needs **C4** (defocus / focus brackets) and **C5** (noise / ISO brackets) from
      `mlxengine-todo/CORPUS-NEEDS.md`. Restormer's channel attention is spatially **global**, so it
      is weak on *spatially varying* blur — C4 scenes must have depth, or a flat test chart will hide
      exactly the failure worth finding.

## Reproduce

```bash
cd oracle
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch numpy einops safetensors
git clone --depth 1 https://github.com/swz30/Restormer.git upstream
curl -sLO https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth  # into weights/
.venv/bin/python verify.py && .venv/bin/python gen_goldens.py && .venv/bin/python convert.py
cd .. && swift run restormer-gate --all oracle/goldens oracle/converted/motion_deblurring/model.safetensors
```

`oracle/upstream`, `oracle/.venv`, `oracle/weights`, `oracle/converted` are generated — not committed.

---

## Stage 2 — package ✅ (2026-07-27)

`MLXRestormer.RestormerRestorePackage` — the **third** package on `imageRestore`, alongside NAFNet
and FFTformer. Same request shape, **no contract change**, variant-selected.

**Conformance 11/11**, including three tests that pin this port's specific hazards rather than
restating the manifest: the norm-kind ↔ variant mapping, the 494-vs-406 tensor-count difference
(asserted without loading any weights), and the 8-aligned tile defaults.

### 🔑 The variant changes the KEY SET, not just the weights

The denoising checkpoint is `BiasFree` and has **no bias vectors at all**:

| | tensors | params |
|---|---|---|
| deblur (`WithBias`) | 494 | 26,126,644 |
| denoise (`BiasFree`) | **406** | **26,111,668** |

An exact 88-tensor difference — the LayerNorm biases. Verified in **both** directions: the right
config loads strict-clean, the wrong one fails with exactly 88 missing keys.

### Tiling — mandatory, and measured

| | MLX peak | phys |
|---|---|---|
| 1080p **full-frame** | 15.50 GB | 48.02 GB |
| 512² tiled | 2.58 GB | — |
| 1024² tiled | 2.59 GB | — |
| 1080p tiled | **2.62 GB** | ~4.3 GB |

The MLX peak is **flat in resolution** because the peak is one-tile-sized — a bigger image runs more
tiles, not bigger ones.

**Tile geometry is 8-aligned**: three `pixelUnshuffle(2)` stages make the 2×2 grouping grid
phase-sensitive to the tile origin at strides 2, 4 and 8 full-res pixels. Same class as FFTformer's
32-alignment, one third the stride.

**Overlap was chosen on seam visibility, not PSNR** — and the two disagree, which is the interesting
part:

| overlap | PSNR vs full-frame | seam / interior gradient |
|---|---|---|
| 0 | **38.70 dB** (best) | **1.31×** — faint but visible |
| 8 | 36.18 dB | 1.13× clean |
| 16 | 37.22 dB | 1.12× clean |
| 32 (default) | 37.09 dB | 1.20× |
| 48 | 37.12 dB | 1.09× clean |

PSNR-against-full-frame *prefers* overlap 0, which produces a real seam. Full-frame is unattainable
at production sizes anyway, so agreement with it is the wrong objective for a tiler; boundary
continuity is the right one.

## Published

- [`mlx-community/Restormer-motion-deblurring-fp32`](https://huggingface.co/mlx-community/Restormer-motion-deblurring-fp32)
- [`mlx-community/Restormer-defocus-deblurring-fp32`](https://huggingface.co/mlx-community/Restormer-defocus-deblurring-fp32)
- [`mlx-community/Restormer-real-denoising-fp32`](https://huggingface.co/mlx-community/Restormer-real-denoising-fp32)
- [collection](https://huggingface.co/collections/mlx-community/restormer-mlx-6a67b47a017c8be2f293d143)

Round-trip verified: freshly downloaded artifacts re-passed S0 (both norm variants) and S3.
