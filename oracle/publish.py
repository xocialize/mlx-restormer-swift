"""Publish the converted Restormer weights to mlx-community.

One architecture, three products:
  motion_deblurring              -> mlx-community/Restormer-motion-deblurring-fp32
  single_image_defocus_deblurring-> mlx-community/Restormer-defocus-deblurring-fp32
  real_denoising                 -> mlx-community/Restormer-real-denoising-fp32

Run:  .venv/bin/python publish.py [--dry-run]
"""
import os
import sys

from huggingface_hub import HfApi, add_collection_item

API = HfApi()
HERE = os.path.dirname(os.path.abspath(__file__))

VARIANTS = {
    "motion_deblurring": (
        "Restormer-motion-deblurring-fp32", "WithBias", 494, "26,126,644",
        "Camera-shake and subject-motion deblur. GoPro 32.92.",
        "Motion deblur"),
    "single_image_defocus_deblurring": (
        "Restormer-defocus-deblurring-fp32", "WithBias", 494, "26,126,644",
        "Out-of-focus deblur from a SINGLE image. DPDD 25.98. Notable because the high end of this "
        "task is licence-poisoned — IFAN, KPAC, DRBNet and LaKDNet are all AGPL-3.0 — so this is "
        "0.44 dB off the best and the only permissively-licensed option.",
        "Single-image defocus deblur"),
    "real_denoising": (
        "Restormer-real-denoising-fp32", "BiasFree", 406, "26,111,668",
        "Real-world denoise (SIDD / DND). Carries the only solid **DND 40.03**, i.e. it generalizes "
        "across sensors rather than to SIDD's five smartphone cameras.",
        "Real denoise"),
}


def card(repo, norm, tensors, params, blurb, title):
    biasfree_note = ""
    if norm == "BiasFree":
        biasfree_note = """
## ⚠️ This checkpoint is `BiasFree` — it has a different key set

Restormer's denoising models normalize `LayerNorm_type='BiasFree'`, which has **no bias vectors at
all**: **406 tensors / 26,111,668 params**, against the deblur checkpoints' 494 / 26,126,644. Build
the model with the wrong norm type and the strict load fails with exactly 88 missing keys.

`BiasFree` also does **not centre its output** — it is `x / sqrt(var + eps) · weight`, with the
variance computed about the mean but no mean subtraction. Implementing it as "LayerNorm minus the
bias term" is wrong.
"""
    return f"""---
library_name: mlx
license: mit
license_link: https://github.com/swz30/Restormer/blob/main/LICENSE.md
base_model: swz30/Restormer
pipeline_tag: image-to-image
tags:
  - mlx
  - image-restoration
  - deblurring
  - denoising
  - restormer
---

# mlx-community/{repo}

[Restormer](https://github.com/swz30/Restormer) — **{title}** — converted to **Apple MLX** for
Apple-Silicon inference via the
[`mlx-restormer-swift`](https://github.com/xocialize/mlx-restormer-swift) Swift package.

Zamir et al., *Restormer: Efficient Transformer for High-Resolution Image Restoration*, **CVPR
2022**. **{params} parameters** ({tensors} tensors, `LayerNorm_type={norm}`).

{blurb}

> **Licence note.** Restormer is **plain MIT** with full commercial rights. The *same author's*
> MIRNet / MIRNetv2 / MPRNet / CycleISP carry an **Academic Public License** — easy to conflate,
> so it is worth stating explicitly.

## Use with mlx-restormer-swift

```swift
import RestormerMLXCore

var cfg = Restormer.Configuration()
cfg.normKind = .{"biasFree" if norm == "BiasFree" else "withBias"}
let model = Restormer(cfg)
try model.loadWeights(from: weightsURL)
let restored = model.restoreTiled(imageNHWC)   // NHWC RGB in [0,1]
```

Or as an MLXEngine `imageRestore` ModelPackage (`MLXRestormer.RestormerRestorePackage`), which
resolves this repo via the Hub.
{biasfree_note}
## ⚠️ Tile — do not run this full-frame

Measured on an M5 Max: a single 1080p frame full-frame costs **15.50 GB of MLX allocation / 48.02 GB
process footprint**. Tiled, the MLX peak is **flat at ~2.6 GB** from 512² to 1080p, because the peak
is one-tile-sized — a bigger image runs *more* tiles, not bigger ones.

**Tile geometry should be 8-aligned.** Three `pixelUnshuffle(2)` stages mean the 2×2 grouping grid is
measured from the tile origin at strides 2, 4 and 8 full-resolution pixels, so an unaligned origin
shifts that grid out of phase with the full-frame decomposition.

On overlap, measured by **seam visibility** rather than PSNR: overlap 0 leaves a faint but real seam
(boundary gradient 1.31× the interior), while every aligned overlap ≥ 8 measures clean (1.09–1.20×).
Note PSNR-against-full-frame actually *prefers* overlap 0 — which is why it is the wrong metric for
a tiler.

## Conversion

MLX **NHWC** layout: conv `(O,I,kH,kW) → (O,kH,kW,I)` (186 standard + 88 depthwise), everything else
passthrough. The upstream `.pth` nests its state dict under `raw['params']` (BasicSR convention).

## Parity

Gated against the PyTorch oracle on the CPU stream, fp32, judged on relative error:

- **key contract** — {tensors} tensors / {params} params / 0 missing / 0 unused, strict load
- **primitives** — both LayerNorm variants, and the pixel-shuffle pair (worst 2.6e-06)
- **blocks** — MDTA, GDFN, TransformerBlock (worst 2.6e-07)
- **full model** — cosine **1.00000000** at 64² / 128² / 256² (worst 1.0e-06)

The pixel-shuffle gate also runs the *wrong* channel ordering — the `(r,r,C)` split that compiles
and silently scrambles the image — and measures it at **493,451× the observed rounding**, so the
tolerance keeps ~5 orders of discrimination.

## Honest limitation

Restormer's attention is spatially **global**, so it is weak on *spatially varying* blur. Benchmark
rank is also a weak predictor of real-world value: *"Deblurring in the Wild"* (2026) found every
GoPro-trained method scored below the blurry input on real smartphone blur. Validate on your own
photographs.

Weights: MIT (swz30/Restormer, first-party GitHub release v1.0). Port code: MIT.
"""


def main():
    dry = "--dry-run" in sys.argv
    published = []
    for stem, (repo, norm, tensors, params, blurb, title) in VARIANTS.items():
        rid = f"mlx-community/{repo}"
        weights = os.path.join(HERE, "converted", stem, "model.safetensors")
        if not os.path.exists(weights):
            print(f"[skip] {rid}: missing {weights}")
            continue
        if dry:
            print(f"[dry-run] would upload {weights} -> {rid}")
            continue
        print(f"[publish] {rid} ({os.path.getsize(weights)/1e6:.2f} MB) …")
        API.create_repo(rid, repo_type="model", exist_ok=True)
        API.upload_file(path_or_fileobj=weights, path_in_repo="model.safetensors", repo_id=rid)
        API.upload_file(path_or_fileobj=card(repo, norm, tensors, params, blurb, title).encode(),
                        path_in_repo="README.md", repo_id=rid)
        published.append((rid, title + " — " + blurb))
        print(f"[publish]   ok → https://huggingface.co/{rid}")

    if published:
        col = API.create_collection(
            title="Restormer (MLX)", namespace="mlx-community", exists_ok=True,
            # HF caps this at 150 characters.
            description="Apple-MLX Restormer (CVPR 2022, MIT): motion deblur, defocus deblur, "
                        "real denoise. Tile it — full-frame 1080p needs 48 GB.")
        for rid, note in published:
            try:
                add_collection_item(col.slug, item_id=rid, item_type="model", note=note[:500])
            except Exception as e:
                print(f"[collection]   {rid}: {str(e)[:70]}")
        print(f"[collection] ok → https://huggingface.co/collections/{col.slug}")


if __name__ == "__main__":
    main()
