"""P10 oracle — per-sub-op goldens for the Restormer Swift port.

fp32, CPU-torch, numpy-seeded, C-contiguous, PyTorch NCHW. The Swift gate transposes to NHWC,
runs, transposes back, compares.

Covers BOTH LayerNorm variants: the deblur checkpoints use WithBias, the denoising ones use
BiasFree, so unlike the sibling FFTformer port neither is dead code.

Run:  .venv/bin/python gen_goldens.py [checkpoint-stem]
Out:  goldens/*.npy  +  goldens/MANIFEST.txt
"""
import importlib.util
import os
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)

_s = importlib.util.spec_from_file_location(
    "restormer_arch", "upstream/basicsr/models/archs/restormer_arch.py")
A = importlib.util.module_from_spec(_s)
_s.loader.exec_module(A)

STEM = sys.argv[1] if len(sys.argv) > 1 else "motion_deblurring"
OUT = "goldens"
os.makedirs(OUT, exist_ok=True)

raw = torch.load(f"weights/{STEM}.pth", map_location="cpu", weights_only=False)
sd = raw["params"] if isinstance(raw, dict) and "params" in raw else raw
model = A.Restormer(LayerNorm_type="WithBias")
model.load_state_dict(sd, strict=True)
model.eval()

manifest = []


def save(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    np.save(os.path.join(OUT, name + ".npy"), a)
    manifest.append(f"{name + '.npy':40s} {str(a.shape):24s} "
                    f"min={a.min():+.6f} max={a.max():+.6f} mean={a.mean():+.6f}")
    print(f"  saved {name}.npy  {a.shape}")


def dump(name, t):
    save(name, t.detach().cpu().numpy())


def seeded(seed, *shape):
    g = np.random.default_rng(seed)
    return torch.from_numpy(g.standard_normal(shape, dtype=np.float32))


print(f"\n=== checkpoint: {STEM} ===")

print("\n=== 1. LayerNorm — BOTH variants (deblur uses WithBias, denoise uses BiasFree) ===")
xl = seeded(7001, 1, 48, 32, 32)
dump("layernorm_in", xl)
dump("layernorm_withbias_out", model.encoder_level1[0].norm1(xl))

# BiasFree is NOT dead here. Note it does NOT subtract the mean — it is x / sqrt(var) * weight,
# with the variance computed about the mean but the output left uncentered.
bf = A.LayerNorm(48, "BiasFree")
bf.body.weight.data = model.encoder_level1[0].norm1.body.weight.data.clone()
dump("layernorm_biasfree_w", bf.body.weight)
dump("layernorm_biasfree_out", bf(xl))

print("\n=== 2. Downsample / Upsample (PixelUnshuffle / PixelShuffle, NOT bilinear) ===")
xd = seeded(7002, 1, 48, 32, 32)
dump("down_in", xd)
dump("down_out", model.down1_2(xd))          # 48 -> 96 ch, /2

xu = seeded(7003, 1, 384, 8, 8)
dump("up_in", xu)
dump("up_out", model.up4_3(xu))              # 384 -> 192 ch, x2

print("\n=== 3. MDTA — channel-dim self-attention with a learned per-head temperature ===")
attn = model.encoder_level1[0].attn
xa = seeded(7004, 1, 48, 16, 16)
dump("mdta_in", xa)
dump("mdta_out", attn(xa))
dump("mdta_temperature", attn.temperature)

print("\n=== 4. GDFN feed-forward ===")
ff = model.encoder_level1[0].ffn
xf = seeded(7005, 1, 48, 16, 16)
dump("gdfn_in", xf)
dump("gdfn_out", ff(xf))

print("\n=== 5. TransformerBlock ===")
xt = seeded(7006, 1, 48, 16, 16)
dump("tblock_in", xt)
dump("tblock_out", model.encoder_level1[0](xt))

print("\n=== 6. Full model ===")
for size in (64, 128, 256):
    xi = torch.from_numpy(
        np.random.default_rng(8000 + size).random((1, 3, size, size), dtype=np.float32))
    dump(f"full_{size}_in", xi)
    dump(f"full_{size}_out", model(xi))

print("\n=== 7. Full model on a structured image (production-like tile) ===")
g = np.random.default_rng(9001)
yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / 255.0
img = np.stack([
    0.5 + 0.35 * np.sin(xx * 9) * np.cos(yy * 7),
    0.5 + 0.35 * np.cos(xx * 6 + yy * 5),
    0.5 + 0.30 * np.sin((xx + yy) * 11),
])[None]
img = np.clip(img + 0.02 * g.standard_normal(img.shape, dtype=np.float32), 0, 1)
dump("full_img256_in", torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)))
dump("full_img256_out", model(torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))))

with open(os.path.join(OUT, "MANIFEST.txt"), "w") as f:
    f.write("Restormer PyTorch goldens — fp32, CPU, PyTorch NCHW, C-contiguous.\n")
    f.write(f"checkpoint: weights/{STEM}.pth  (state dict under raw['params'])\n")
    f.write("constructor: Restormer(LayerNorm_type='WithBias')  dim=48 num_blocks=[4,6,6,8]\n")
    f.write("             num_refinement_blocks=4 heads=[1,2,4,8] ffn=2.66 bias=False\n")
    f.write("input contract: RGB [0,1]; pad to a multiple of 8 (three /2 stages).\n\n")
    f.write("\n".join(manifest) + "\n")

print(f"\n✅ {len(manifest)} goldens written to {OUT}/")
