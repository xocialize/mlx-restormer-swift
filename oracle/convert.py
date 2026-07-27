"""P10 weight conversion: Restormer .pth -> safetensors in MLX NHWC layout.

Two things, one of which differs from the sibling FFTformer port:

1. **Unwrap `raw['params']`.** Restormer follows the BasicSR convention of nesting the state dict;
   FFTformer's release was a flat dict. Getting this wrong yields zero matching keys.
2. Conv weight `(O, I, kH, kW) -> (O, kH, kW, I)`; depthwise `(C,1,3,3) -> (C,3,3,1)`, which is what
   MLX `Conv2d(groups: C)` wants.

`temperature` is `(heads, 1, 1)` and passes through untouched, as do the LayerNorm vectors.

Run:  .venv/bin/python convert.py [stem ...]
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file

STEMS = sys.argv[1:] or ["motion_deblurring", "single_image_defocus_deblurring"]

for stem in STEMS:
    src = f"weights/{stem}.pth"
    if not os.path.exists(src):
        print(f"[skip] {src} not present")
        continue

    raw = torch.load(src, map_location="cpu", weights_only=False)
    sd = raw["params"] if isinstance(raw, dict) and "params" in raw else raw
    wrapper = "raw['params']" if (isinstance(raw, dict) and "params" in raw) else "flat"

    out_dir = os.path.join("converted", stem)
    os.makedirs(out_dir, exist_ok=True)

    converted, stats = {}, {"conv": 0, "depthwise": 0, "passthrough": 0}
    for k, v in sd.items():
        a = v.detach().cpu().numpy().astype(np.float32)
        if a.ndim == 4:
            depthwise = a.shape[1] == 1 and a.shape[0] > 1
            a = np.transpose(a, (0, 2, 3, 1))
            stats["depthwise" if depthwise else "conv"] += 1
        else:
            stats["passthrough"] += 1
        converted[k] = np.ascontiguousarray(a)

    total = sum(int(np.prod(v.shape)) for v in converted.values())
    print(f"=== {stem} ===")
    print(f"  wrapper: {wrapper}   tensors: {len(converted)}")
    print(f"  transforms: conv {stats['conv']} · depthwise {stats['depthwise']} "
          f"· passthrough {stats['passthrough']}")
    print(f"  params: {total:,}  ({total * 4 / 1e6:.2f} MB fp32)")

    meta = {"format": "pt", "source": f"swz30/Restormer {stem}.pth", "license": "MIT",
            "layout": "MLX NHWC; conv (O,kH,kW,I)", "params": str(total)}
    save_file(converted, os.path.join(out_dir, "model.safetensors"), metadata=meta)
    with open(os.path.join(out_dir, "CONVERSION.json"), "w") as f:
        json.dump({"stem": stem, "wrapper": wrapper, "transforms": stats, "params": total}, f, indent=2)
    sz = os.path.getsize(os.path.join(out_dir, "model.safetensors")) / 1e6
    print(f"  written: {out_dir}/model.safetensors ({sz:.2f} MB)\n")
