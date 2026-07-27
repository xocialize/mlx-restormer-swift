"""P10 Stage 0: do the released checkpoints load strict=True, and what is the real param count?"""
import importlib.util, torch

torch.set_grad_enabled(False)
_s = importlib.util.spec_from_file_location(
    "restormer_arch", "upstream/basicsr/models/archs/restormer_arch.py")
A = importlib.util.module_from_spec(_s); _s.loader.exec_module(A)

for name, ln in [("motion_deblurring", "WithBias"),
                 ("single_image_defocus_deblurring", "WithBias")]:
    raw = torch.load(f"weights/{name}.pth", map_location="cpu", weights_only=False)
    sd = raw.get("params", raw) if isinstance(raw, dict) else raw
    wrapper = "raw['params']" if (isinstance(raw, dict) and "params" in raw) else "flat"
    m = A.Restormer(LayerNorm_type=ln)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    n = sum(p.numel() for p in m.parameters())
    nb = sum(p.numel() * p.element_size() for p in m.parameters())
    try:
        m.load_state_dict(sd, strict=True); strict = "✅ CLEAN"
    except Exception as e:
        strict = f"❌ {str(e)[:90]}"
    m.eval()
    x = torch.rand(1, 3, 128, 128)
    y = m(x)
    print(f"=== {name} (LayerNorm={ln}) ===")
    print(f"   wrapper      : {wrapper}   tensors: {len(sd)}")
    print(f"   params       : {n:,}  = {nb/1e6:.2f} MB fp32 / {nb/2e6:.2f} MB fp16")
    print(f"   strict load  : {strict}   (missing {len(missing)}, unexpected {len(unexpected)})")
    print(f"   forward      : {tuple(x.shape)} -> {tuple(y.shape)}  range [{y.min():.4f}, {y.max():.4f}]")
