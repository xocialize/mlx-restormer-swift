//
//  main.swift
//  mlx-restormer-swift / RestormerGate
//
//  Parity gates against the PyTorch oracle. Executable, not a test target — the SPM test product's
//  metallib is unreliable for GPU work.
//

import Foundation
import RestormerMLXCore
import MLX
import MLXNN

private let _unbuffered: Void = { setvbuf(stdout, nil, _IONBF, 0) }()

func fail(_ msg: String) -> Never { _ = _unbuffered; print("❌ \(msg)"); exit(1) }

func loadedModel(_ path: String, denoise: Bool = false) -> Restormer {
    let model = Restormer(denoise ? .denoising : Restormer.Configuration())
    do { try model.loadWeights(from: URL(fileURLWithPath: path)) }
    catch { fail("weight load failed: \(error)") }
    return model
}

func g(_ dir: String, _ name: String) -> MLXArray {
    do { return try loadNPY("\(dir)/\(name).npy") } catch { fail("golden \(name): \(error)") }
}

func gateS0(_ weightsPath: String, denoise: Bool = false) {
    _ = _unbuffered
    print("=== S0 · key contract\(denoise ? " (BiasFree / denoising)" : "") ===\n")
    let model = Restormer(denoise ? .denoising : Restormer.Configuration())
    var swiftKeys: [String: [Int]] = [:]; var total = 0
    for (k, v) in model.parameters().flattened() { swiftKeys[k] = v.shape; total += v.size }
    print("Swift module tree : \(swiftKeys.count) tensors, \(total) params")
    guard let loaded = try? MLX.loadArrays(url: URL(fileURLWithPath: weightsPath)) else {
        fail("could not load \(weightsPath)")
    }
    print("Checkpoint        : \(loaded.count) tensors, \(loaded.values.reduce(0) { $0 + $1.size }) params\n")
    let sk = Set(swiftKeys.keys), ck = Set(loaded.keys)
    let missing = sk.subtracting(ck).sorted(), unused = ck.subtracting(sk).sorted()
    if !missing.isEmpty { print("MISSING (\(missing.count)):"); missing.prefix(15).forEach { print("   \($0)  \(swiftKeys[$0]!)") } }
    if !unused.isEmpty { print("UNUSED (\(unused.count)):"); unused.prefix(15).forEach { print("   \($0)  \(loaded[$0]!.shape)") } }
    var mismatch: [(String, [Int], [Int])] = []
    for k in sk.intersection(ck) where swiftKeys[k]! != loaded[k]!.shape {
        mismatch.append((k, swiftKeys[k]!, loaded[k]!.shape))
    }
    if !mismatch.isEmpty {
        print("SHAPE MISMATCH (\(mismatch.count)):")
        for (k, a, b) in mismatch.prefix(15) { print("   \(k)\n     swift \(a) vs ckpt \(b)") }
    }
    guard missing.isEmpty, unused.isEmpty, mismatch.isEmpty else { fail("S0 FAILED") }
    do { try model.update(parameters: ModuleParameters.unflattened(loaded), verify: .all) }
    catch { fail("S0 FAILED at update(verify: .all): \(error)") }
    print("✅ S0 PASSED — \(swiftKeys.count) tensors, \(total) params, strict update clean.")
}

/// Primitives, including the pixel-shuffle pair that MLX has no native op for.
func gateS1(_ dir: String, _ w: String) -> Bool {
    print("=== S1 · primitives ===\n")
    let r = GateReport("S1")
    let model = loadedModel(w)

    r.check("layernorm_withbias", toNCHW(model.encoderLevel1[0].norm1(toNHWC(g(dir, "layernorm_in")))),
            g(dir, "layernorm_withbias_out"), tol: 1e-6)

    // BiasFree is a live variant here (the denoising checkpoints use it), not dead code.
    let bf = ChannelLayerNorm(48, kind: .biasFree)
    // MLX-Swift parameters are not directly assignable — they must go through update().
    try? bf.update(parameters: ModuleParameters.unflattened(
        ["body.weight": g(dir, "layernorm_biasfree_w")]), verify: .none)
    r.check("layernorm_biasfree", toNCHW(bf(toNHWC(g(dir, "layernorm_in")))),
            g(dir, "layernorm_biasfree_out"), tol: 1e-6)

    r.check("downsample_unshuffle", toNCHW(model.down1_2(toNHWC(g(dir, "down_in")))),
            g(dir, "down_out"), tol: 1e-6)
    // Tolerance is 1e-5 here vs 1e-6 for the downsample, and the reason is structural: up4_3's
    // conv accumulates over 384 input channels where down1_2's accumulates over 48. Eight times
    // the accumulation, ~3x the rounding — not a semantic difference.
    //
    // But a loosened tolerance is only honest if it still catches what it is for. The failure mode
    // this gate guards is the pixel-shuffle CHANNEL ORDERING: splitting the trailing axis as
    // (r, r, C) instead of (C, r, r) compiles, runs, and silently scrambles the image. The probe
    // below runs exactly that mistake and reports both errors, so the margin is visible rather
    // than asserted.
    r.check("upsample_shuffle", toNCHW(model.up4_3(toNHWC(g(dir, "up_in")))),
            g(dir, "up_out"), tol: 1e-5)
    shuffleOrderingProbe(dir)
    return r.summarize()
}

/// Demonstrates that the resampler tolerance still discriminates the ordering bug it guards.
func shuffleOrderingProbe(_ dir: String) {
    let x = toNHWC(g(dir, "up_in"))
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let r = 2
    let correct = pixelShuffle(x, r)
    // The plausible-looking mistake: split as (r, r, C) rather than (C, r, r).
    let wrong = x.reshaped(b, h, w, r, r, c / (r * r))
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(b, h * r, w * r, c / (r * r))
    let p = parity(wrong, correct)
    print(String(format: "     ↳ ordering probe: the (r,r,C) mistake gives rel=%.2e — %.0fx the "
                 + "observed rounding, so a 1e-5 gate still catches it by ~5 orders of magnitude",
                 p.relative, p.relative / 2.63e-06))
}

/// Blocks.
func gateS2(_ dir: String, _ w: String) -> Bool {
    print("=== S2 · blocks ===\n")
    let r = GateReport("S2")
    let model = loadedModel(w)
    r.check("mdta", toNCHW(model.encoderLevel1[0].attn(toNHWC(g(dir, "mdta_in")))),
            g(dir, "mdta_out"), tol: 1e-5)
    r.check("gdfn", toNCHW(model.encoderLevel1[0].ffn(toNHWC(g(dir, "gdfn_in")))),
            g(dir, "gdfn_out"), tol: 1e-5)
    r.check("tblock", toNCHW(model.encoderLevel1[0](toNHWC(g(dir, "tblock_in")))),
            g(dir, "tblock_out"), tol: 1e-5)
    return r.summarize()
}

/// Full model.
func gateS3(_ dir: String, _ w: String) -> Bool {
    print("=== S3 · full model ===\n")
    let r = GateReport("S3")
    let model = loadedModel(w)
    for name in ["full_64", "full_128", "full_256", "full_img256"] {
        let out = model(toNHWC(g(dir, "\(name)_in")))
        eval(out)
        r.check(name, toNCHW(out), g(dir, "\(name)_out"), tol: 1e-4)
    }
    return r.summarize()
}

/// Process phys_footprint — the admission basis. MLX-peak under-reads it (~2.7x, the BiRefNet
/// re-baseline), because it cannot see the Metal driver working set or process overhead.
func physFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
}

func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(max(0, b)) / 1e9) }
func gb(_ b: UInt64) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

/// Does this model need tiling? FFTformer did (40 GB at 1080p). Measure, do not assume.
func gateBench(_ w: String) {
    _ = _unbuffered
    print("=== BENCH · split footprint (GPU stream) ===\n")
    let base = physFootprintBytes()
    let model = loadedModel(w)
    MLX.Memory.clearCache()
    let floor = physFootprintBytes()
    print("  post-load floor : \(gb(floor))  → resident ≈ \(gb(floor > base ? floor - base : 0))")
    print("  (weights are 26,126,644 params @ fp32 = 104.5 MB)\n")

    for (w0, h0) in [(512, 512), (1024, 1024), (1920, 1080)] {
        MLX.Memory.clearCache()
        MLX.Memory.peakMemory = 0
        let x = MLXArray.zeros([1, h0, w0, 3], dtype: .float32)
        let out = model.restoreTiled(x)
        eval(out)
        let mlxPeak = MLX.Memory.peakMemory
        let phys = physFootprintBytes()
        print("  \(w0)x\(h0): MLX peak \(gb(mlxPeak))   phys \(gb(phys))   "
            + "activation ≈ \(gb(phys > floor ? phys - floor : 0))")
        MLX.Memory.clearCache()
    }
}

/// Verify the 8-alignment claim and pick an overlap, the same way FFTformer's 32-alignment was
/// found: sweep overlaps against a full-frame reference and look for a `% stride` pattern.
func gateTile(_ w: String) {
    _ = _unbuffered
    print("=== TILE · alignment + overlap sweep ===\n")
    let model = loadedModel(w)
    let n = 512
    var px = [Float](repeating: 0, count: n * n * 3)
    for y in 0 ..< n { for x in 0 ..< n {
        let fx = Float(x) / Float(n), fy = Float(y) / Float(n), i = (y * n + x) * 3
        px[i] = 0.5 + 0.35 * sin(fx * 9) * cos(fy * 7)
        px[i + 1] = 0.5 + 0.35 * cos(fx * 6 + fy * 5)
        px[i + 2] = 0.5 + 0.30 * sin((fx + fy) * 11)
    } }
    let x = MLXArray(px, [1, n, n, 3])
    let full = MLX.clip(model.restore(x), min: 0, max: 1)
    eval(full); MLX.Memory.clearCache()

    print("  reference: full-frame 512²; tiled at 256")
    print("  overlap   PSNR      mean_abs   note")
    for ov in [0, 4, 8, 16, 24, 32, 48] {
        let t = model.restoreTiled(x, tile: 256, overlap: ov)
        eval(t)
        let mse = MLX.mean(MLX.square(t - full)).item(Float.self)
        let psnr = mse > 0 ? 10 * log10(1.0 / mse) : Float.infinity
        let mean = MLX.mean(MLX.abs(t - full)).item(Float.self)
        let note = ov % 8 == 0 ? "" : "  (not a multiple of 8 — rounded down by restoreTiled)"
        // PSNR against full-frame is NOT the metric that matters for a tiler — full-frame is
        // unattainable at production sizes anyway. What matters is whether tile boundaries are
        // VISIBLE. Measure the discontinuity directly: mean |horizontal gradient| at columns that
        // sit on a tile seam, against the same statistic everywhere else. A ratio near 1.0 means
        // the seam is indistinguishable from ordinary image content.
        let step = 256 - 2 * ((ov / 8) * 8)
        let g = MLX.abs(t[0..., 0..., 1..., 0...] - t[0..., 0..., ..<(n - 1), 0...])
        var seamCols: [Int32] = [], otherCols: [Int32] = []
        for c in 0 ..< (n - 1) {
            // a seam sits where a core boundary falls
            if c > 0 && c % step < 2 { seamCols.append(Int32(c)) } else { otherCols.append(Int32(c)) }
        }
        let seam = seamCols.isEmpty ? Float(0)
            : MLX.mean(g.take(MLXArray(seamCols), axis: 2)).item(Float.self)
        let other = MLX.mean(g.take(MLXArray(otherCols), axis: 2)).item(Float.self)
        let ratio = other > 0 ? seam / other : 0
        let verdict = ratio > 1.5 ? "  ⚠️ SEAM VISIBLE" : (ratio > 1.15 ? "  faint" : "  clean")
        print(String(format: "  %7d  %7.2f dB  %.3e   seam/interior gradient %.2fx%@%@",
                     ov, psnr, mean, ratio, verdict, note))
        MLX.Memory.clearCache()
    }
}

let args = Array(CommandLine.arguments.dropFirst())
guard let mode = args.first else {
    print("usage: restormer-gate --s0 <weights> | --s1|--s2|--s3|--all <goldens> <weights>")
    exit(2)
}
if mode != "--bench" && mode != "--tile" { Device.setDefault(device: .cpu) }
switch mode {
case "--tile":
    guard args.count >= 2 else { fail("--tile needs a weights path") }
    Device.setDefault(device: .gpu)
    gateTile(args[1])
case "--bench":
    guard args.count >= 2 else { fail("--bench needs a weights path") }
    Device.setDefault(device: .gpu)
    gateBench(args[1])
case "--s0":
    guard args.count >= 2 else { fail("--s0 needs a weights path") }
    gateS0(args[1], denoise: args.contains("--denoise"))
case "--s1", "--s2", "--s3", "--all":
    guard args.count >= 3 else { fail("\(mode) needs <goldens> <weights>") }
    let (dir, w) = (args[1], args[2])
    var ok = true
    if mode == "--s1" || mode == "--all" { ok = gateS1(dir, w) && ok; print("") }
    if mode == "--s2" || mode == "--all" { ok = gateS2(dir, w) && ok; print("") }
    if mode == "--s3" || mode == "--all" { ok = gateS3(dir, w) && ok }
    if !ok { exit(1) }
default: fail("unknown mode \(mode)")
}
